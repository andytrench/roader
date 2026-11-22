"""Stages 3 to 9: points and vectors in, the section 5 parametric model out.

``build_model`` is the single entry point. It is deterministic: the same inputs
produce a byte-identical ``roader.model.dumps`` (AC-H5, FR-PL-3). Points flagged
in ``holdout`` are removed before any stage sees them (D-018), so AC-A1 can be
measured on them afterwards.

Order: vector graph (FR-VC-1) -> evidence on the corridor (FR-PN-2/3, FR-RS-1..4)
-> per road: sheet tracking, registration, L1, L2, L3 (FR-VC-2/3, FR-AF-1..5,
FR-XS-1..3) -> crossings (FR-CR-1/2) -> structure ends (FR-VC-4, FR-CR-3) ->
barriers and structures (FR-RS-4, FR-ST-2..6, FR-XS-5) -> completion and
clearance (FR-OC-1..3) -> earthwork and terrain (L4, L7) -> unmatched sheets
(FR-VC-5).

Not yet: joint fitting across links (FR-AF-3; no links in milestone 1),
inventory matching (FR-ST-1), observed piers, median classification (FR-XS-4),
multiple collections (FR-IN-4).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from roader.complete import (
    check_clearances,
    label_under_deck,
    side_slopes,
    split_at_structures,
    terrain_patch,
)
from roader.conflate.crossings import CrossingMeasure, find_crossings
from roader.conflate.graph import data_extent, select_roads
from roader.conflate.register import cross_profiles, near_points
from roader.conflate.structures import StructureExtent, find_extents, side_drop
from roader.conflate.track import track_line
from roader.evidence import Evidence, build_evidence
from roader.evidence.strips import STRIP_TOL_M
from roader.fit.road import RoadFit, fit_road
from roader.fit.section import barriers_from_evidence
from roader.io.points import PointSet
from roader.io.vectors import VectorEdge
from roader.model.provenance import Source
from roader.model.schema import Flag, Frame, RoadModel, UnitRecord
from roader.rules import RuleBook
from roader.structures import infer_structure, parapet_fill
from roader.structures.infer import _footprint_on

UNMATCHED_SHEET_CELLS = 400  # FR-VC-5: paved sheets this large with no road are reported
RULE_EXTENT_PAD_M = 1.0
DECK_LEVEL_M = 2.0


@dataclass
class PipelineResult:
    model: RoadModel
    evidence: Evidence
    fits: list[RoadFit]
    crossing_measures: list[CrossingMeasure] = field(default_factory=list)
    extents: list[StructureExtent] = field(default_factory=list)

    def measured(self) -> dict:
        """Pipeline measurements in the shape ``roader.validate.run_gates`` takes."""
        return {
            "crossing_dz": {m.crossing_id: m.dz_data_m for m in self.crossing_measures if m.dz_data_m is not None},
        }


def _flatten(vectors) -> list[VectorEdge]:
    if isinstance(vectors, Mapping):
        return [e for name in sorted(vectors) for e in vectors[name]]
    return list(vectors)


def run_pipeline(
    points: PointSet,
    vectors: Sequence[VectorEdge] | Mapping[str, Sequence[VectorEdge]],
    frame: Frame,
    unit: UnitRecord,
    rules: RuleBook | None = None,
    *,
    holdout: np.ndarray | None = None,
) -> PipelineResult:
    rules = rules or RuleBook.load()
    edges = _flatten(vectors)
    extent = data_extent(points.xyz[:, :2])
    fit_points = points if holdout is None else points.subset(~np.asarray(holdout, dtype=bool))
    seeds = select_roads(edges, extent)
    flags: list[Flag] = []
    if not seeds:
        raise ValueError("no in-scope roads in the vectors")

    ev = build_evidence(fit_points, [s.line.xy for s in seeds])
    strip_factor = 1.0
    if ev.strips.p95_m is not None and ev.strips.p95_m > STRIP_TOL_M and not ev.strips.corrected:
        strip_factor = 0.8
        flags.append(Flag("FR-PN-2:strips", unit.unit_id, "FR-PN-2", f"strip disagreement p95 {ev.strips.p95_m:.3f} m; confidence lowered"))
    for code in ev.strips.corrected:
        flags.append(Flag("FR-PN-2:corrected", unit.unit_id, "FR-PN-2", f"strip {code} shifted by {-ev.strips.offsets_m[code]:+.3f} m"))

    fits = [fit_road(seed, ev, rules, strip_factor=strip_factor) for seed in seeds]
    by_id = {f.road.id: f for f in fits}
    held_blocks = _held_blocks(points, holdout)
    for f in fits:
        withheld = _mark_withheld(f.road, held_blocks)
        flags.extend(fl for fl in f.flags if not (fl.code == "FR-VC-2:unsupported" and any(f"s={a:.1f}-{b:.1f}" in fl.message for a, b in withheld)))

    crossings, measures, cflags = find_crossings(fits)
    flags.extend(cflags)

    # Structure ends from data, then a rule extent for any crossing left uncovered.
    extents: list[StructureExtent] = []
    drops = {}
    for f in fits:
        drops[f.road.id] = side_drop(f, ev)
        skews = [(c.upper_s, c.skew_rad) for c in crossings if c.upper_road == f.road.id]
        extents.extend(find_extents(f, ev, skews))
    for c in sorted(crossings, key=lambda c: c.id):
        if any(e.road_id == c.upper_road and e.s_start - 1.0 <= c.upper_s <= e.s_end + 1.0 for e in extents):
            continue
        lateral, _ = rules.fill("clearance.lateral.support", road_class=by_id[c.lower_road].road.road_class)
        a, b = _footprint_on(by_id[c.upper_road], by_id[c.lower_road], c, lateral + RULE_EXTENT_PAD_M)
        extents.append(StructureExtent(c.upper_road, a, b, "rule", None, None))
        flags.append(Flag("FR-VC-4:not_found", c.id, "FR-VC-4", "no structure ends found in the data; span set from the crossed road and lateral clearance rule"))
    extents.sort(key=lambda e: (e.road_id, e.s_start))

    barriers = {}
    for f in fits:
        ranges = [(e.s_start, e.s_end) for e in extents if e.road_id == f.road.id]
        xs = f.road.cross_section
        barriers[f.road.id] = barriers_from_evidence(f.road.id, f.line, ev, xs.width_left, xs.width_right, structure_ranges=ranges, rules=rules)

    structures = []
    for e in extents:
        res = infer_structure(e, by_id[e.road_id], crossings, by_id, ev, rules, drops[e.road_id], barriers[e.road_id])
        if e.method == "rule":
            res.structure.prov.source = Source.RULE
            res.structure.prov.rule_id = "clearance.lateral.support"
            res.structure.prov.confidence = 0.3
        structures.append(res.structure)
        flags.extend(res.flags)
        filled = parapet_fill(by_id[e.road_id].road, res.structure, barriers[e.road_id], rules)
        barriers[e.road_id].extend(filled)
        _fit_overhang(by_id[e.road_id].road, res.structure, barriers[e.road_id])

    for f in fits:
        f.road.cross_section.barriers = sorted(barriers[f.road.id], key=lambda b: (b.side, b.s0, b.id))

    slopes = []
    for f in fits:
        slopes.extend(side_slopes(f, structures, rules, strip_factor))

    unit = copy.deepcopy(unit)
    for rec in unit.lidar:
        rec.density_pavement_pts_m2 = ev.density.median_pts_m2
        rec.strip_dz_p95_m = ev.strips.p95_m

    model = RoadModel(
        unit=unit,
        frame=frame,
        roads=sorted([f.road for f in fits], key=lambda r: r.id),
        crossings=sorted(crossings, key=lambda c: c.id),
        structures=sorted(structures, key=lambda s: s.id),
        side_slopes=sorted(slopes, key=lambda s: s.id),
        terrain=terrain_patch(ev, structures, by_id),
    )
    for road in model.roads:
        if any(st.carried_road == road.id for st in model.structures):
            split_at_structures(road, model.structures)
        label_under_deck(road, model)
    flags.extend(check_clearances(model, rules))
    flags.extend(_unmatched_sheets(ev, fits))
    model.flags = sorted(flags, key=lambda f: (f.req, f.code, f.element_id, f.message))
    return PipelineResult(model, ev, fits, measures, extents)


def build_model(
    points: PointSet,
    vectors: Sequence[VectorEdge] | Mapping[str, Sequence[VectorEdge]],
    frame: Frame,
    unit: UnitRecord,
    rules: RuleBook | None = None,
    *,
    holdout: np.ndarray | None = None,
) -> RoadModel:
    """The section 5 model of one unit. ``holdout``: boolean per point, True = withheld (D-018)."""
    return run_pipeline(points, vectors, frame, unit, rules, holdout=holdout).model


def _fit_overhang(road, structure, barriers) -> None:
    """Every barrier on the deck must stand on the deck: widen the overhang if one would hang off it."""
    need = 0.0
    for b in barriers:
        if b.s1 < structure.s_start or b.s0 > structure.s_end:
            continue
        piece = road.cross_section.width_left if b.side == "left" else road.cross_section.width_right
        need = max(need, b.offset_m + b.base_width_m - piece(0.5 * (structure.s_start + structure.s_end)) + 0.05)
    if need > structure.deck.overhang_m:
        structure.deck.overhang_m = need
        structure.deck.prov.flags.append("FR-XS-5:overhang_fits_barriers")


def _held_blocks(points: PointSet, holdout: np.ndarray | None) -> set[tuple[int, int]]:
    if holdout is None or not np.any(holdout):
        return set()
    from roader.validate.accuracy import BLOCK_M

    blocks = np.unique(np.floor(points.xyz[np.asarray(holdout, dtype=bool), :2] / BLOCK_M).astype(np.int64), axis=0)
    return {(int(a), int(b)) for a, b in blocks}


def _mark_withheld(road, held: set[tuple[int, int]]) -> list[tuple[float, float]]:
    """Unsupported intervals inside held-out blocks get reason 'withheld' (D-018); returns their ranges."""
    from roader.validate.accuracy import BLOCK_M

    out = []
    for iv in road.support:
        if iv.kind != "unsupported" or not held:
            continue
        samples = np.linspace(iv.s0, iv.s1, 5)
        cells = [tuple(np.floor(np.array(road.alignment.offset_point(float(s), 0.0)) / BLOCK_M).astype(int).tolist()) for s in samples]
        if sum(c in held for c in cells) >= 3:
            iv.reason = "withheld for validation (D-018)"
            out.append((iv.s0, iv.s1))
    return out


def _unmatched_sheets(ev: Evidence, fits: list[RoadFit]) -> list[Flag]:
    """FR-VC-5: large paved sheets no road runs on, reported as candidates with geometry."""
    used = {int(s) for f in fits for s in f.track.sheet if s >= 0}
    out = []
    for sh in ev.sheets:
        if sh.index in used or sh.pavement_cells < UNMATCHED_SHEET_CELLS:
            continue
        out.append(
            Flag(
                "FR-VC-5:candidate",
                f"sheet_{sh.index}",
                "FR-VC-5",
                f"paved sheet without a vector: {sh.pavement_cells} m2 paved, centroid ({sh.centroid[0]:.1f}, {sh.centroid[1]:.1f}), "
                f"z {sh.z_min:.1f}-{sh.z_max:.1f}",
            )
        )
    return out


# ---------------------------------------------------------------- measurement helpers for tests and reports


def pavement_points(points: PointSet, vectors) -> np.ndarray:
    """(N, 3) returns labelled pavement or deck by the evidence stage (labelling only, no fitting)."""
    seeds = select_roads(_flatten(vectors), data_extent(points.xyz[:, :2]))
    ev = build_evidence(points, [s.line.xy for s in seeds])
    return ev.xyz[ev.pavement_mask()]


def holdout_for(points: PointSet, pavement_xyz: np.ndarray) -> np.ndarray:
    """Point mask of the D-018 holdout blocks chosen on the pavement returns, applied to every point."""
    from roader.validate.accuracy import BLOCK_M, holdout_mask

    held = holdout_mask(pavement_xyz[:, :2])
    blocks = {tuple(b) for b in np.floor(pavement_xyz[held, :2] / BLOCK_M).astype(np.int64).tolist()}
    pb = np.floor(points.xyz[:, :2] / BLOCK_M).astype(np.int64)
    key = pb[:, 0] * 1_000_003 + pb[:, 1]
    wanted = np.array(sorted(bx * 1_000_003 + by for bx, by in blocks), dtype=np.int64)
    return np.isin(key, wanted)


def deck_edge_mask(model: RoadModel, xyz: np.ndarray) -> np.ndarray:
    """True for returns on a deck footprint but outside the carried road's pavement (walkways, curbs, parapets).

    ``roader.validate.surface_accuracy`` assigns such returns to the road below
    the deck, seven meters down; they are not travel surface of either road.
    """
    from roader.conflate.polyline import Polyline
    from roader.validate.accuracy import _pw

    out = np.zeros(len(xyz), dtype=bool)
    for st in model.structures:
        road = model.road(st.carried_road)
        line = Polyline.from_alignment(road.alignment)
        s, u, ok = line.project(xyz[:, :2], max_dist=40.0)
        s = np.nan_to_num(s)
        xs = road.cross_section
        on_deck = ok & (s >= st.s_start) & (s <= st.s_end)
        on_deck &= (u <= _pw(xs.width_left, s) + st.deck.overhang_m + 0.5) & (u >= -_pw(xs.width_right, s) - st.deck.overhang_m - 0.5)
        inside = (u <= _pw(xs.width_left, s)) & (u >= -_pw(xs.width_right, s))
        idx = np.flatnonzero(on_deck & ~inside)
        deck_z = np.array([road.profile.z(float(v)) for v in s[idx]])
        # Only returns at deck level; the lower road seen past the deck edge stays.
        out[idx[np.abs(xyz[idx, 2] - deck_z) < DECK_LEVEL_M]] = True
    return out


def measure_edges(model: RoadModel, points: PointSet, holdout: np.ndarray | None = None) -> list[dict]:
    """AC-A3 reference: pavement edges measured across each model road.

    With ``holdout``, edges are measured on all points but kept only where the
    edge lies in a held-out block (D-018), so every kept edge comes from returns
    the fit never saw while the cross-section around it is complete.
    """
    from roader.conflate.polyline import Polyline
    from roader.validate.accuracy import BLOCK_M

    held = _held_blocks(points, holdout)

    lines = [Polyline.from_alignment(r.alignment) for r in model.roads]
    ev = build_evidence(points, [ln.xy for ln in lines])
    out = []
    for road, line in zip(model.roads, lines, strict=True):
        tr = track_line(line, ev)
        lp = near_points(line, ev)
        if len(lp.s) == 0:
            continue
        prof = cross_profiles(lp, tr.z_at, line.length)
        valid = prof.valid & (prof.n > 0)
        for side, values in (("left", prof.left), ("right", prof.right)):
            for s, u in zip(prof.s[valid], values[valid], strict=True):
                if any(iv.kind == "occluded" and iv.s0 <= s <= iv.s1 for iv in road.support):
                    continue
                x, y = line.offset_point(float(s), float(u))[0]
                if held and (int(np.floor(x / BLOCK_M)), int(np.floor(y / BLOCK_M))) not in held:
                    continue
                out.append({"x": float(x), "y": float(y), "side": side, "road_id": road.id})
    return out
