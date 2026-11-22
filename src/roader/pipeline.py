"""Stages 3 to 9: points and vectors in, the section 5 parametric model out.

``build_model`` is the single entry point. It is deterministic: the same inputs
produce a byte-identical ``roader.model.dumps`` (AC-H5, FR-PL-3). Points flagged
in ``holdout`` are removed before any stage sees them (D-018), so AC-A1 can be
measured on them afterwards.

Order: vector graph (FR-VC-1) -> evidence on the corridor (FR-PN-2/3, FR-RS-1..4)
-> divided highways drawn as one way split into twins (FR-XS-4) -> per road,
parents before the ramps hanging off them: sheet tracking, registration, L1,
L2, L3 (FR-VC-2/3, FR-AF-1..5, FR-XS-1..3) -> crossings (FR-CR-1/2) -> links
and the joint L2 solve per connected component (FR-AF-3) -> structure ends
(FR-VC-4, FR-CR-3) -> barriers and structures (FR-RS-4, FR-ST-2..6, FR-XS-5)
-> medians (FR-XS-4) -> completion and clearance (FR-OC-1..3) -> earthwork and
terrain (L4, L7) -> junction areas (L6, FR-SG-3) -> unmatched sheets (FR-VC-5).

Not yet: inventory matching (FR-ST-1), observed piers, multiple collections
(FR-IN-4).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from roader.complete import (
    Completion,
    check_clearances,
    complete_occluded,
    complete_terrain,
    completion_flags,
    label_under_deck,
    seat_supports,
    side_slopes,
    split_at_structures,
)
from roader.complete.junctions import build_junctions
from roader.conflate.candidates import SheetCandidate, candidate_flag, unmatched_sheets
from roader.conflate.crossings import CrossingMeasure, find_crossings
from roader.conflate.graph import data_extent, select_roads
from roader.conflate.register import cross_profiles, near_points
from roader.conflate.structures import (
    StructureExtent,
    chain_extents,
    find_extents,
    side_drop,
)
from roader.conflate.track import track_line
from roader.evidence import Evidence, build_evidence
from roader.evidence.strips import STRIP_TOL_M
from roader.fit.joint import joint_fit_roads
from roader.fit.links import drop_junction_crossings, find_links, fit_network
from roader.fit.median import classify_medians, split_divided
from roader.fit.road import RoadFit
from roader.fit.section import barriers_from_evidence
from roader.io.points import PointSet
from roader.io.vectors import VectorEdge
from roader.model.provenance import Provenance, Source
from roader.model.schema import Flag, Frame, RoadModel, UnitRecord
from roader.model.serialize import dumps, loads
from roader.rules import RuleBook
from roader.structures import infer_structure, parapet_fill
from roader.structures.infer import _footprint_on

RULE_EXTENT_PAD_M = 1.0
SKEW_REACH_M = 15.0
PARAPET_EDGE_GAP_M = 0.02  # keeps the parapet face and the pavement edge distinct sweep lines


@dataclass
class PipelineResult:
    model: RoadModel
    evidence: Evidence
    fits: list[RoadFit]
    crossing_measures: list[CrossingMeasure] = field(default_factory=list)
    extents: list[StructureExtent] = field(default_factory=list)
    candidates: list[SheetCandidate] = field(default_factory=list)  # FR-VC-5
    completions: list[Completion] = field(default_factory=list)  # FR-OC-1

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

    seeds, dflags = split_divided(seeds, ev)
    flags.extend(dflags)
    fits, junctions = fit_network(seeds, ev, rules, strip_factor=strip_factor)
    by_id = {f.road.id: f for f in fits}
    held_blocks = _held_blocks(points, holdout)
    for f in fits:
        withheld = _mark_withheld(f.road, held_blocks)
        flags.extend(fl for fl in f.flags if not (fl.code == "FR-VC-2:unsupported" and any(f"s={a:.1f}-{b:.1f}" in fl.message for a, b in withheld)))

    crossings, measures, cflags = find_crossings(fits)
    flags.extend(cflags)
    links, lflags = find_links(fits, junctions)
    flags.extend(lflags)
    crossings, measures, jflags = drop_junction_crossings(crossings, measures, links)
    flags.extend(jflags)
    flags.extend(joint_fit_roads(fits, links, crossings, rules))
    flags.extend(fl for f in fits for fl in f.profile_flags)

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
    extents = chain_extents(extents, drops, {f.road.id: f.road.length for f in fits})
    extents, xflags = _over_something(extents, crossings)
    flags.extend(xflags)

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
        _clip_parapets(barriers[f.road.id], [st for st in structures if st.carried_road == f.road.id])
        _parapets_outside_pavement(f.road, barriers[f.road.id])
        for st in structures:
            if st.carried_road == f.road.id:
                _fit_overhang(f.road, st, barriers[f.road.id])
        f.road.cross_section.barriers = sorted(barriers[f.road.id], key=lambda b: (b.side, b.s0, b.id))
    flags.extend(classify_medians(fits, ev))

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
        links=links,
        crossings=sorted(crossings, key=lambda c: c.id),
        structures=sorted(structures, key=lambda s: s.id),
        side_slopes=sorted(slopes, key=lambda s: s.id),
    )
    for road in model.roads:
        if any(st.carried_road == road.id for st in model.structures):
            split_at_structures(road, model.structures)
        label_under_deck(road, model)
        road.prov = road_summary(road)
    flags.extend(check_clearances(model, rules))
    completions = complete_occluded(model, {f.road.id: (f.track.s, f.track.z) for f in fits})
    flags.extend(completion_flags(completions))
    # Stage 8 terrain after the clearance refit, so caps and soffits are final (FR-OC-3).
    model.terrain = complete_terrain(fit_points, model, rules)
    seat_supports(model, rules)
    flags.extend(_buildable_fills(model))
    # L6 areas lifted from the canonical roads, so they match the sweeps of the saved model (FR-SG-3, D-043, D-046).
    model.junctions, junction_flags = build_junctions(loads(dumps(dataclasses.replace(model, terrain=None))))
    flags.extend(junction_flags)
    candidates = unmatched_sheets(ev, fits)
    flags.extend(candidate_flag(c) for c in candidates)
    model.flags = sorted(flags, key=lambda f: (f.req, f.code, f.element_id, f.message))
    # The canonical form (D-011): a model built here and the same model read back
    # from its native file render byte-identical exports (AC-H5).
    model = loads(dumps(model))
    return PipelineResult(model, ev, fits, measures, extents, candidates, completions)


def road_summary(road) -> Provenance:
    """D-045: whole-road source for the rendered body, by length over alignment and profile."""
    weight: dict[Source, float] = {}
    conf = 0.0
    total = 0.0
    for e in [*road.alignment.elements, *road.profile.elements]:
        weight[e.prov.source] = weight.get(e.prov.source, 0.0) + e.length
        conf += e.prov.confidence * e.length
        total += e.length
    source = max(weight, key=lambda s: (weight[s], s.value))
    share = weight[source] / total
    rule_id = None
    if source == Source.RULE:
        rule_id = max((e for e in [*road.alignment.elements, *road.profile.elements] if e.prov.source == Source.RULE), key=lambda e: e.length).prov.rule_id
    return Provenance(source, round(conf / total, 6), observed_fraction=round(weight.get(Source.MEASURED, 0.0) / total, 6), rule_id=rule_id, flags=[f"FR-RH-1:road_summary:{source.value}:{share:.2f}"])


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


FOLD_FRACTION = 0.7  # a fill toe on the inside of a curve stays within this share of the radius
FOLD_STEP_M = 5.0


def _buildable_fills(model: RoadModel) -> list[Flag]:
    """L4 fills whose toe cannot be built are written as kind ``none`` with the measured ratio kept (D-041).

    On the inside of a tight curve (a cloverleaf loop over its lower infield) a fill
    toe reaching toward the centre of curvature folds the embankment over itself;
    a pavement underside below the terrain has no fill at all. The toe is found as
    the solids find it, on the finished terrain (L7), every ``FOLD_STEP_M``.
    """
    if model.terrain is None:
        return []
    from roader.solids.bodies import _toe
    from roader.solids.terrain import TerrainSampler

    sampler = TerrainSampler(model.terrain)
    out = []
    for ss in model.side_slopes:
        if ss.kind != "fill":
            continue
        road = model.road(ss.road_id)
        sign = 1.0 if ss.side == "left" else -1.0
        toe = _toe(road, sign, ss.ratio, sampler, ss.id)
        reason = None
        for s in np.linspace(ss.s0, ss.s1, max(2, int(np.ceil((ss.s1 - ss.s0) / FOLD_STEP_M)) + 1)):
            k = road.alignment.evaluate(float(s))[3]
            try:
                u = abs(toe(float(s)))
            except ValueError as exc:
                reason = f"no toe at s={s:.1f}: {exc}"
                break
            if sign * k > 0.0 and u * abs(k) >= FOLD_FRACTION:
                reason = f"toe {u:.1f} m out on the inside of a {1.0 / abs(k):.0f} m radius at s={s:.1f}"
                break
        if reason is None:
            continue
        ss.prov.flags.append(f"L4:measured_fill_unbuildable:ratio={ss.ratio:.2f}")
        ss.kind, ss.ratio = "none", 0.0
        out.append(Flag("L4:fill_unbuildable", ss.id, "FR-OC-3", f"fill written as none: {reason}"))
    return out


def _over_something(extents: list[StructureExtent], crossings) -> tuple[list[StructureExtent], list[Flag]]:
    """A structure carries its road over something (FR-VC-4, section 1 scope).

    An extent with no crossing under it and no deck-class returns on it is a ground
    drop beside the road (a steep embankment, a wall, the deck of a road above), not
    a bridge: it is flagged and dropped. Deck-class returns keep an extent, since a
    ramp leaving on a deck rides that deck without crossing anything itself.
    """
    keep, flags = [], []
    for e in extents:
        over = any(c.upper_road == e.road_id and e.s_start - 1.0 <= c.upper_s <= e.s_end + 1.0 for c in crossings)
        if over or e.method in ("class17", "class17_only") or e.method == "rule":
            keep.append(e)
            continue
        flags.append(Flag("FR-VC-4:over_nothing", e.road_id, "FR-VC-4", f"{e.method} run s={e.s_start:.1f}-{e.s_end:.1f}{' (one side)' if e.one_sided else ''} crosses no road and has no deck-class returns; no structure"))
    return keep, flags


def _clip_parapets(barriers, structures) -> None:
    """Parapets stand on the deck: their station range is the structure's (FR-RD-4, FR-SG-1)."""
    for b in barriers:
        if b.kind != "parapet":
            continue
        for st in structures:
            if b.s1 > st.s_start and b.s0 < st.s_end:
                b.s0, b.s1 = max(b.s0, st.s_start), min(b.s1, st.s_end)
                if abs(b.s0 - st.s_start) < 2.0:
                    b.s0 = st.s_start
                if abs(b.s1 - st.s_end) < 2.0:
                    b.s1 = st.s_end
                break


def _parapets_outside_pavement(road, barriers) -> None:
    """A parapet's inner face never cuts into the pavement it bounds: offset >= widest pavement over its range."""
    for b in barriers:
        if b.kind != "parapet":
            continue
        piece = road.cross_section.width_left if b.side == "left" else road.cross_section.width_right
        # Skewed sweep stations sample the edge up to |tan(skew)| * width beyond the span.
        a, c = b.s0 - SKEW_REACH_M, b.s1 + SKEW_REACH_M
        stations = [a, c] + [k.s for k in piece.knots if a < k.s < c]
        widest = max(piece(s) for s in stations) + PARAPET_EDGE_GAP_M
        if b.offset_m < widest:
            b.offset_m = widest
            b.prov.flags.append("FR-SG-1:offset_at_pavement_edge")


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
