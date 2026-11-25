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
import math
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
from roader.conflate.register import (
    Neighbour,
    corridor_limits,
    cross_profiles,
    near_points,
)
from roader.conflate.structures import (
    StructureExtent,
    absorb_diverging,
    chain_extents,
    cover_crossed_roads,
    find_extents,
    side_drop,
)
from roader.conflate.track import track_line
from roader.evidence import Evidence, build_evidence
from roader.evidence.strips import STRIP_TOL_M
from roader.fit.joint import joint_fit_roads
from roader.fit.links import (
    bound_tapers,
    child_anchors,
    drop_junction_crossings,
    find_links,
    fit_network,
    least_half_width,
)
from roader.fit.median import cap_at, classify_medians, reference_caps, split_divided
from roader.fit.road import RoadFit, fit_road
from roader.fit.section import barriers_from_evidence
from roader.io.points import PointSet
from roader.io.vectors import VectorEdge
from roader.model.provenance import Provenance, Source
from roader.model.schema import ELEMENT_FAILED, Flag, Frame, RoadModel, UnitRecord
from roader.model.serialize import dumps, loads
from roader.rules import RuleBook, era_of
from roader.structures import attach_shared, infer_structure, parapet_fill, twin_joints
from roader.structures.held_skew import CAP_MIN, capped, held_skew
from roader.structures.infer import _footprint_on, abutment_thickness

RULE_EXTENT_PAD_M = 1.0
MAX_RELAYOUTS = 3  # D-171: layouts read for the skew the sweep holds
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
    fits, oflags = _streets_off_decks(fits, junctions, ev, rules, strip_factor)
    flags.extend(oflags)
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
    extents, xflags = cover_crossed_roads(extents, crossings, lambda c, t: _crossed_footprint(by_id, c, t, rules))
    flags.extend(xflags)
    extents = chain_extents(extents, drops, {f.road.id: f.road.length for f in fits})
    extents, xflags = _over_something(extents, crossings)
    flags.extend(xflags)
    roads = {f.road.id: f.road for f in fits}
    extents, shares, xflags = absorb_diverging(extents, links, roads)
    flags.extend(xflags)

    barriers = {}
    for f in fits:
        ranges = [(e.s_start, e.s_end) for e in extents if e.road_id == f.road.id]
        xs = f.road.cross_section
        try:
            barriers[f.road.id] = barriers_from_evidence(f.road.id, f.line, ev, xs.width_left, xs.width_right, structure_ranges=ranges, rules=rules)
        except Exception as exc:  # noqa: BLE001 - one road's barriers never abort the unit (FR-EX-1, D-119)
            barriers[f.road.id] = []
            flags.append(element_failed(f.road.id, "FR-RS-4", "no measured barriers on the road", exc))

    unit = copy.deepcopy(unit)
    for rec in unit.lidar:
        rec.density_pavement_pts_m2 = ev.density.median_pts_m2
        rec.strip_dz_p95_m = ev.strips.p95_m

    def layout(fits, barriers, extents, hold) -> tuple[RoadModel, list[Flag], list]:
        """Structures, earthwork and terrain for one layout; ``hold`` caps support skew (D-171). Mutates ``fits``."""
        out: list[Flag] = []
        by_id = {f.road.id: f for f in fits}
        roads = {f.road.id: f.road for f in fits}
        structures = []
        for e in extents:
            try:
                res = infer_structure(e, by_id[e.road_id], crossings, by_id, ev, rules, drops[e.road_id], barriers[e.road_id], hold=hold.get(e.road_id))
            except Exception as exc:  # noqa: BLE001 - one structure never aborts the unit (FR-EX-1, D-119)
                out.append(element_failed(e.road_id, "FR-ST-2", f"no structure for the {e.method} span s={e.s_start:.1f}-{e.s_end:.1f}", exc))
                continue
            if e.method == "rule":
                res.structure.prov.source = Source.RULE
                res.structure.prov.rule_id = "clearance.lateral.support"
                res.structure.prov.confidence = 0.3
            structures.append(res.structure)
            out.extend(res.flags)
            filled = parapet_fill(by_id[e.road_id].road, res.structure, barriers[e.road_id], rules)
            barriers[e.road_id].extend(filled)
            _fit_overhang(by_id[e.road_id].road, res.structure, barriers[e.road_id])
        out.extend(attach_shared(structures, shares, roads, barriers, crossings))

        for f in fits:
            _clip_parapets(barriers[f.road.id], [st for st in structures if st.carried_road == f.road.id])
            _parapets_outside_pavement(f.road, barriers[f.road.id])
            for st in structures:
                if st.carried_road == f.road.id:
                    _fit_overhang(f.road, st, barriers[f.road.id])
            f.road.cross_section.barriers = sorted(barriers[f.road.id], key=lambda b: (b.side, b.s0, b.id))
        # A parent and its linked child share their taper boundary, off their decks (FR-XS-1, D-104).
        out.extend(bound_tapers(fits, junctions, _decks(structures)))
        out.extend(twin_joints(structures, roads, crossings))
        out.extend(classify_medians(fits, ev))

        slopes = []
        for f in fits:
            try:
                slopes.extend(side_slopes(f, structures, rules, strip_factor))
            except Exception as exc:  # noqa: BLE001 - one road's earthwork never aborts the unit (FR-EX-1, D-119)
                out.append(element_failed(f.road.id, "FR-OC-3", "no side slopes on the road", exc))

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
        out.extend(check_clearances(model, rules))
        completions = complete_occluded(model, {f.road.id: (f.track.s, f.track.z) for f in fits})
        out.extend(completion_flags(completions))
        # Stage 8 terrain after the clearance refit, so caps and soffits are final (FR-OC-3).
        model.terrain = complete_terrain(fit_points, model, rules)
        seat_supports(model, rules)
        out.extend(_buildable_fills(model))
        return model, out, completions

    start_roads = copy.deepcopy([f.road for f in fits])
    start_barriers = copy.deepcopy(barriers)
    model, layout_flags, completions = layout(fits, barriers, extents, {})
    # D-171: supports carry the skew the sweep can hold, read from the solids on each layout
    # until the layout holds it (a capped skew field can itself fail and be squared).
    first, hold = model.structures, {}
    for _ in range(MAX_RELAYOUTS):
        found = held_skew(model)
        if not _skew_caps(model.structures, found):
            break
        hold = _tighter(hold, found)
        fits = [dataclasses.replace(f, road=r) for f, r in zip(fits, copy.deepcopy(start_roads), strict=True)]
        by_id = {f.road.id: f for f in fits}
        overhang = {st.carried_road: max(st.deck.overhang("left"), st.deck.overhang("right"), 0.0) for st in model.structures}
        extents, xflags = cover_crossed_roads(
            extents, crossings, lambda c, t, by_id=by_id, ov=overhang: _crossed_footprint(by_id, c, t, rules, ov.get(c.upper_road, 0.0)), hold
        )
        flags.extend(xflags)
        model, layout_flags, completions = layout(fits, copy.deepcopy(start_barriers), extents, hold)
    if hold:
        layout_flags.extend(_capped_flags(first, model.structures))
    flags.extend(layout_flags)
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


DECK_SHARE_DZ_M = 2.0  # a street crossing a limited-access road closer than this in height shares its surface
DECK_CLASS = 17  # ASPRS bridge deck
DECK_SHARE_FRACTION = 0.5  # share of deck-class returns that makes the shared surface a deck
DECK_NEAR_M = 15.0
DECK_LEVEL_M = 1.5  # cells this close to the deck road's surface are its surface


def _streets_off_decks(fits: list[RoadFit], js, ev: Evidence, rules: RuleBook, strip_factor: float) -> tuple[list[RoadFit], list[Flag]]:
    """A street tracked onto the deck of a limited-access road it crosses is refit below it (FR-VC-2, FR-CR-1, D-144).

    Limited-access roads have no at-grade crossings, so a street meeting one in plan at
    almost its height, without a junction between them, has taken that road's deck as
    its own surface (a street under a viaduct whose only long visible surface is the
    deck). When the shared surface is a deck (mostly class 17), the street is refit with
    that deck's sheets withheld inside the deck road's footprint; it is then completed
    under the deck from what it does see (FR-OC-1). The deck road keeps its fit.
    """
    crossings, _measures, _flags = find_crossings(fits)
    by_id = {f.road.id: f for f in fits}
    linked = {frozenset((j.child, j.parent)) for j in js}
    hits: dict[str, list[tuple[str, float]]] = {}
    for c in crossings:
        if abs(c.dz_m) >= DECK_SHARE_DZ_M or frozenset((c.upper_road, c.lower_road)) in linked:
            continue
        pair = [(by_id[c.upper_road], c.upper_s), (by_id[c.lower_road], c.lower_s)]
        core = [(f, s) for f, s in pair if f.road.kind != "crossing_street"]
        street = [f for f, _s in pair if f.road.kind == "crossing_street"]
        if len(core) == 1 and len(street) == 1:
            hits.setdefault(street[0].road.id, []).append((core[0][0].road.id, core[0][1]))
    if not hits:
        return fits, []
    lay = ev.layers
    centers = lay.center_xy()
    point_cells = lay.point_layer
    flags: list[Flag] = []
    fitted = dict(by_id)
    for sid in sorted(hits):
        mask = np.zeros(len(lay.z), dtype=bool)
        decks = []
        for core_id, s_c in sorted(hits[sid]):
            cf = by_id[core_id]
            xs = cf.road.cross_section
            half = max(max(k.value for k in xs.width_left.knots), max(k.value for k in xs.width_right.knots)) + 2.0
            cs, cu, ok = cf.line.project(centers, max_dist=half)
            # The deck road's own surface: cells inside its footprint at its height (any sheet).
            ok &= np.isfinite(cs)
            zc = np.full(len(lay.z), np.inf)
            zc[ok] = np.array([cf.road.surface_z(float(a), float(b)) for a, b in zip(cs[ok], cu[ok], strict=True)])
            cells = ok & (np.abs(lay.z - zc) <= DECK_LEVEL_M)
            x, y = cf.road.alignment.offset_point(s_c, 0.0)
            pts = cells[point_cells] & (np.hypot(ev.xyz[:, 0] - x, ev.xyz[:, 1] - y) <= DECK_NEAR_M)
            if pts.sum() < 10 or float(np.mean(ev.classification[pts] == DECK_CLASS)) < DECK_SHARE_FRACTION:
                continue
            mask |= cells
            decks.append(cf.road.name)
        if not mask.any():
            continue
        old = fitted[sid]
        masked = dataclasses.replace(ev, layer_label=np.where(mask, 3, ev.layer_label), layer_prob=np.where(mask, 0.0, ev.layer_prob))
        neighbours = [Neighbour(o.seed.line, least_half_width(o.seed, rules)) for o in fits if o.road.id != sid]
        try:
            new = fit_road(old.seed, masked, rules, strip_factor=strip_factor, anchors=child_anchors(old.seed, js, fitted), neighbours=neighbours)
        except Exception as exc:  # noqa: BLE001 - keep the first fit, flagged (FR-EX-1, D-119)
            flags.append(element_failed(sid, "FR-VC-2", f"refit below the deck of {', '.join(decks)} failed; first fit kept", exc))
            continue
        fitted[sid] = new
        flags.append(Flag("FR-VC-2:off_deck", sid, "FR-VC-2", f"{old.road.name} met {', '.join(decks)} at deck level with no junction; refit with that deck's surface withheld"))
    return [fitted[f.road.id] for f in fits], flags


def element_failed(element_id: str, req: str, what: str, exc: BaseException) -> Flag:
    """FR-EX-1 flag for an element left out because building it raised; holds the unit (D-119)."""
    return Flag(ELEMENT_FAILED, element_id, req, f"{what}; {type(exc).__name__}: {exc}")


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


def _skew_caps(structures, hold) -> bool:
    """True when some support is steeper than its road's sweep holds at its station (FR-ST-4, D-171)."""
    return any(
        abs(math.tan(sp.skew_rad)) - abs(capped(math.tan(sp.skew_rad), hold[st.carried_road], sp.station)) > CAP_MIN
        for st in structures
        if st.carried_road in hold
        for sp in st.supports
    )


def _tighter(a: dict, b: dict) -> dict:
    """Per road, the smaller of two held |tan skew| fields."""
    out = dict(a)
    for road_id, f in b.items():
        g = out.get(road_id)
        out[road_id] = f if g is None else (lambda s, f=f, g=g: min(f(s), g(s)))
    return out


def _capped_flags(first, final) -> list[Flag]:
    """``FR-ST-4:skew_capped`` on each final structure whose supports lost skew against the first layout's (D-171)."""
    out = []
    for st in final:
        before = [x for x in first if x.carried_road == st.carried_road and x.s_start < st.s_end and x.s_end > st.s_start]
        full = max((abs(math.tan(sp.skew_rad)) for x in before for sp in x.supports), default=0.0)
        cut = [sp for sp in st.supports if full - abs(math.tan(sp.skew_rad)) > CAP_MIN]
        if cut:
            text = ", ".join(f"{sp.id} {full:.2f} -> {abs(math.tan(sp.skew_rad)):.2f}" for sp in cut)
            out.append(Flag("FR-ST-4:skew_capped", st.id, "FR-ST-4", f"|tan skew| capped to what the sweep holds near the road end or steep toes: {text}"))
    return out


def _crossed_footprint(by_id, c, t: float, rules, overhang: float = 0.0) -> tuple[float, float]:
    """Deck-end stations whose abutment block clears the crossed pavement by the least lateral clearance (D-166, D-170).

    The block reaches ``abutment_thickness`` from the deck end into the span, so the
    support-line footprint of the pavement is widened by that much at each end. The block
    spans the deck, ``overhang`` beyond each pavement edge (known once laid out, D-171).
    """
    upper, lower = by_id[c.upper_road], by_id[c.lower_road]
    least = rules.get("clearance.lateral.support", road_class=lower.road.road_class).low
    xs = upper.road.cross_section
    across = (-xs.width_right(c.upper_s) - overhang, xs.width_left(c.upper_s) + overhang)
    a, b = _footprint_on(upper, lower, c, least, t=t, across=across)
    thick = abutment_thickness(rules, era_of(None))
    return a - thick, b + thick


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


def _least_half_width(road, rules) -> float:
    """Half the narrowest pavement a model road can have (as ``fit.links.least_half_width``)."""
    counts = []
    for ref in road.vector_refs:
        try:
            counts.append(int(ref.tags.get("lanes", "")))
        except (ValueError, AttributeError):
            continue
    lanes = min(counts) if counts else (1 if road.oneway else 2)
    return 0.5 * max(lanes, 1) * rules.get("lane.width", road_class=road.road_class).low


def _decks(structures) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for st in structures:
        out.setdefault(st.carried_road, []).append((float(st.s_start), float(st.s_end)))
    return out


def _clamp_edges(st, left, right, caps: dict, limits) -> tuple[np.ndarray, np.ndarray]:
    """Edges found past a cap end at it (D-103, D-104); then those past a neighbour limit are dropped (D-068)."""
    cap_l, cap_r = cap_at(caps.get(1, []), st), cap_at(caps.get(-1, []), st)
    left = np.where(left > cap_l, cap_l, left)
    right = np.where(-right > cap_r, -cap_r, right)
    if limits is not None:
        left_max, right_max = limits(st)
        left = np.where(left > left_max, np.nan, left)
        right = np.where(-right > right_max, np.nan, right)
    return left, right


def measure_edges(model: RoadModel, points: PointSet, holdout: np.ndarray | None = None) -> list[dict]:
    """AC-A3 reference: pavement edges measured across each model road.

    With ``holdout``, edges are measured on all points but kept only where the
    edge lies in a held-out block (D-018), so every kept edge comes from returns
    the fit never saw while the cross-section around it is complete.
    """
    from roader.conflate.polyline import Polyline
    from roader.validate.accuracy import BLOCK_M

    held = _held_blocks(points, holdout)
    rules = RuleBook.load()

    lines = [Polyline.from_alignment(r.alignment) for r in model.roads]
    halves = [_least_half_width(r, rules) for r in model.roads]
    ev = build_evidence(points, [ln.xy for ln in lines])
    # D-103, D-104: the search stops where the fit's edge does, at a twin's median cap or a taper boundary.
    fit_pts = ~np.asarray(ev.transient, dtype=bool)
    if holdout is not None and len(holdout) == len(fit_pts):
        fit_pts &= ~np.asarray(holdout, dtype=bool)
    caps = reference_caps(model.roads, model.links, ev.xyz[fit_pts], _decks(model.structures))
    out = []
    for k, (road, line) in enumerate(zip(model.roads, lines, strict=True)):
        tr = track_line(line, ev)
        lp = near_points(line, ev)
        if len(lp.s) == 0:
            continue
        # D-068: the same neighbour limits as the fit (D-063), so a reference edge never lies on another road.
        neighbours = [Neighbour(lines[j], halves[j]) for j in range(len(lines)) if j != k]
        prof = cross_profiles(lp, tr.z_at, line.length)
        left, right = _clamp_edges(prof.s, prof.left, prof.right, caps.get(road.id, {}), corridor_limits(line, neighbours))
        prof = dataclasses.replace(prof, left=left, right=right)
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
