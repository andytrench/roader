"""Milestone 2 structures: twin decks, a ramp diverging on a deck, observed piers, class 17 ends.

FR-CR-3 (decks shared or joined), FR-ST-3 (observed piers first), FR-VC-4 / D-031 (deck ends),
AC-H1 and AC-H3 on the rendered result.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from roader.conflate.structures import StructureExtent, absorb_diverging, find_extents
from roader.model.provenance import Source
from roader.solids import generate
from roader.structures.infer import observed_piers
from roader.structures.shared import attach_shared, twin_joints
from roader.validate.gates import check_solids, check_voids
from tests.pipeline.twin_scene import CENTRE_GAP, HALF, OVERHANG, RAMP_HALF, twin_model


def _plan(mesh) -> shapely.Geometry:
    tris = [shapely.Polygon(mesh.vertices[f][:, :2]) for f in mesh.faces]
    return shapely.union_all([t for t in tris if t.area > 1e-9])


def _deck(meshes, st):
    return next(m for m in meshes if m.id == f"{st.id}:deck")


# ---------------------------------------------------------------- twin decks


@pytest.fixture(scope="module")
def twins():
    model = twin_model()
    roads = {r.id: r for r in model.roads}
    flags = twin_joints(model.structures, roads, model.crossings)
    return model, flags, generate(model)


def test_fr_cr_3_twin_decks_meet_at_a_joint_on_the_facing_sides(twins):
    model, flags, _meshes = twins
    gap = CENTRE_GAP - 2.0 * HALF
    for st in model.structures:
        assert st.deck.overhang("left") <= 0.5 * gap + 1e-6  # each twin faces the other on its left
        assert st.deck.overhang("right") == pytest.approx(OVERHANG)
    assert sorted(f.element_id for f in flags if f.code == "FR-CR-3:twin_joint") == sorted(st.id for st in model.structures)


def test_fr_cr_3_no_rule_parapet_on_a_twin_joint(twins):
    model, _flags, _meshes = twins
    for st in model.structures:
        road = model.road(st.carried_road)
        sides = sorted(b.side for b in road.cross_section.barriers if b.kind == "parapet" and b.s1 > st.s_start and b.s0 < st.s_end)
        assert sides == ["right"]


def test_fr_cr_3_twin_decks_are_disjoint_in_plan(twins):
    model, _flags, meshes = twins
    a, b = (_plan(_deck(meshes, st)) for st in model.structures)
    assert a.intersection(b).area < 1e-6


def test_ac_h3_rays_under_a_twin_deck_reach_its_own_soffit(twins):
    model, _flags, meshes = twins
    result = check_voids(model, meshes)
    assert result.passed, [v.message for v in result.violations]


def test_ac_h1_twin_decks_valid(twins):
    _model, _flags, meshes = twins
    result = check_solids([m for m in meshes if m.closed])
    native = [v for v in result.violations if v.req != "val3dity"]
    assert not native, [v.message for v in native][:5]


def test_fr_cr_3_decks_far_apart_are_not_twins():
    model = twin_model()
    for st in model.structures:
        st.deck.overhang_m = 0.3  # 0.6 m of overhang in a 1 m gap and no parapets: the decks never meet
        model.road(st.carried_road).cross_section.barriers = []
    assert twin_joints(model.structures, {r.id: r for r in model.roads}, model.crossings) == []
    assert all(st.deck.overhang_left_m is None for st in model.structures)


# ---------------------------------------------------------------- a ramp diverging on a deck


@pytest.fixture(scope="module")
def diverge():
    model = twin_model(ramp=True)
    roads = {r.id: r for r in model.roads}
    west = model.structures[0]
    ramp = model.roads[-1]
    extents = [
        StructureExtent(west.carried_road, west.s_start, west.s_end, "class17", None, (west.s_start, west.s_end)),
        StructureExtent(ramp.id, 0.0, 17.0, "class17_only", None, (0.0, 17.0)),
    ]
    kept, shares, flags = absorb_diverging(extents, model.links, roads)
    barriers = {r.id: list(r.cross_section.barriers) for r in model.roads}
    flags += attach_shared(model.structures, shares, roads, barriers, model.crossings)
    for r in model.roads:
        r.cross_section.barriers = barriers[r.id]
    return model, kept, shares, flags


def test_fr_cr_3_ramp_diverging_on_a_deck_has_no_structure_of_its_own(diverge):
    model, kept, shares, _flags = diverge
    ramp = model.roads[-1]
    assert [e.road_id for e in kept] == [model.structures[0].carried_road]
    assert len(shares) == 1 and shares[0].road_id == ramp.id


def test_fr_cr_3_shared_span_runs_to_where_the_ramp_clears_the_deck_end(diverge):
    model, _kept, _shares, flags = diverge
    west, ramp = model.structures[0], model.roads[-1]
    (span,) = west.shared
    assert span.road_id == ramp.id and span.s_start == 0.0
    upper = model.road(west.carried_road)
    t = math.tan(west.supports[-1].skew_rad)
    for u in (RAMP_HALF, -RAMP_HALF):
        x, y = ramp.alignment.offset_point(span.s_end, u)
        s, v = upper.alignment.project(x, y)
        assert s - t * v >= west.s_end - 1e-3  # past the skewed deck end
    reach = RAMP_HALF + 0.45  # parapet outer face
    q = [upper.alignment.project(*ramp.alignment.offset_point(span.s_end - 1.0, u)) for u in (reach, -reach)]
    assert min(s - t * v for s, v in q) < west.s_end  # and not further than needed
    assert any(f.code == "FR-CR-3:shares_deck" and f.element_id == west.id for f in flags)
    assert all(b.s0 >= span.s_end - 1e-9 for b in ramp.cross_section.barriers)


def test_fr_cr_3_ramp_renders_nothing_on_the_shared_deck(diverge):
    model, _kept, _shares, _flags = diverge
    west, ramp = model.structures[0], model.roads[-1]
    meshes = generate(model)
    deck = _plan(_deck(meshes, west)).buffer(-0.01)
    mine = [m for m in meshes if m.element_id == ramp.id or m.element_id in {b.id for b in ramp.cross_section.barriers}]
    assert mine and not [m for m in mine if m.part in ("deck", "abutment")]
    for m in mine:
        assert _plan(m).intersection(deck).area < 1e-6, m.id
    result = check_solids([m for m in meshes if m.element_id in {west.id, *(sp.id for sp in west.supports)}])
    assert not [v for v in result.violations if v.req != "val3dity"]


def test_fr_cr_3_ramp_leaving_its_own_deck_beyond_the_parent_is_kept():
    model = twin_model(ramp=True)
    roads = {r.id: r for r in model.roads}
    west = model.structures[0]
    ramp = model.roads[-1]
    extents = [
        StructureExtent(west.carried_road, west.s_start, west.s_end, "class17", None, None),
        StructureExtent(ramp.id, 0.0, 60.0, "class17_only", None, None),  # its own deck runs on 45 m past the parent's end
    ]
    kept, shares, flags = absorb_diverging(extents, model.links, roads)
    assert len(kept) == 2 and not shares
    assert any(f.code == "FR-CR-3:diverges_on_deck" for f in flags)


# ---------------------------------------------------------------- observed piers (FR-ST-3 first priority)


def _pier_scene(t: float, rng) -> SimpleNamespace:
    """Returns below a deck at stations 100..180: a median column at 140 seen on the deck edge,
    trucks on the lower lanes, a shrub at the ground, a stray return pair, and the deck top itself."""
    s, u, drop, on_lower = [], [], [], []

    def add(ss, uu, dd, lower=False):
        s.extend(ss); u.extend(uu); drop.extend(dd); on_lower.extend([lower] * len(ss))

    k = 7
    uu = rng.uniform(-8.6, -8.2, k)
    add(140.0 + t * uu + rng.normal(0, 0.15, k), uu, rng.uniform(1.8, 4.4, k))  # column
    uu = rng.uniform(-8, 8, 60)
    add(120.0 + t * uu + rng.uniform(0, 4, 60), uu, rng.uniform(3.5, 4.5, 60), lower=True)  # truck
    uu = rng.uniform(-8.6, -8.0, 8)
    add(160.0 + t * uu + rng.normal(0, 0.3, 8), uu, rng.uniform(6.3, 6.7, 8))  # shrub 0.4 m tall
    add([150.0, 150.2], [-8.4, -8.3], [2.5, 2.6])  # two stray returns
    return SimpleNamespace(s=np.array(s), u=np.array(u), drop=np.array(drop), on_lower=np.array(on_lower), ground_drop=7.0)


@pytest.mark.parametrize("skew", [0.0, 0.5])
def test_fr_st_3_observed_pier_from_returns_under_the_deck(skew):
    t = math.tan(skew)
    sc = _pier_scene(t, np.random.default_rng(3))
    piers = observed_piers(sc.s, sc.u, sc.drop, sc.on_lower, sc.ground_drop, (100.0, 180.0), t)
    assert [round(p.station) for p in piers] == [140]
    assert piers[0].n_returns == 7 and piers[0].top_drop_m < 2.0


def test_fr_st_3_no_pier_from_returns_near_the_abutments():
    sc = _pier_scene(0.0, np.random.default_rng(3))
    assert observed_piers(sc.s, sc.u, sc.drop, sc.on_lower, sc.ground_drop, (137.0, 200.0), 0.0) == []


# ---------------------------------------------------------------- D-031: class 17 ends win


def _deck_road(length: float = 300.0):
    from roader.model.geometry import Profile, VerticalElement
    from roader.model.provenance import Provenance
    from tests.pipeline.twin_scene import _road

    flat = Profile([VerticalElement(0.0, length, 110.0, 0.0, 0.0, Provenance(Source.MEASURED, 1.0))])
    return _road("deck", 0.0, 0.0, 0.0, length, flat, 6.0, "mainline", "motorway")


def _deck_fit(road, deck: tuple[float, float], left_drop: tuple[float, float], right_drop: tuple[float, float] | None):
    """Surface returns along a flat road; class 17 on it over ``deck``; ground 6 m down beside each
    edge over the drop runs, flush elsewhere; no returns beside the right edge when it is None."""
    rng = np.random.default_rng(5)
    s = np.arange(0.0, road.length, 0.25)
    pts_s, pts_u, pts_z, cls = [], [], [], []
    for u in np.arange(-6.0, 6.0, 0.5):
        pts_s.append(s); pts_u.append(np.full(s.size, u)); pts_z.append(np.full(s.size, 110.0) + rng.normal(0, 0.01, s.size))
        cls.append(np.where((s >= deck[0]) & (s <= deck[1]), 17, 2))
    for sign, run in ((1.0, left_drop), (-1.0, right_drop)):
        if run is None:
            continue
        for d in np.arange(1.6, 4.0, 0.4):
            z = np.where((s >= run[0]) & (s <= run[1]), 104.0, 110.0)
            pts_s.append(s); pts_u.append(np.full(s.size, sign * (6.0 + d))); pts_z.append(z); cls.append(np.full(s.size, 2))
    s, u, z, c = (np.concatenate(v) for v in (pts_s, pts_u, pts_z, cls))
    fit = SimpleNamespace(road=road, points=SimpleNamespace(s=s, u=u, z=z, index=np.arange(s.size)))
    return fit, SimpleNamespace(classification=c.astype(np.uint8))


def test_d031_class17_ends_win_over_a_sheet_break_more_than_5_m_off():
    road = _deck_road()
    fit, ev = _deck_fit(road, (100.0, 180.0), (92.0, 188.0), (92.0, 188.0))  # wingwalls: 8 m long each end
    (e,) = find_extents(fit, ev)
    assert e.method == "class17"
    assert e.s_start == pytest.approx(100.0, abs=0.5) and e.s_end == pytest.approx(180.0, abs=0.5)
    assert e.geom[0] == pytest.approx(92.0, abs=1.5)


def test_d031_class17_ends_win_when_the_side_drop_is_one_sided():
    road = _deck_road()
    fit, ev = _deck_fit(road, (100.0, 180.0), (108.0, 172.0), None)
    (e,) = find_extents(fit, ev)
    assert e.method == "class17" and e.one_sided
    assert e.s_start == pytest.approx(100.0, abs=0.5) and e.s_end == pytest.approx(180.0, abs=0.5)


# ---------------------------------------------------------------- one parapet per deck side


def test_fr_st_5_partly_observed_parapet_is_extended_not_doubled():
    import dataclasses

    from roader.rules import RuleBook
    from roader.structures.infer import parapet_fill

    model = twin_model()
    st = model.structures[0]
    road = model.road(st.carried_road)
    seen = dataclasses.replace(road.cross_section.barriers[0], id="br_seen", s0=st.s_start + 20.0, s1=st.s_start + 30.0, offset_m=HALF + 0.1, height_m=0.8, prov=dataclasses.replace(road.cross_section.barriers[0].prov, source=Source.MEASURED))
    existing = [seen]
    filled = parapet_fill(road, st, existing, RuleBook.load())
    left = [b for b in existing + filled if b.side == seen.side]
    assert len(left) == 1
    (b,) = left
    assert (b.s0, b.s1) == (st.s_start, st.s_end) and b.offset_m == pytest.approx(HALF + 0.1) and b.height_m == 0.8
    assert any(f.startswith("FR-ST-5:parapet_extended_from:br_seen") for f in b.prov.flags)


def test_fr_st_3_layout_gap_is_between_merged_clear_zones():
    """A ramp beside a mainline under the deck (Washington Street): its clear zone lies inside the
    mainline's, so the only gap is past the mainline's zone, not past the ramp's (FR-ST-3 layout rule)."""
    from roader.structures.infer import _merged

    assert _merged([(589.5, 611.1), (591.9, 602.5), (612.6, 634.5)]) == [(589.5, 611.1), (612.6, 634.5)]


# ---------------------------------------------------------------- FR-SG-2 on skewed decks


def test_fr_sg_2_skewed_deck_with_a_superelevation_transition_stays_within_chord_tolerance():
    """On a skewed deck a station line runs along s as it crosses the road, so a cross-slope
    rate bends it: z'' = 2 tan(skew) dslope/ds. Its chord across the deck must stay within 1 cm,
    or rays stopping 1 cm under the soffit hit the deck (AC-H3, D-019)."""
    from roader.model.geometry import Knot, Piecewise
    from roader.model.provenance import Provenance

    model = twin_model()
    model.structures = model.structures[:1]
    st = model.structures[0]
    road = model.road(st.carried_road)
    model.crossings = [c for c in model.crossings if c.upper_road == road.id]
    p = Provenance(Source.MEASURED, 1.0)
    mid = 0.5 * (st.s_start + st.s_end)
    road.cross_section.slope_left = Piecewise([Knot(mid - 12.0, 0.08, p), Knot(mid + 12.0, -0.02, p)])
    meshes = generate(model)
    deck = _deck(meshes, st)
    worst = 0.0
    for f in deck.faces:
        tri = deck.vertices[f]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if n[2] > -1e-9:
            continue  # soffit faces only
        for w in ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)):
            q = np.dot(w, tri)
            s, u = road.alignment.project(float(q[0]), float(q[1]))
            worst = max(worst, road.surface_z(s, u) - st.deck.depth_m - float(q[2]))
    assert worst <= 0.01
    assert check_voids(model, meshes).passed


def test_fr_st_3_observed_pier_moves_into_the_median_not_onto_a_lane():
    from roader.structures.infer import clear_of_lanes

    lanes = [(508.0, 530.5), (532.6, 554.0)]  # a 2.1 m median between two lower carriageways
    assert clear_of_lanes(531.98, lanes, 1.5) == pytest.approx(531.55)  # median centre
    assert clear_of_lanes(506.0, lanes, 1.5) == 506.0  # beyond the shoulder already
    assert clear_of_lanes(507.6, lanes, 1.5) == pytest.approx(507.25)  # just off the outer edge
    assert clear_of_lanes(520.0, lanes, 1.5) is None  # mid-lane: not a pier
    assert clear_of_lanes(531.5, [(508.0, 531.0), (532.0, 554.0)], 1.5) is None  # median narrower than the column


def test_fr_cr_3_ramp_without_a_run_of_its_own_still_shares_the_deck():
    """A loop leaving 9 m before the deck end has too short a deck-class run for a structure;
    its first metres still ride the parent's deck (Woburn NE loop off I-93 northbound)."""
    model = twin_model(ramp=True)
    roads = {r.id: r for r in model.roads}
    west = model.structures[0]
    extents = [StructureExtent(west.carried_road, west.s_start, west.s_end, "class17", None, None)]
    kept, shares, _flags = absorb_diverging(extents, model.links, roads)
    assert len(kept) == 1 and [s.road_id for s in shares] == [model.roads[-1].id]
