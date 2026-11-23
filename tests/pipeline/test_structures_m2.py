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
from tests.pipeline.twin_scene import (
    CENTRE_GAP,
    HALF,
    OVERHANG,
    RAMP_HALF,
    RAMP_TURN,
    RAMP_U0,
    twin_model,
)


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


@pytest.fixture
def overlapping_twins(monkeypatch):
    """Fitted pavements 2 m into each other across the median (Woburn I-93 after D-084)."""
    import tests.pipeline.twin_scene as scene

    monkeypatch.setattr(scene, "CENTRE_GAP", 2.0 * HALF - 2.0)
    model = scene.twin_model()
    roads = {r.id: r for r in model.roads}
    flags = twin_joints(model.structures, roads, model.crossings)
    return model, flags, generate(model)


def test_fr_cr_3_overlapping_twin_pavements_still_give_disjoint_decks(overlapping_twins):
    model, flags, meshes = overlapping_twins
    for st in model.structures:
        assert st.deck.overhang("left") == pytest.approx(-(2.0 + 0.02) / 2.0, abs=0.05)
    a, b = (_plan(_deck(meshes, st)) for st in model.structures)
    assert a.intersection(b).area < 1e-6
    assert sorted(f.element_id for f in flags if f.code == "FR-CR-3:twin_overlap") == sorted(st.id for st in model.structures)
    # The overlapping approach pavements themselves stay a fit problem; the structures are clean.
    parts = {st.id for st in model.structures} | {sp.id for st in model.structures for sp in st.supports}
    result = check_solids([m for m in meshes if m.element_id in parts])
    assert not result.violations, [v.message for v in result.violations][:3]
    result = check_voids(model, meshes)
    assert result.passed, [v.message for v in result.violations]


def test_fr_cr_3_roads_overlapping_by_more_than_a_fit_error_are_not_trimmed_as_twins(monkeypatch):
    """Dallas stack: viaducts converging at a merge overlap by up to 16 m; trimming would cut a
    deck past its own centerline. Beyond TWIN_MAX_OVERLAP_M they are converging roads, not twins."""
    import tests.pipeline.twin_scene as scene
    from roader.structures.shared import TWIN_MAX_OVERLAP_M

    monkeypatch.setattr(scene, "CENTRE_GAP", 2.0 * HALF - (TWIN_MAX_OVERLAP_M + 2.0))
    model = scene.twin_model()
    flags = twin_joints(model.structures, {r.id: r for r in model.roads}, model.crossings)
    for st in model.structures:
        assert st.deck.overhang("left") == pytest.approx(OVERHANG)
        assert any(b.side == "left" for b in model.road(st.carried_road).cross_section.barriers)
    assert sorted(f.element_id for f in flags if f.code == "FR-CR-3:converging_decks") == sorted(st.id for st in model.structures)
    assert not [f for f in flags if f.code == "FR-CR-3:twin_joint"]


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
    assert any(f.code == "FR-CR-3:shares_deck" and f.element_id == west.id for f in flags)
    assert all(b.s0 >= span.s_end - 1e-9 for b in ramp.cross_section.barriers)


def _inside_parent(upper, ramp, s: float) -> bool:
    for u in (RAMP_HALF, -RAMP_HALF):
        ps, pu = upper.alignment.project(*ramp.alignment.offset_point(s, u))
        if not -upper.cross_section.width_right(ps) - 1e-3 <= pu <= upper.cross_section.width_left(ps) + 1e-3:
            return False
    return True


def test_fr_cr_3_shared_span_runs_on_while_the_ramp_is_inside_the_parent_pavement(diverge):
    model, _kept, _shares, _flags = diverge
    west, ramp = model.structures[0], model.roads[-1]
    (span,) = west.shared
    upper = model.road(west.carried_road)
    # RAMP_U0 - RAMP_HALF reaches the parent's edge -HALF after (HALF + RAMP_U0 - RAMP_HALF) / |RAMP_TURN| m
    leaves = (HALF + RAMP_U0 - RAMP_HALF) / abs(RAMP_TURN)
    assert span.s_end == pytest.approx(leaves, abs=0.6)
    assert all(_inside_parent(upper, ramp, s) for s in np.arange(span.s_start, span.s_end - 0.6, 1.0))
    assert not _inside_parent(upper, ramp, span.s_end + 0.6)


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


def test_fr_st_3_returns_8_m_from_an_abutment_are_the_spill_slope_not_a_pier():
    sc = _pier_scene(0.0, np.random.default_rng(3))
    assert observed_piers(sc.s, sc.u, sc.drop, sc.on_lower, sc.ground_drop, (132.0, 200.0), 0.0) == []


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


def _fit_of(road):
    from roader.conflate.polyline import Polyline

    xy = np.array([road.alignment.offset_point(float(s), 0.0) for s in np.arange(0.0, road.length + 0.25, 0.5)])
    return SimpleNamespace(road=road, line=Polyline(xy))


def test_fr_st_3_lower_pavement_measured_along_the_whole_support_line():
    """A lower road almost parallel to the support line (Woburn I-95 under the skewed I-93 decks):
    a pier line just off its edge at the carried centerline runs onto it at the deck edges."""
    from roader.model.geometry import Profile, VerticalElement
    from roader.model.provenance import Provenance
    from roader.structures.infer import _footprint_on
    from tests.pipeline.twin_scene import _road

    flat = Profile([VerticalElement(0.0, 200.0, 110.0, 0.0, 0.0, Provenance(Source.MEASURED, 1.0))])
    upper = _road("deck", 0.0, 0.0, 0.0, 200.0, flat, 13.0, "mainline", "motorway")
    t = math.tan(0.55)
    heading = math.atan2(1.0, t) + 0.05  # 0.05 rad off the support line
    x0, y0 = 100.0 - 80.0 * math.cos(heading), -80.0 * math.sin(heading)
    lower = _road("under", x0, y0, heading, 160.0, flat, 5.0, "mainline", "motorway")
    c = SimpleNamespace(lower_s=80.0, upper_s=100.0, skew_rad=0.55)
    across = (-13.5, 13.5)
    a0, b0 = _footprint_on(_fit_of(upper), _fit_of(lower), c, 0.0)
    a, b = _footprint_on(_fit_of(upper), _fit_of(lower), c, 0.0, t=t, across=across)
    assert a < a0 - 0.3 and b > b0 + 0.3
    for q in (a - 0.05, b + 0.05):  # a support line just outside the interval clears the pavement everywhere
        for u in np.linspace(*across, 28):
            _s, u_l = lower.alignment.project(*upper.alignment.offset_point(q + t * u, float(u)))
            assert abs(u_l) > 5.0 - 1e-6


def test_deck_prefilter_keeps_every_point_the_projection_would_keep():
    from roader.structures.infer import _near

    model = twin_model()
    fit = _fit_of(model.road(model.structures[0].carried_road))
    xy = np.random.default_rng(7).uniform(-400.0, 400.0, (20000, 2))
    s, _u, ok = fit.line.project(xy, max_dist=30.0)
    want = set(np.flatnonzero(ok & (s >= 40.0) & (s <= 90.0)).tolist())
    assert want and want <= set(_near(fit, xy, 40.0, 90.0, 30.0).tolist())


def test_vectorized_surface_matches_the_road_surface():
    from roader.structures.infer import _surface

    road = twin_model().roads[0]
    s = np.linspace(-20.0, road.length + 20.0, 401)
    u = np.linspace(-8.0, 8.0, 401)
    want = [road.surface_z(float(a), float(b)) for a, b in zip(s, u, strict=True)]
    assert np.allclose(_surface(road, s, u), want, atol=1e-9)


def test_fr_cr_3_ramp_without_a_run_of_its_own_still_shares_the_deck():
    """A loop leaving 9 m before the deck end has too short a deck-class run for a structure;
    its first metres still ride the parent's deck (Woburn NE loop off I-93 northbound)."""
    model = twin_model(ramp=True)
    roads = {r.id: r for r in model.roads}
    west = model.structures[0]
    extents = [StructureExtent(west.carried_road, west.s_start, west.s_end, "class17", None, None)]
    kept, shares, _flags = absorb_diverging(extents, model.links, roads)
    assert len(kept) == 1 and [s.road_id for s in shares] == [model.roads[-1].id]


def test_fr_st_4_support_skew_follows_the_crossing_up_to_the_cap():
    from roader.structures.infer import MAX_SUPPORT_SKEW_RAD, support_skew

    assert support_skew([0.4, 0.6]) == (pytest.approx(0.5), [])
    skew, flags = support_skew([-math.radians(78.7)])  # Dallas ramp nearly parallel to the road below
    assert skew == pytest.approx(-MAX_SUPPORT_SKEW_RAD) and flags == ["FR-ST-4:skew_capped_from_-78.7deg"]
    assert support_skew([]) == (0.0, [])


def test_fr_st_4_each_support_takes_the_skew_of_what_it_crosses():
    from roader.structures.infer import support_skews

    # A viaduct 0-300 m over two roads; only its far end span passes over one.
    crossed = [(0.3, (60.0, 90.0)), (-0.5, (271.0, 295.0)), (0.1, (400.0, 420.0))]
    skews = [sk for sk, _f in support_skews([0.0, 40.0, 75.0, 150.0, 290.0, 300.0], crossed)]
    # Piers follow the nearest crossing; an end over nothing takes its neighbour's skew (D-025), the far end its road's.
    assert skews == [0.3, 0.3, 0.3, 0.3, -0.5, -0.5]
    assert [sk for sk, _f in support_skews([0.0, 30.0], [(0.4, (100.0, 120.0))])] == [0.0, 0.0]  # a span over nothing
    assert support_skews([0.0, 20.0], []) == [(0.0, []), (0.0, [])]
    (sk, flags), _ = support_skews([0.0, 20.0], [(math.radians(-75.0), (-5.0, 25.0))])
    assert sk == pytest.approx(-math.radians(60.0)) and flags == ["FR-ST-4:skew_capped_from_-75.0deg"]


def test_fr_sg_2_skew_field_is_sweepable_and_only_rounds_supports_toward_square():
    import itertools

    from roader.solids.stations import SkewKnot, _sweepable, skew_at

    rate = 0.01
    # Deck 2-60 m on a 100 m road, abutment at the road start; piers at 30 m (+1.0) and 40 m (-1.0); one mild pier.
    knots = [SkewKnot(2.0, 0.5, False), SkewKnot(30.0, 1.0, False), SkewKnot(40.0, -1.0, False), SkewKnot(50.0, 0.05, False), SkewKnot(60.0, 0.5, False), SkewKnot(90.0, 0.0, True)]
    half = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    out = _sweepable(knots, half, rate, 100.0)
    assert skew_at(out, 0.0) == 0.0 and skew_at(out, 100.0) == 0.0  # a station line at a road end stays on the road
    for a, b in itertools.pairwise(out):
        assert abs(b.t - a.t) <= rate * (b.s - a.s) + 1e-12
    for k, h in zip(knots, half, strict=True):
        v = skew_at(out, k.s)
        assert v * k.t >= 0.0 and abs(v) <= abs(k.t) + 1e-12  # never past square, never steeper
        if h:
            assert skew_at(out, k.s - h) == skew_at(out, k.s + h) == v  # a cap is one support line
    feasible = [SkewKnot(10.0, 0.0, True), SkewKnot(40.0, 0.2, False), SkewKnot(60.0, 0.2, False), SkewKnot(90.0, 0.0, True)]
    assert [(k.s, k.t) for k in _sweepable(feasible, [0.0] * 4, rate, 100.0)][1:-1] == [(k.s, k.t) for k in feasible]


def test_fr_st_3_long_span_over_merged_clear_zones_takes_piers_in_gores_off_the_pavements():
    from roader.rules import RuleBook
    from roader.structures.infer import _even_piers

    rules = RuleBook.load()
    # A 300 m stack deck whose crossed roads' clear zones merge into one 20-280 m zone;
    # their pavements leave gores at 100-120 and 180-200 m.
    extent = StructureExtent("rd", 0.0, 300.0, "class17", None, None)
    zones = [(20.0, 280.0)]
    pavements = [(25.0, 100.0), (120.0, 180.0), (200.0, 275.0)]
    piers, even, in_zone = _even_piers(extent, [], zones, 2.0, rules, "*", pavements)
    spans = np.diff([0.0, *piers, 300.0])
    assert in_zone and in_zone <= set(piers) and even >= in_zone
    assert all(not (a - 1.0 < p < b + 1.0) for p in piers for a, b in pavements)  # never on a lower pavement
    assert spans.max() < 120.0
    # Clear zones with room between them: no pier goes inside one.
    _piers2, _even2, in_zone2 = _even_piers(extent, [], [(20.0, 100.0), (180.0, 280.0)], 2.0, rules, "*", [(25.0, 95.0), (185.0, 275.0)])
    assert not in_zone2


def test_fr_st_5_rule_depth_leaves_the_vertical_clearance_over_every_road_spanned():
    from roader.structures.infer import clearance_limited_depth

    # A long stack span gives a 11.8 m rule depth; roads pass 8.4 and 16.9 m below the deck.
    depth, flag = clearance_limited_depth(11.8, [(8.4, 4.877), (16.9, 4.877), (0.0, 4.877)], 0.2)
    assert depth == pytest.approx(8.4 - 4.877) and flag == "FR-ST-5:depth_limited_by_clearance:8.40m_below"
    assert clearance_limited_depth(1.5, [(8.4, 4.877)], 0.2) == (1.5, None)  # already clear
    assert clearance_limited_depth(2.0, [(3.0, 4.877), (0.02, 4.3)], 0.2) == (2.0, None)  # no room under it: not a road it spans
