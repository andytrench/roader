"""Milestone 2 layers on a synthetic four-level stack: FR-RS-2, FR-VC-5, FR-CR-1..3, FR-OC-1..3, AC-H2."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

from roader.conflate.crossings import input_crossings
from roader.conflate.graph import RoadSeed
from roader.conflate.polyline import Polyline
from roader.conflate.structures import StructureExtent, chain_extents, rejoins_ground
from roader.io import from_synthetic
from roader.model.synthetic import overpass_model
from roader.pipeline import run_pipeline
from tests.pipeline.stack_scene import (
    FREEWAY_HALF,
    RAMP_HALF,
    TAPER,
    stack_points,
    stack_scene,
    stack_vectors,
)

LEVELS = ("eb", "wb", "l1", "l2", "l3")


@pytest.fixture(scope="module")
def scene():
    return stack_scene()


@pytest.fixture(scope="module")
def stack(scene):
    sp, owner = stack_points(scene)
    base = overpass_model()
    res = run_pipeline(from_synthetic(sp), stack_vectors(scene), base.frame, base.unit)
    return sp, owner, res


def _road(model, fid):
    return next(r for r in model.roads if r.vector_refs[0].feature_id == fid)


def _truth(scene, fid):
    return next(r for r in scene.roads if r.vector_refs[0].feature_id == fid)


# ---------------------------------------------------------------- FR-RS-2


def test_fr_rs_2_cells_under_deck_edges_hold_two_paved_sheets(stack):
    lay = stack[2].evidence.layers
    keys, _first, _count = lay.cell_ranges()
    paved = (stack[2].evidence.layer_prob >= 0.5) & (lay.sheet >= 0)
    per_cell = np.bincount(np.searchsorted(keys, lay.key[paved]), minlength=len(keys))
    assert (per_cell >= 2).sum() >= 20, "oblique returns under deck edges see the road below"


def test_fr_rs_2_every_level_is_its_own_sheet_where_levels_overlap(stack):
    _sp, owner, res = stack
    ev = res.evidence
    sheet = ev.point_sheet
    own = owner[ev.index]
    near = np.hypot(ev.xyz[:, 0], ev.xyz[:, 1]) <= 40.0
    level = {"eb": 0, "wb": 0, "l1": 1, "l2": 2, "l3": 3}
    sheets_of = {}
    for fid in LEVELS:
        m = near & (own == fid) & (sheet >= 0)
        assert m.sum() > 100, fid
        sheets_of.setdefault(level[fid], set()).update(np.unique(sheet[m]).tolist())
    # A road hidden under a higher deck may break into two sheets; no sheet spans two levels.
    for a in range(4):
        for b in range(a + 1, 4):
            assert not sheets_of[a] & sheets_of[b], (a, b)
    assert len(sheets_of[3]) == 1


def test_fr_rs_2_no_sheet_holds_two_heights_in_one_cell(stack):
    lay = stack[2].evidence.layers
    m = lay.sheet >= 0
    pairs = np.unique(np.column_stack([lay.key[m], lay.sheet[m]]), axis=0)
    assert len(pairs) == int(m.sum())


# ---------------------------------------------------------------- FR-VC-5


def test_fr_vc_5_unvectored_ramp_is_one_candidate_with_geometry(stack, scene):
    res = stack[2]
    assert len(res.candidates) == 1
    cand = res.candidates[0]
    poly = shapely.Polygon(cand.outline)
    ramp = shapely.LineString(scene.unvectored[30:])
    assert ramp.intersection(poly).length >= 0.8 * ramp.length
    assert 7.0 <= cand.width_m <= 12.0
    for road in res.model.roads:
        line = shapely.LineString(Polyline.from_alignment(road.alignment).xy)
        assert poly.intersection(line).length < 5.0, road.name
    flag = next(f for f in res.model.flags if f.code == "FR-VC-5:candidate")
    assert flag.element_id == cand.id and "POLYGON" in flag.message


def test_fr_vc_5_sheet_shared_with_a_road_is_still_reported(stack, scene):
    """The candidate's sheet is the eastbound one (joined at the taper): per-sheet matching would miss it."""
    res = stack[2]
    eb = next(f for f in res.fits if f.seed.vector_refs[0].feature_id == "eb")
    used = {int(s) for s in eb.track.sheet if s >= 0}
    assert set(res.candidates[0].sheets) & used
    assert TAPER[1] > TAPER[0]


# ---------------------------------------------------------------- FR-CR-1, AC-H2


def test_fr_cr_1_every_crossing_is_ordered_by_level(stack, scene):
    model = stack[2].model
    fid = {r.id: r.vector_refs[0].feature_id for r in model.roads}
    level = {"eb": 0, "wb": 0, "l1": 1, "l2": 2, "l3": 3}
    pairs = {(fid[c.upper_road], fid[c.lower_road]) for c in model.crossings}
    expected = {(u, lo) for u in ("l1", "l2", "l3") for lo in LEVELS if level[lo] < level[u]}
    assert pairs == expected
    assert len(model.crossings) == len(expected)
    for c in model.crossings:
        truth = 8.0 * (level[fid[c.upper_road]] - level[fid[c.lower_road]])
        assert abs(c.dz_m - truth) < 0.15, (fid[c.upper_road], fid[c.lower_road], c.dz_m)


def test_fr_cr_1_roads_meeting_at_a_node_do_not_cross():
    main = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    ramp = np.array([[50.0, -30.0], [100.0, 0.0]])
    over = np.array([[150.0, -50.0], [150.0, 50.0]])

    def seed(xy, name):
        return RoadSeed(name, name, "motorway", "mainline", True, Polyline(xy), [], nodes=xy)

    assert input_crossings(seed(main, "m"), seed(ramp, "r")) == []
    pts = input_crossings(seed(main, "m"), seed(over, "o"))
    assert len(pts) == 1 and abs(pts[0].x - 150.0) < 1e-9


# ---------------------------------------------------------------- FR-CR-3


def test_fr_cr_3_one_structure_per_elevated_level(stack, scene):
    model = stack[2].model
    fid = {r.id: r.vector_refs[0].feature_id for r in model.roads}
    by_road = {}
    for st in model.structures:
        by_road.setdefault(fid[st.carried_road], []).append(st)
    assert sorted(by_road) == ["l1", "l2", "l3"]
    for f, sts in by_road.items():
        assert len(sts) == 1, f
        st = sts[0]
        on = [c for c in model.crossings if c.upper_road == st.carried_road]
        assert {c.structure_id for c in on} == {st.id}
        assert sorted(st.crossed_roads) == sorted({c.lower_road for c in on})
        assert len(st.crossed_roads) == len(set(st.crossed_roads))


def test_fr_cr_3_viaduct_ends_from_data(stack, scene):
    model = stack[2].model
    for f in ("l1", "l2", "l3"):
        road, truth = _road(model, f), _truth(scene, f)
        st = next(s for s in model.structures if s.carried_road == road.id)
        # Model stations start at the clipped input; compare in plan.
        for s_model, s_truth in ((st.s_start, scene.decks[f][0]), (st.s_end, scene.decks[f][1])):
            xm = np.array(road.alignment.offset_point(s_model, 0.0))
            xt = np.array(truth.alignment.offset_point(s_truth, 0.0))
            assert np.hypot(*(xm - xt)) < 4.0, (f, s_model, s_truth)


def test_fr_st_3_piers_never_in_a_lower_travel_lane(stack, scene):
    model = stack[2].model
    for st in model.structures:
        upper = model.road(st.carried_road)
        assert all(st.s_start < sp.station < st.s_end for sp in st.supports if sp.kind == "pier")
        spans = np.diff(sorted(sp.station for sp in st.supports))
        assert spans.max() <= 70.0
        for sp in st.supports:
            if sp.kind != "pier":
                continue
            x, y = upper.alignment.offset_point(sp.station, 0.0)
            for lid in st.crossed_roads:
                lower = model.road(lid)
                s, u = lower.alignment.project(x, y)
                if 0.0 <= s <= lower.length:
                    half = lower.cross_section.width_left(s) + 0.5
                    assert abs(u) * abs(math.cos(sp.skew_rad)) > half or abs(u) > half, (st.id, sp.station, lower.name)


def _drop(stations, left, right):
    return np.asarray(stations, float), np.asarray(left, float), np.asarray(right, float)


def test_fr_cr_3_chain_merges_runs_that_never_touch_ground():
    st = np.arange(0.0, 200.0, 1.0)
    deep = np.where((st >= 20) & (st <= 180), -7.0, -0.2)
    unseen = deep.copy()
    unseen[(st > 80) & (st < 100)] = np.nan  # under a higher deck
    runs = [StructureExtent("r", 20.0, 80.0, "class17", (20.0, 80.0), (20.0, 80.0)), StructureExtent("r", 100.0, 180.0, "sheet_break", (100.0, 180.0), None)]
    out = chain_extents(runs, {"r": _drop(st, unseen, unseen)})
    assert len(out) == 1 and (out[0].s_start, out[0].s_end) == (20.0, 180.0)
    assert out[0].method == "sheet_break" and len(out[0].chained) == 2


def test_fr_cr_3_chain_keeps_bridges_apart_when_the_road_lands_between():
    st = np.arange(0.0, 200.0, 1.0)
    drop = np.where(((st >= 20) & (st <= 80)) | ((st >= 100) & (st <= 180)), -7.0, -0.5)
    assert rejoins_ground(_drop(st, drop, drop), 80.0, 100.0)
    runs = [StructureExtent("r", 20.0, 80.0, "class17", None, None), StructureExtent("r", 100.0, 180.0, "class17", None, None)]
    assert len(chain_extents(runs, {"r": _drop(st, drop, drop)})) == 2


def test_fr_cr_3_twin_deck_beside_one_side_is_not_ground():
    st = np.arange(0.0, 100.0, 1.0)
    left = np.full(st.size, 0.1)  # the other carriageway's deck
    right = np.full(st.size, -7.0)
    assert not rejoins_ground(_drop(st, left, right), 10.0, 90.0)


def test_fr_cr_3_overlapping_rule_extents_merge_and_clamp_to_the_road():
    st = np.arange(0.0, 100.0, 1.0)
    d = np.full(st.size, -7.0)
    runs = [StructureExtent("r", -10.0, 30.0, "rule", None, None), StructureExtent("r", 25.0, 60.0, "rule", None, None)]
    out = chain_extents(runs, {"r": _drop(st, d, d)}, {"r": 100.0})
    assert len(out) == 1 and out[0].s_start == 0.0 and out[0].s_end == 60.0 and out[0].clipped


# ---------------------------------------------------------------- FR-OC-1..3


def test_fr_oc_1_every_hidden_gap_is_completed_within_its_bound(stack, scene):
    res = stack[2]
    model = res.model
    gaps = [(r, iv) for r in model.roads for iv in r.support if iv.kind == "occluded"]
    fids = {r.vector_refs[0].feature_id for r, _ in gaps}
    assert {"eb", "wb", "l1", "l2"} <= fids  # ground, level 1 and level 2 are hidden by the decks above
    assert "l3" not in fids
    by_gap = {(c.road_id, round(c.s0, 3)): c for c in res.completions}
    for road, iv in gaps:
        c = by_gap[(road.id, round(iv.s0, 3))]
        assert c.sides == 2 and c.bound_m is not None and c.bound_m <= 0.25, (road.name, c)
        truth = _truth(scene, road.vector_refs[0].feature_id)
        tl = Polyline.from_alignment(truth.alignment)
        s = np.linspace(iv.s0, iv.s1, 9)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        st, _u, _ok = tl.project(xy)
        err = [abs(road.profile.z(float(a)) - truth.profile.z(float(b))) for a, b in zip(s, st, strict=True)]
        assert max(err) <= c.bound_m + 0.03, (road.name, max(err), c.bound_m)
    flags = [f for f in model.flags if f.code == "FR-OC-1:completed"]
    assert len(flags) == len(gaps)
    assert not [f for f in model.flags if f.code in ("FR-OC-1:uncertain", "FR-OC-1:unbounded")]


def test_fr_oc_1_gap_under_three_decks_names_them(stack):
    res = stack[2]
    model = res.model
    eb = _road(model, "eb")
    c = next(c for c in res.completions if c.road_id == eb.id and c.reason == "under_deck")
    assert len(c.above) == 3


def test_fr_oc_1_canopy_gap_is_completed_not_under_deck(stack):
    res = stack[2]
    wb = _road(res.model, "wb")
    c = [c for c in res.completions if c.road_id == wb.id and c.reason != "under_deck"]
    assert len(c) == 1 and not c[0].above
    x, _y = wb.alignment.offset_point(0.5 * (c[0].s0 + c[0].s1), 0.0)
    assert abs(x + 200.0) < 10.0


def test_fr_oc_2_stacked_clearances_are_physical(stack):
    model = stack[2].model
    for c in model.crossings:
        assert c.structure_id is not None
        assert c.clearance_m is not None and c.clearance_m >= 4.877, (c.id, c.clearance_m)
    assert not [f for f in model.flags if f.req == "FR-OC-2"]


def test_fr_oc_3_ground_under_the_stack_is_below_every_deck(stack, scene):
    model = stack[2].model
    t = model.terrain
    assert t is not None
    for x, y in ((0.0, 0.0), (5.0, -3.0), (-4.0, 4.0)):
        ix, iy = int((x - t.origin_x) // t.cell_m), int((y - t.origin_y) // t.cell_m)
        k = iy * t.nx + ix
        z = t.z[k]
        assert z is not None and z < 100.0 - 0.3, (x, y, z)
        assert t.fill[k] != 0  # never a measured node: nothing was seen there
    assert FREEWAY_HALF > RAMP_HALF
