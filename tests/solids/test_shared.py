"""Shared pavement, merged areas and areas beside decks on synthetic scenes (FR-SG-3, FR-SV-5, FR-SV-7, AC-H1, D-069, D-070)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from roader.complete.junctions import _conform, _snap_rings, build_junctions
from roader.model import dumps, loads
from roader.model.geometry import AlignElement, Alignment, Profile, VerticalElement
from roader.model.schema import DeckSpec, Link, Road, RoadModel, Structure, Support
from roader.model.serialize import stable_id
from roader.model.synthetic import RULE, TRUTH, _section, diverge_model
from roader.solids import generate
from roader.validate.gates import check_continuity
from roader.validate.profile import check_profile
from roader.validate.seams import junction_violations

from . import meshcheck as mc

Z = 100.0


def _road(name: str, elements: list[tuple[str, float, float]], x0: float, y0: float, heading: float,
          half_l: float, half_r: float, kind: str = "ramp") -> Road:
    """A flat road from (kind, length, curvature) pieces."""
    out, s, x, y, h = [], 0.0, x0, y0, heading
    for k, length, curv in elements:
        e = AlignElement(k, s, length, x, y, h, curv, curv, TRUTH)
        out.append(e)
        x, y, h, _k = e.end_state()
        s += length
    return Road(
        id=stable_id("rd", "scene", name),
        name=name,
        road_class="motorway_link" if kind == "ramp" else "motorway",
        kind=kind,
        oneway=True,
        alignment=Alignment(out),
        profile=Profile([VerticalElement(0.0, s, Z, 0.0, 0.0, TRUTH)]),
        cross_section=_section(half_l, half_r, 2, 0.3),
    )


def _model(roads: list[Road], *, terrain: bool = False, links=None, structures=None) -> RoadModel:
    base = diverge_model(with_junctions=False)
    model = dataclasses.replace(base, roads=roads, side_slopes=[], links=links or [], structures=structures or [],
                                terrain=base.terrain if terrain else None)
    model = loads(dumps(model))
    model.junctions, model.flags = build_junctions(model)
    return loads(dumps(model))


def _plan(mesh) -> Polygon:
    v = mesh.vertices
    return unary_union([Polygon(v[f, :2]) for f in mesh.faces if abs(Polygon(v[f, :2]).area) > 1e-9])


def _closed_ok(meshes):
    closed = [m for m in meshes if m.closed]
    assert all(mc.solid_problems(m) == [] for m in closed), [(m.id, mc.solid_problems(m)) for m in closed if mc.solid_problems(m)]
    assert check_profile(closed) == []


# ---------------------------------------------------------------- parallel ramps that separate late (D-070)


def _parallel(terrain: bool = False) -> RoadModel:
    """Ramp B starts beside ramp A, overlaps its left 1 m for 150 m, then curves away on a 300 m radius."""
    a = _road("a", [("line", 400.0, 0.0)], -200.0, 0.0, 0.0, 4.0, 4.0)
    b = _road("b", [("line", 150.0, 0.0), ("arc", 150.0, 1.0 / 300.0)], -150.0, 6.5, 0.0, 3.5, 3.5)
    return _model([a, b], terrain=terrain)


@pytest.fixture(scope="module")
def parallel():
    return _parallel()


def test_fr_sg_3_parallel_ramps_get_one_shared_area_each(parallel):
    a, b = parallel.roads
    assert len(parallel.junctions) == 2
    assert all(jn.kind == "shared" and len(jn.road_ids) == 1 for jn in parallel.junctions)
    assert sorted(jn.road_ids[0] for jn in parallel.junctions) == sorted([a.id, b.id])
    codes = [f.code for f in parallel.flags]
    assert "FR-SG-3:shared_split" in codes and "FR-SV-7:pavement_overlap" not in codes
    for jn in parallel.junctions:
        assert "FR-SG-3:shared_split" in jn.prov.flags
    by_road = {jn.road_ids[0]: jn for jn in parallel.junctions}
    # A keeps seams at both ends of its interval; B starts inside the overlap, so only its far end is a seam.
    assert sorted(s.side for s in by_road[a.id].seams) == [-1, 1]
    assert [s.side for s in by_road[b.id].seams] == [-1]
    lo, hi = sorted(s.s for s in by_road[a.id].seams)
    assert 45.0 < lo < 50.0 and 200.0 < hi < 280.0


def test_fr_sv_7_shared_areas_partition_the_overlap_midway(parallel):
    """The two areas meet without overlap; the split runs midway between A's left edge (y = 4) and B's right edge (y = 3)."""
    pa, pb = (Polygon(j.outline, j.holes) for j in parallel.junctions)
    assert pa.is_valid and pb.is_valid
    assert pa.intersection(pb).area < 1e-6
    touch = pa.boundary.intersection(pb.boundary)
    assert touch.length > 140.0
    ys = [y for x, y in np.asarray(touch.coords if touch.geom_type == "LineString" else
                                   [c for g in touch.geoms for c in g.coords]) if -140.0 < x < -10.0]
    assert ys and all(abs(y - 3.5) < 0.02 for y in ys)


def test_d021_shared_split_vertices_conform(parallel):
    """Every outline vertex lying on the other area's outline is also one of its vertices."""
    ja, jb = parallel.junctions
    for x, y in (ja, jb), (jb, ja):
        other = set(y.outline)
        ring = Polygon(y.outline).exterior
        for p in x.outline:
            if ring.distance(Polygon([p, (p[0] + 1e-9, p[1]), (p[0], p[1] + 1e-9)]).centroid) < 1e-7:
                assert p in other


def test_ac_h1_shared_area_solids(parallel):
    meshes = generate(parallel)
    _closed_ok(meshes)
    for m in meshes:
        if m.closed and m.layer == "L3":
            for jn in parallel.junctions:
                assert _plan(m).intersection(Polygon(jn.outline)).area < 1e-5, (m.id, jn.id)
    assert check_continuity(parallel).passed


def test_ac_h1_shared_areas_over_terrain_and_realtime():
    from roader.validate.realtime import check_realtime

    model = _parallel(terrain=True)
    meshes = generate(model)
    _closed_ok(meshes)
    assert any(m.part == "embankment" and m.layer == "L6" for m in meshes)
    rt = generate(model, profile="realtime")
    assert check_realtime([m for m in rt if m.closed]) == []


def test_fr_sg_3_contained_road_keeps_its_pavement():
    """A narrow road drawn wholly on a wide one keeps all of it; the wide road's area goes round it."""
    wide = _road("wide", [("line", 300.0, 0.0)], 0.0, 0.0, 0.0, 10.0, 10.0, kind="mainline")
    narrow = _road("narrow", [("line", 100.0, 0.0)], 100.0, -5.0, 0.0, 3.0, 3.0)
    model = _model([wide, narrow])
    by_road = {jn.road_ids[0]: jn for jn in model.junctions}
    assert set(by_road) == {wide.id, narrow.id}
    jn_n = by_road[narrow.id]
    assert jn_n.seams == []  # the whole road lies on the other
    assert Polygon(jn_n.outline).area == pytest.approx(100.0 * 6.0, rel=1e-3)
    pw = Polygon(by_road[wide.id].outline, by_road[wide.id].holes)
    assert pw.intersection(Polygon(jn_n.outline)).area < 1e-6
    meshes = generate(model)
    assert not [m for m in meshes if m.element_id == narrow.id]
    _closed_ok(meshes)


def test_fr_sg_3_roads_at_different_levels_are_not_shared():
    a = _road("lo", [("line", 300.0, 0.0)], 0.0, 0.0, 0.0, 4.0, 4.0)
    b = _road("hi", [("line", 200.0, 0.0)], 50.0, 6.0, 0.0, 3.5, 3.5)
    b.profile = Profile([VerticalElement(0.0, 200.0, Z + 6.0, 0.0, 0.0, TRUTH)])
    model = _model([a, b])
    assert model.junctions == [] and not [f for f in model.flags if f.code.startswith("FR-SV-7")]


# ---------------------------------------------------------------- overlapping junction areas merge (D-069)


def _twin_diverges() -> RoadModel:
    """Two carriageways whose medians overlap locally, each with an exit ramp: their junction areas overlap."""
    fa = _road("fa", [("line", 400.0, 0.0)], -200.0, 0.0, 0.0, 5.5, 5.5, kind="mainline")
    tilt = 0.01
    fb = _road("fb", [("line", 300.0, 0.0)], -10.0, 10.4 + 0.01 * -10.0, tilt, 5.5, 5.5, kind="mainline")
    ra = _road("ra", [("line", 10.0, 0.0), ("arc", 150.0, -1.0 / 300.0)], 0.0, -2.75, 0.0, 2.5, 2.5)
    x0 = 5.0
    y0 = 10.4 + 0.01 * x0
    rb = _road("rb", [("line", 10.0, 0.0), ("arc", 150.0, 1.0 / 300.0)], x0 - 2.75 * math.sin(tilt), y0 + 2.75 * math.cos(tilt), tilt, 2.5, 2.5)
    links = [
        Link(stable_id("lk", fa.id, ra.id), fa.id, "end", ra.id, "start", "diverge", from_s=200.0, prov=RULE),
        Link(stable_id("lk", fb.id, rb.id), fb.id, "end", rb.id, "start", "diverge", from_s=15.0, prov=RULE),
    ]
    return _model([fa, fb, ra, rb], links=links)


def test_d069_overlapping_junction_areas_are_merged():
    model = _twin_diverges()
    junctions = [j for j in model.junctions if j.kind != "shared"]
    assert len(junctions) == 1, [(j.id, j.road_ids) for j in model.junctions]
    assert len(junctions[0].road_ids) == 4
    polys = [Polygon(j.outline, j.holes) for j in model.junctions]
    assert all(p.is_valid for p in polys)
    assert not [f for f in model.flags if f.code in ("FR-SG-3:overlap", "FR-SV-7:pavement_overlap")], model.flags
    assert junction_violations(junctions[0], {r.id: r for r in model.roads})[0] == []
    _closed_ok(generate(model))


# ---------------------------------------------------------------- areas beside decks (D-069)


def _with_bridge(skew: float, s0: float, carried: int = 0, dz: float = 0.0) -> RoadModel:
    base = diverge_model(with_junctions=False)
    road = base.roads[carried]
    if dz:
        road = dataclasses.replace(road, id=stable_id("rd", "scene", "over"), name="over")
        x0 = 30.0
        road.alignment = Alignment([AlignElement("line", 0.0, 200.0, x0, -100.0, math.pi / 2, 0.0, 0.0, TRUTH)])
        road.profile = Profile([VerticalElement(0.0, 200.0, Z + 1.0 + dz, 0.0, 0.0, TRUTH)])
        base.roads.append(road)
    sid = stable_id("st", "scene", s0)
    s1 = s0 + 40.0
    supports = [
        Support(stable_id("sp", sid, "a1"), s0, "abutment", skew, 0, 0.0, 1.2, 1.5, 95.0, RULE),
        Support(stable_id("sp", sid, "a2"), s1, "abutment", skew, 0, 0.0, 1.2, 1.5, 95.0, RULE),
    ]
    st = Structure(sid, "bridge", road.id, s0, s1, "multi_girder", None, [], DeckSpec(1.3, 0.22, 0.9, RULE), supports, TRUTH)
    model = dataclasses.replace(base, structures=[st], side_slopes=[ss for ss in base.side_slopes if ss.road_id != road.id])
    model = loads(dumps(model))
    model.junctions, model.flags = build_junctions(model)
    return loads(dumps(model))


def test_d069_junction_beside_a_square_deck_is_built():
    """The freeway interval ends at s = 272.5; a square bridge from s = 275 no longer blocks the area (was 30 m)."""
    model = _with_bridge(0.0, 275.0)
    assert [j.kind for j in model.junctions] == ["diverge"], model.flags
    jn = model.junctions[0]
    hi = max(s.s for s in jn.seams if s.road_id == model.roads[0].id)
    assert hi < 275.0 - 1.0
    _closed_ok(generate(dataclasses.replace(model, terrain=None, side_slopes=[])))


def test_d073_seam_in_a_skewed_deck_ramp_follows_the_station_line():
    """Near a skewed deck end the seam is the grid's skewed row (here the deck end itself) and the solids still conform."""
    model = _with_bridge(0.4, 275.0)
    assert [j.kind for j in model.junctions] == ["diverge"], model.flags
    bare = dataclasses.replace(model, terrain=None, side_slopes=[])
    meshes = generate(bare)
    _closed_ok(meshes)
    jn = model.junctions[0]
    slab = next(m for m in meshes if m.id == f"{jn.id}:pavement")
    fwy = model.roads[0]
    seam = max((s for s in jn.seams if s.road_id == fwy.id), key=lambda s: s.s)
    t = math.tan(0.4)
    corner = fwy.alignment.offset_point(seam.s + t * 5.5, 5.5)
    assert min(math.dist(v[:2], corner) for v in slab.vertices) < 1e-3  # the row is the deck's skewed end
    assert seam.skew == pytest.approx(t)
    # AC-H4 reads the straight section at the seam: the area answers for it on both sides of the skewed row.
    assert junction_violations(jn, {r.id: r for r in model.roads})[0] == []
    deck = next(m for m in meshes if m.part == "deck")
    keys = {tuple(np.round(v, 9)) for v in deck.vertices}
    shared = [v for v in slab.vertices if tuple(np.round(v, 9)) in keys]
    assert len(shared) >= 4  # the interval stops at the deck: slab and deck share its skewed end row


def test_d073_seam_inside_a_skew_ramp():
    """A strongly skewed deck 20 m past the junction: the seam falls on a ramped (skewed) station line."""
    model = _with_bridge(0.9, 295.0)
    assert [j.kind for j in model.junctions] == ["diverge"], model.flags
    fwy = model.roads[0]
    hi = max(s.s for s in model.junctions[0].seams if s.road_id == fwy.id)
    assert 270.0 < hi < 287.0
    meshes = generate(dataclasses.replace(model, terrain=None, side_slopes=[]))
    _closed_ok(meshes)
    slab = next(m for m in meshes if m.id == f"{model.junctions[0].id}:pavement")
    on_straight_row = [v for v in slab.vertices if abs(fwy.alignment.project(v[0], v[1])[0] - hi) < 1e-6]
    assert len(on_straight_row) < 4


def test_d069_area_under_a_high_deck_is_built_but_not_at_its_level():
    high = _with_bridge(0.0, 80.0, dz=7.0)
    assert [j.kind for j in high.junctions] == ["diverge"], high.flags
    low = _with_bridge(0.0, 80.0, dz=0.3)
    assert not [j for j in low.junctions if j.kind == "diverge"]
    assert any("at the level of the area" in f.message for f in low.flags)


# ---------------------------------------------------------------- gore side slopes (D-071)


def test_d071_gore_side_slopes_stop_at_the_pavement_edge():
    """Two roads 1.5 m apart, each on fill towards the other: the slopes would overlap; both stop at their edges."""
    from roader.model.schema import SideSlope

    a = _road("ga", [("line", 300.0, 0.0)], -150.0, 0.0, 0.0, 4.0, 4.0)
    b = _road("gb", [("line", 300.0, 0.0)], -150.0, 10.0, 0.0, 4.5, 4.5)
    slopes = [
        SideSlope(stable_id("ss", a.id), a.id, "left", 0.0, 300.0, "fill", 2.0, RULE),
        SideSlope(stable_id("ss", b.id), b.id, "right", 0.0, 300.0, "fill", 2.0, RULE),
    ]
    model = dataclasses.replace(_model([a, b], terrain=True), side_slopes=slopes)
    assert model.junctions == []
    meshes = generate(model)
    _closed_ok(meshes)
    fills = [_plan(m) for m in meshes if m.part == "embankment"]
    assert len(fills) == 2 and fills[0].intersection(fills[1]).area < 1e-6
    assert all(f.bounds[1] > -1e-6 and f.bounds[3] < 10.0 + 1e-6 for f in fills)  # kept under their own pavements
    assert any(m.id.startswith("terrain") for m in meshes)


# ---------------------------------------------------------------- outline hygiene (FR-SV-5)


def test_fr_sv_5_near_coincident_outline_vertices_merge():
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0000004, 5.0000003), (0.0, 5.0)]
    (out,) = _snap_rings([ring], set())
    assert len(out) == 4
    spike = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (12.0, 5.0000001), (9.0, 5.0), (0.0, 5.0)]
    (out,) = _snap_rings([spike], set())
    assert (12.0, 5.0000001) not in out and Polygon(out).is_valid


def test_d159_outlines_tracing_one_edge_with_different_vertices_conform():
    """FR-SV-7, D-021: Woburn jn_8030 (I-95 WB) and jn_b31a (its ramp) traced their common edge with
    different vertices, 38 um to 2.9 mm off each other's edges; their fill tops were near-coplanar
    faces 38 um apart, which the crossing test cannot tell from a crossing (AC-H1 held the unit)."""
    a = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
    b = [(0.0, -5.0), (10.0, -5.0), (10.0, 0.0), (7.0, -0.0029), (3.0, 0.000038), (0.0, 0.0)]
    a2, b2 = _conform([a, b])
    assert (7.0, -0.0029) in a2 and (3.0, 0.000038) in a2
    pa, pb = Polygon(a2), Polygon(b2)
    assert pa.is_valid and pb.is_valid
    assert pa.intersection(pb).area < 1e-12 and abs(pa.union(pb).area - pa.area - pb.area) < 1e-12
    assert pa.intersection(pb).length > 10.0 - 1e-6  # one shared edge, no gap


def test_fr_sv_5_seam_row_wins_over_a_near_outline_vertex():
    """An outline vertex 1e-6 m from a seam row end (the jn_bbe2 sliver) is dropped in favour of the road's row."""
    model = diverge_model()
    jn = model.junctions[0]
    k = next(i for i, p in enumerate(jn.outline) if any(math.dist(p, model.road(s.road_id).alignment.offset_point(s.s, s.u1)) < 1e-5
                                                          for s in jn.seams))
    x, y = jn.outline[k]
    jn.outline.insert(k + 1, (x + 1e-6, y))
    meshes = generate(dataclasses.replace(model, terrain=None, side_slopes=[]))
    _closed_ok(meshes)


def test_d090_split_tip_left_inside_a_neighbour_is_dropped():
    """A vertex the 1 cm snap leaves 1.6 cm inside the neighbouring region is removed (Woburn 13e5/1dda tip)."""
    from roader.complete.junctions import _unpoke

    a = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (5.0, 4.016), (0.0, 4.0)]
    b = [(0.0, 4.0), (10.0, 4.0), (10.0, 8.0), (0.0, 8.0)]
    out = _unpoke([a, b], [(0, -1), (1, -1)], set())
    assert (5.0, 4.016) not in out[0] and out[1] == b
    assert Polygon(out[0]).intersection(Polygon(out[1])).area == 0.0


def test_d092_realtime_uv_on_partly_trimmed_gore_fill():
    """Real-time profile: a fill trimmed over part of its length keeps uv0 within AC-P4 (D-092 guard; the
    Woburn case needs a curved, high fill to show the jump and is checked on that unit)."""
    from roader.model.schema import SideSlope
    from roader.validate.assets import check_uv_stretch

    a = _road("ta", [("line", 300.0, 0.0)], -150.0, 0.0, 0.0, 4.0, 4.0)
    b = _road("tb", [("line", 300.0, 0.0)], -150.0, 13.0, 0.06, 4.5, 4.5)
    for r in (a, b):  # 3.7 m of fill: the slope arc is well longer than its plan width
        r.profile = Profile([VerticalElement(0.0, 300.0, Z + 3.0, 0.0, 0.0, TRUTH)])
    slopes = [
        SideSlope(stable_id("ss", a.id), a.id, "left", 0.0, 300.0, "fill", 2.0, RULE),
        SideSlope(stable_id("ss", b.id), b.id, "right", 0.0, 300.0, "fill", 2.0, RULE),
    ]
    model = dataclasses.replace(_model([a, b], terrain=True), side_slopes=slopes)
    analytic = generate(model)
    fills = [m for m in analytic if m.part == "embankment"]
    assert any(len(m.faces) for m in fills)
    rt = generate(model, profile="realtime")
    result = check_uv_stretch([m for m in rt if m.closed])
    assert result.passed, result.violations


def test_d130_seam_patch_takes_the_road_slopes_at_the_seam_not_past_a_nearby_knot():
    """A cross-slope knot 5 cm past a seam: the patch's slope rates are the road's at the seam on the junction side (AC-H4, D-130)."""
    from roader.complete.junctions import _patch
    from roader.model.geometry import Knot, Piecewise

    road = diverge_model().roads[0]
    s = 248.80
    knots = [Knot(0.0, -0.02, TRUTH), Knot(s + 0.05, 0.005, TRUTH), Knot(s + 20.0, 0.02, TRUTH)]
    road.cross_section.slope_left = Piecewise(knots)
    seam = _patch(road, s, 1)
    before = (0.005 + 0.02) / (s + 0.05)
    assert seam.slope_left_rate == pytest.approx(before, rel=1e-6)
    behind = _patch(road, s + 0.10, -1)  # looking back across the knot from beyond it
    assert behind.slope_left_rate == pytest.approx((0.02 - 0.005) / 19.95, rel=1e-6)


# ---------------------------------------------------------------- a road entering an area alongside a participant (D-136)


def _merge_beside_a_road_end() -> RoadModel:
    """Ramp R comes in beside road L, runs over L's last 40 m and merges into M 60 m past M's start, where L ends (a trumpet stem)."""
    m = _road("m", [("line", 300.0, 0.0)], 0.0, 0.0, 0.0, 5.5, 5.5, kind="mainline")
    lead = _road("l", [("line", 200.0, 0.0)], -200.0, 0.0, 0.0, 3.0, 3.0)
    a, rad = 0.15, 300.0
    x0 = -40.0 - 40.0 * math.cos(a) - rad * math.sin(a)
    y0 = -5.0 - 40.0 * math.sin(a) - rad * (1.0 - math.cos(a))
    r = _road("r", [("line", 40.0, 0.0), ("arc", a * rad, -1.0 / rad), ("line", 100.0, 0.0)], x0, y0, a, 3.0, 3.0)
    links = [Link(stable_id("lk", r.id, m.id), r.id, "end", m.id, "start", "merge", to_s=60.0, prov=RULE)]
    return _model([m, lead, r], links=links)


def test_d136_a_road_entering_a_merge_area_alongside_joins_it():
    model = _merge_beside_a_road_end()
    roads = {r.id: r for r in model.roads}
    junctions = [j for j in model.junctions if j.kind != "shared"]
    assert len(junctions) == 1 and sorted(junctions[0].road_ids) == sorted(roads), [(j.kind, j.road_ids) for j in model.junctions]
    bad = [f for f in model.flags if f.code in ("FR-SG-3:overlap", "FR-SG-3:shared_skipped", "FR-SV-7:pavement_overlap")]
    assert not bad, bad
    assert junction_violations(junctions[0], roads)[0] == []
    _closed_ok(generate(model))


# ---------------------------------------------------------------- abutting pavements: shared regions end where they part (D-137)


def test_d137_abutting_carriageways_split_only_where_they_overlap():
    """Twin carriageways 0.5 m apart; B bends onto A over its last 6 m. An exit from A at s = 250 has its own area.

    B's row clears A by 0.25 m about 4 m before the overlap, so the shared regions stay clear of the exit's
    area; a 1 m gore gap never clears beside B and would run them into it.
    """
    a = _road("a", [("line", 400.0, 0.0)], 0.0, 0.0, 0.0, 5.5, 5.5, kind="mainline")
    b = _road("b", [("line", 380.0, 0.0), ("arc", 20.0, -1.0 / 200.0)], 0.0, 11.5, 0.0, 5.5, 5.5, kind="mainline")
    ra = _road("ra", [("line", 10.0, 0.0), ("arc", 150.0, -1.0 / 300.0)], 250.0, -2.75, 0.0, 2.5, 2.5)
    links = [Link(stable_id("lk", a.id, ra.id), a.id, "end", ra.id, "start", "diverge", from_s=250.0, prov=RULE)]
    model = _model([a, b, ra], links=links)
    shared = sorted((j for j in model.junctions if j.kind == "shared"), key=lambda j: j.road_ids[0] != a.id)
    assert [j.road_ids[0] for j in shared] == [a.id, b.id], [(j.kind, j.road_ids) for j in model.junctions]
    assert 380.0 < min(s.s for s in shared[0].seams) < 392.0
    bad = [f for f in model.flags if f.code in ("FR-SG-3:shared_skipped", "FR-SV-7:pavement_overlap")]
    assert not bad, bad
    _closed_ok(generate(model))


# ---------------------------------------------------------------- no-crash contract (FR-EX-1, D-119, D-150)


def _two_diverges() -> RoadModel:
    """Two mainlines 200 m apart, each with an exit ramp: two independent junction areas."""
    roads, links = [], []
    for k, y in enumerate((0.0, 200.0)):
        main = _road(f"main{k}", [("line", 400.0, 0.0)], -200.0, y, 0.0, 5.5, 5.5, kind="mainline")
        exit_ = _road(f"exit{k}", [("line", 10.0, 0.0), ("arc", 150.0, -1.0 / 300.0)], 0.0, y - 2.75, 0.0, 2.5, 2.5)
        roads += [main, exit_]
        links.append(Link(stable_id("lk", main.id, exit_.id), main.id, "end", exit_.id, "start", "diverge", from_s=200.0, prov=RULE))
    return _model(roads, links=links)


def test_d150_a_raise_in_one_area_drops_that_area_and_holds_the_unit(monkeypatch):
    import roader.complete.junctions as J
    from roader.model.schema import ELEMENT_FAILED
    from roader.validate.gates import check_degraded

    healthy = _two_diverges()
    assert len(healthy.junctions) == 2 and check_degraded(healthy) is None
    bad = stable_id("rd", "scene", "exit0")
    real = J._junction

    def flaky(model, roads, nodes, intervals):
        if bad in intervals:
            raise RuntimeError("forced")
        return real(model, roads, nodes, intervals)

    monkeypatch.setattr(J, "_junction", flaky)
    model = _two_diverges()
    # The failed diverge's overlap falls back to shared pavement (D-070); the other diverge is untouched.
    assert [sorted(j.road_ids) for j in model.junctions if j.kind != "shared"] == [sorted(stable_id("rd", "scene", n) for n in ("main1", "exit1"))]
    failed = sorted({f.element_id for f in model.flags if f.code == ELEMENT_FAILED})
    assert failed == sorted([bad, stable_id("rd", "scene", "main0")])
    assert all("RuntimeError: forced" in f.message for f in model.flags if f.code == ELEMENT_FAILED)
    gate = check_degraded(model)
    assert gate is not None and not gate.passed
    _closed_ok(generate(model))


def test_d150_a_raise_in_a_shared_region_drops_only_that_region(monkeypatch):
    import roader.complete.junctions as J
    from roader.model.schema import ELEMENT_FAILED

    def boom(*_a, **_k):
        raise ValueError("forced")

    monkeypatch.setattr(J, "_shared_group", boom)
    model = _parallel()
    assert model.junctions == []
    assert sorted(f.element_id for f in model.flags if f.code == ELEMENT_FAILED) == sorted(r.id for r in model.roads)


def test_d150_build_junctions_never_raises(monkeypatch):
    import roader.complete.junctions as J
    from roader.model.schema import ELEMENT_FAILED

    def boom(*_a, **_k):
        raise RuntimeError("forced")

    monkeypatch.setattr(J, "_touches", boom)
    model = _two_diverges()
    assert model.junctions == []
    assert sorted(f.element_id for f in model.flags if f.code == ELEMENT_FAILED) == sorted(r.id for r in model.roads)


# ---------------------------------------------------------------- streets crossing at grade (D-151)


def test_d151_streets_crossing_at_grade_get_one_intersection_area():
    """Two through streets crossing at one level (a shared vector node, no link) meet in one area, seamed on all four legs."""
    a = _road("ave", [("line", 300.0, 0.0)], -150.0, 0.0, 0.0, 6.0, 6.0, kind="crossing_street")
    b = _road("st", [("line", 300.0, 0.0)], 20.0, -150.0, math.pi / 2.0 - 0.2, 5.0, 5.0, kind="crossing_street")
    model = _model([a, b])
    assert [(j.kind, sorted(j.road_ids)) for j in model.junctions] == [("intersection", sorted([a.id, b.id]))]
    assert sorted(s.road_id for s in model.junctions[0].seams) == sorted([a.id, a.id, b.id, b.id])
    assert not [f for f in model.flags if f.code in ("FR-SV-7:pavement_overlap", "FR-SG-3:crossing_overlap")], model.flags
    assert junction_violations(model.junctions[0], {r.id: r for r in model.roads})[0] == []
    _closed_ok(generate(model))

    b.profile = Profile([VerticalElement(0.0, 300.0, Z + 7.0, 0.0, 0.0, TRUTH)])
    assert _model([a, b]).junctions == []

    # Two ramps crossing on one level at 35 degrees (Bellevue links 169303084 / 161930315) also meet in one area.
    r1 = _road("r1", [("line", 300.0, 0.0)], -150.0, 0.0, 0.0, 4.0, 4.0)
    r2 = _road("r2", [("line", 300.0, 0.0)], -150.0 * math.cos(0.61), -150.0 * math.sin(0.61), 0.61, 3.7, 3.7)
    acute = _model([r1, r2])
    assert [sorted(j.road_ids) for j in acute.junctions] == [sorted([r1.id, r2.id])]
    assert not [f for f in acute.flags if f.code in ("FR-SV-7:pavement_overlap", "FR-SG-3:shared_skipped")], acute.flags


def test_d152_two_road_ends_meeting_at_a_corner_get_one_area():
    """A two-way avenue and a one-way street start at one node 65 degrees apart (Bellevue 116th Ave / NE 8th St)."""
    av = _road("av", [("line", 200.0, 0.0)], 0.0, 0.0, math.pi / 2.0, 6.0, 6.0, kind="crossing_street")
    st = _road("st", [("line", 200.0, 0.0)], 0.0, 0.0, math.pi / 2.0 - 1.13, 5.0, 5.0, kind="crossing_street")
    model = _model([av, st])
    assert [(j.kind, sorted(j.road_ids)) for j in model.junctions] == [("intersection", sorted([av.id, st.id]))]
    assert not [f for f in model.flags if f.code.startswith("FR-SV-7")], model.flags
    _closed_ok(generate(model))


def test_d153_an_avenue_across_a_divided_street_regrows_against_both_halves():
    """Two crossing nodes on the avenue merge; each half's interval, grown against the avenue only,
    still ends inside the other half's pavement until the merged group is regrown (Bellevue NE 12th St)."""
    c1 = _road("c1", [("line", 400.0, 0.0)], -200.0, 0.0, 0.0, 4.0, 4.0, kind="crossing_street")
    tilt = math.atan2(7.0, 400.0)
    c2 = _road("c2", [("line", 400.0, 0.0)], 200.0 * math.cos(tilt), 11.0, math.pi + tilt, 4.0, 4.0, kind="crossing_street")
    av = _road("av", [("line", 300.0, 0.0)], 0.0, -150.0, math.pi / 2.0, 6.0, 6.0, kind="crossing_street")
    model = _model([c1, c2, av])
    assert [(j.kind, sorted(j.road_ids)) for j in model.junctions if j.kind != "shared"] == [("intersection", sorted([c1.id, c2.id, av.id]))]
    bad = [f for f in model.flags if f.code in ("FR-SV-7:pavement_overlap", "FR-SG-3:skipped", "FR-SG-3:shared_skipped")]
    assert not bad, bad
    _closed_ok(generate(model))



def test_d136_a_participant_running_back_into_its_area_beyond_its_interval_is_regrown():
    """A participant's own pavement outside its interval inside the area (Bellevue NE 12th St halves) is found like a third road."""
    import roader.complete.junctions as J

    model = _merge_beside_a_road_end()
    roads = {r.id: r for r in model.roads}
    area = next(j for j in model.junctions if j.kind != "shared")
    r = stable_id("rd", "scene", "r")
    lo_r = min(s.s for s in area.seams if s.road_id == r)
    intervals = {rid: (0.0, roads[rid].length) for rid in area.road_ids if rid != r}
    intervals[r] = (lo_r + 30.0, roads[r].length)
    J._set_spans(model)
    hit = J._entering(roads, area, Polygon(area.outline, area.holes), intervals, {}, set())
    assert hit is not None and hit[0] == r
    assert hit[1][0] < lo_r + 30.0 - 5.0 and hit[1][1] == pytest.approx(roads[r].length)


# ---------------------------------------------------------------- a shared stretch starting at an area's seam (D-154)


def _on_plane(road: Road, y: float, dz: float = 0.0) -> Road:
    """Put a road ``dz`` over the plane of a crowned mainline's right half (z = Z - 0.02 |y|, y < 0)."""
    from roader.model.geometry import constant

    road.profile = Profile([VerticalElement(0.0, road.length, Z + 0.02 * y + dz, 0.0, 0.0, TRUTH)])
    road.cross_section.slope_left = constant(0.02, TRUTH)
    road.cross_section.slope_right = constant(0.02, TRUTH)
    return road


def _ramp_beside_after_diverge() -> RoadModel:
    """Ramp R diverges from A, then overlaps mainline B's edge by 1 m to its end (Bellevue link 995574460 beside I-405 NB).

    A and R are fitted 8 cm over B's surface there, as at Bellevue."""
    b = _road("b", [("line", 400.0, 0.0)], -100.0, 0.0, 0.0, 5.5, 5.5, kind="mainline")
    a = _on_plane(_road("a", [("line", 240.0, 0.0), ("arc", 150.0, -1.0 / 200.0)], -300.0, -9.5, 0.0, 3.0, 3.0), -9.5, 0.08)
    r = _on_plane(_road("r", [("line", 260.0, 0.0)], -60.0, -7.5, 0.0, 3.0, 3.0), -7.5, 0.08)
    links = [Link(stable_id("lk", a.id, r.id), a.id, "end", r.id, "start", "diverge", from_s=240.0, prov=RULE)]
    return _model([a, b, r], links=links)


def test_d154_a_ramp_riding_beside_a_road_after_its_diverge_shares_from_the_area_seam():
    """R never clears B, so B cannot join the diverge area (D-136); the R/B stretch starts at the area's seam on R,
    B's region stops at the area's outline and the area takes the regions' vertices there."""
    model = _ramp_beside_after_diverge()
    roads = {r.id: r for r in model.roads}
    a, b, r = (stable_id("rd", "scene", n) for n in ("a", "b", "r"))
    area = next(j for j in model.junctions if j.kind != "shared")
    assert sorted(area.road_ids) == sorted([a, r])
    shared = {j.road_ids[0]: j for j in model.junctions if j.kind == "shared"}
    assert set(shared) == {b, r}, [(j.kind, j.road_ids) for j in model.junctions]
    seam_r = next(s.s for s in area.seams if s.road_id == r)
    assert [s.s for s in shared[r].seams if s.side > 0] == [pytest.approx(seam_r)]
    bad = [f for f in model.flags if f.code in ("FR-SV-7:pavement_overlap", "FR-SG-3:shared_skipped", "FR-SG-3:overlap")]
    assert not bad, bad
    polys = [Polygon(j.outline, j.holes) for j in model.junctions]
    assert all(p.is_valid for p in polys)
    assert all(polys[i].intersection(polys[j]).area < 1e-4 for i in range(len(polys)) for j in range(i + 1, len(polys)))
    assert junction_violations(area, roads)[0] == []
    _closed_ok(generate(model))


def _end_beyond_a_deck(skew: float = 0.0) -> RoadModel:
    """Ramp A ends 0.5 m past its deck end; ramp B starts 0.1 m before A's end and overlaps it (Lorain 19124514 / 19126770)."""
    a = _road("a", [("line", 100.0, 0.0)], -100.0, 0.0, 0.0, 4.0, 4.0)
    b = _road("b", [("line", 100.0, 0.0)], -0.1, 0.3, 0.02, 4.0, 4.0)
    sid = stable_id("st", "scene", "a")
    supports = [Support(stable_id("sp", sid, k), s, "abutment", skew, 0, 0.0, 1.2, 1.5, 95.0, RULE) for k, s in (("a1", 59.5), ("a2", 99.5))]
    st = Structure(sid, "bridge", a.id, 59.5, 99.5, "multi_girder", None, [], DeckSpec(1.3, 0.22, 0.9, RULE), supports, TRUTH)
    return _model([a, b], structures=[st])


def test_d172_a_short_shared_piece_at_a_road_end_gets_its_own_region():
    """FR-SG-3, FR-SV-7: A's part of the shared pavement (0.5 m between its deck end and its road end) was
    under the 1 m seam margin, so A got no region and kept its pavement there, while B's region took the
    whole overlap: road and area overlapped (Lorain rd_3160 / jn_b475)."""
    model = _end_beyond_a_deck()
    assert not [f for f in model.flags if f.code == "FR-SV-7:pavement_overlap"], [f.message for f in model.flags]
    assert any(j.kind == "shared" for j in model.junctions)
    _closed_ok(generate(model))


def test_d172_areas_overlapping_along_a_short_link_meet_at_one_seam(monkeypatch):
    """FR-SG-3, FR-SV-7, D-069, D-090: a link leaves P and joins Q 15 m away; its diverge and merge areas
    overlap along it. Lorain: the merge could not be built (the link would lie wholly inside it) and the
    split left a seam row inside the other area, so the areas overlapped by 21 m2 (jn_9813 / jn_d6b1)."""
    import roader.complete.junctions as J

    regrow = J._regrow
    monkeypatch.setattr(J, "_regrow", lambda roads, iv: None if len(iv) > 2 else regrow(roads, iv))
    monkeypatch.setattr(J, "_merge", lambda groups: groups)
    monkeypatch.setattr(J, "_split_pair", lambda *args, **kwargs: ([], "no split"))
    gap, radius = 15.0, 200.0
    turn = math.acos(1.0 - (gap - 4.0) / (2.0 * radius))
    p = _road("p", [("line", 300.0, 0.0)], -150.0, 0.0, 0.0, 3.5, 3.5, kind="mainline")
    q = _road("q", [("line", 300.0, 0.0)], -150.0, -gap, 0.0, 3.5, 3.5, kind="mainline")
    link = _road("l", [("arc", radius * turn, -1.0 / radius), ("arc", radius * turn, 1.0 / radius)], -20.0, -2.0, 0.0, 3.5, 3.5)
    model = _model([p, q, link])
    assert sorted(j.kind for j in model.junctions) == ["diverge", "merge"], [f.message for f in model.flags]
    assert not [f for f in model.flags if f.code in ("FR-SG-3:overlap", "FR-SV-7:pavement_overlap")]
    at = [round(s.s, 6) for j in model.junctions for s in j.seams if s.road_id == link.id]
    assert len(at) == 2 and at[0] == at[1]
    _closed_ok(generate(model))


def test_d173_a_region_at_a_skewed_deck_end_near_the_road_end_meets_the_squared_grid_row():
    """FR-SG-3, D-073, D-118: the region's seam follows the deck end's full skew (-0.066 rad, Lorain
    st_7ab3), but 0.5 m from the road end the grid squares the row off; the area was left out."""
    model = _end_beyond_a_deck(skew=-0.066)
    assert any(j.kind == "shared" and j.seams for j in model.junctions), [f.message for f in model.flags]
    failures = []
    meshes = generate(model, failures=failures)
    assert not [f for f in failures if f.code == "FR-EX-1:element_failed"], [f.message for f in failures]
    _closed_ok(meshes)
