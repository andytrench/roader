"""Shared pavement, merged areas and areas beside decks on synthetic scenes (FR-SG-3, FR-SV-5, FR-SV-7, AC-H1, D-069, D-070)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from roader.complete.junctions import _snap_rings, build_junctions
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
