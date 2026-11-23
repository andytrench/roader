"""Curved, banked road with a skewed bridge on the curve (FR-SG-1, FR-SG-2, FR-SV-1..7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from roader.model import (
    AlignElement,
    Alignment,
    Barrier,
    CrossSection,
    DeckSpec,
    Frame,
    Knot,
    Piecewise,
    Profile,
    Provenance,
    Road,
    RoadModel,
    SideSlope,
    Source,
    Structure,
    Support,
    TerrainPatch,
    UnitRecord,
    VerticalElement,
    constant,
)
from roader.solids import generate

from . import meshcheck as mc
from .test_overpass import _surface_hits, check_terrain

TOL = 0.01
OBS = Provenance(Source.MEASURED, 0.9)
RULE = Provenance(Source.RULE, 0.5, rule_id="test")


def _pw(*pairs: tuple[float, float]) -> Piecewise:
    return Piecewise([Knot(s, v, OBS) for s, v in pairs])


def _tilted(x: float, y: float) -> float:
    return 98.0 + 0.004 * x - 0.003 * y


def _rolling(x: float, y: float) -> float:
    return 97.5 + 0.6 * math.sin(x / 17.0) * math.cos(y / 23.0) + 0.002 * x


def curved_model(ground=_tilted) -> RoadModel:
    k = 1.0 / 120.0
    spec = [("line", 40.0, 0.0, 0.0), ("spiral", 50.0, 0.0, k), ("arc", 70.0, k, k), ("spiral", 50.0, k, 0.0), ("line", 40.0, 0.0, 0.0)]
    elements, s, x, y, h = [], 0.0, 0.0, 0.0, 0.3
    for kind, length, k0, k1 in spec:
        e = AlignElement(kind, s, length, x, y, h, k0, k1, OBS)
        elements.append(e)
        x, y, h, _ = e.end_state()
        s += length
    z1 = 103.0 + 0.02 * 80.0
    z2 = z1 + 0.02 * 100.0 - 0.5 * 0.04 * 100.0
    profile = Profile(
        [
            VerticalElement(0.0, 80.0, 103.0, 0.02, 0.02, OBS),
            VerticalElement(80.0, 100.0, z1, 0.02, -0.02, OBS),
            VerticalElement(180.0, 70.0, z2, -0.02, -0.02, OBS),
        ]
    )
    xs = CrossSection(
        width_left=_pw((100.0, 5.5), (140.0, 6.5)),
        width_right=constant(5.5, OBS),
        slope_left=_pw((40.0, -0.02), (90.0, -0.06), (160.0, -0.06), (210.0, 0.02)),
        slope_right=_pw((40.0, 0.02), (90.0, -0.06), (160.0, -0.06), (210.0, 0.02)),
        pavement_thickness=constant(0.3, RULE),
        barriers=[
            Barrier("guard_r", "right", 20.0, 95.0, 4.5, 0.8, 0.6, 0.2, "jersey", RULE),
            Barrier("par_l", "left", 105.0, 145.0, 6.6, 1.0, 0.4, 0.25, "parapet", RULE),
            Barrier("par_r", "right", 105.0, 145.0, 5.6, 1.0, 0.4, 0.4, "parapet", RULE),
        ],
    )
    road = Road("rd_curve", "Ramp", "motorway_link", "ramp", True, Alignment(elements), profile, xs)
    skew = 0.15
    st = Structure(
        "st_curve", "bridge", road.id, 105.0, 145.0, "slab", None, [],
        DeckSpec(1.2, 0.25, 0.5, RULE),
        [
            Support("sp_a1", 105.0, "abutment", skew, 0, 0.0, 1.0, 1.2, 96.0, RULE),
            Support("sp_p1", 125.0, "pier", skew, 2, 0.9, 1.0, 1.2, 96.0, RULE),
            Support("sp_a2", 145.0, "abutment", skew, 0, 0.0, 1.0, 1.2, 96.0, RULE),
        ],
        OBS,
    )
    slopes = [
        SideSlope("ss_l", road.id, "left", 0.0, road.length, "fill", 3.0, RULE),
        SideSlope("ss_r", road.id, "right", 0.0, road.length, "fill", 3.0, RULE),
    ]
    cell = 2.0
    ox, oy, nx, ny = -60.0, -60.0, 141, 141
    z = [ground(ox + i * cell, oy + j * cell) for j in range(ny) for i in range(nx)]
    terrain = TerrainPatch(ox, oy, cell, nx, ny, z, OBS)
    unit = UnitRecord("curved", (0.0, 0.0, 0.0, 0.0), "test")
    frame = Frame("+proj=tmerc +lat_0=42 +lon_0=-72 +ellps=GRS80 +units=m", -72.0, 42.0)
    return RoadModel(unit, frame, roads=[road], structures=[st], side_slopes=slopes, terrain=terrain)


@pytest.fixture(scope="module")
def model() -> RoadModel:
    return curved_model()


@pytest.fixture(scope="module")
def meshes(model):
    return generate(model, chord_tol_m=TOL)


def test_curved_bodies(meshes):
    parts = sorted(m.id for m in meshes)
    assert "rd_curve:pavement:0" in parts and "rd_curve:pavement:1" in parts
    assert "guard_r:barrier" in parts and "par_l:parapet" in parts and "par_r:parapet" in parts
    assert "ss_l:embankment:0" in parts and "ss_r:embankment:1" in parts
    assert "st_curve:deck" in parts and "sp_p1:pier_cap" in parts and "sp_p1:pier_column:1" in parts


def test_curved_solids_valid_fr_sv_1_to_6(meshes):
    for m in meshes:
        if m.closed:
            assert mc.solid_problems(m) == [], m.id


def test_curved_no_crossings_or_overlap_fr_sv_7(meshes):
    closed = [m for m in meshes if m.closed]
    assert mc.crossing_pairs(closed) == []
    for mesh in closed:
        pts = mc.interior_samples(mesh)
        for other in closed:
            if other is not mesh and len(pts):
                assert not mc.inside(pts, other).any(), (mesh.id, other.id)
    for a in meshes:
        for b in meshes:
            if a is not b and mc.boxes_overlap(a, b):
                assert mc.t_junctions(a, b) == 0, (a.id, b.id)


def test_curved_structure_union_fr_sv_7(meshes):
    parts = [m for m in meshes if m.element_id in {"st_curve", "sp_a1", "sp_p1", "sp_a2", "par_l", "par_r"}]
    verts, faces = mc.merge_union(parts)
    from roader.model import Mesh

    union = Mesh("u", "u", "L5", "deck", "x", "rule", verts, faces)
    assert mc.closed_and_oriented(union) == [] and mc.components(union) == 1


def test_chord_tolerance_fr_sg_2(model, meshes):
    road = model.roads[0]
    xs = road.cross_section
    tops = [m for m in meshes if m.id.startswith("rd_curve:pavement") or m.id == "st_curve:deck"]
    s = np.arange(0.25, road.length - 0.25, 0.37)
    # Vertical: the drivable surface matches the model surface at arbitrary points.
    worst = 0.0
    for u_of in (lambda s: 0.0, lambda s: 0.5 * xs.width_left(s), lambda s: -0.5 * xs.width_right(s)):
        pts = np.array([road.alignment.offset_point(float(v), u_of(float(v))) for v in s])
        true = np.array([road.surface_z(float(v), u_of(float(v))) for v in s])
        hits = [np.concatenate([_surface_hits(m, pts[n : n + 1])[0] for m in tops]) for n in range(len(pts))]
        for z, h in zip(true, hits, strict=True):
            assert len(h)
            worst = max(worst, float(np.min(np.abs(h.max() - z))))
    assert worst <= TOL
    # Plan: the true pavement edges lie within tol of the generated outline.
    outline = unary_union(
        [Polygon(t[:, :2]) for m in tops for t in m.vertices[m.faces] if Polygon(t[:, :2]).area > 1e-9]
    ).boundary
    for sign, width in ((1.0, xs.width_left), (-1.0, xs.width_right)):
        for v in s:
            v = float(v)
            if 104.0 < v < 146.0:
                continue  # the deck is wider than the pavement
            x, y = road.alignment.offset_point(v, sign * width(v))
            assert outline.distance(Point(x, y)) <= TOL


def test_curved_terrain_fr_sg_6(model, meshes):
    terrain = check_terrain(model, meshes, TOL)
    assert len(terrain) == 1  # fills end inside the patch


def test_rolling_terrain_fr_sg_6():
    model = curved_model(_rolling)
    meshes = generate(model, chord_tol_m=TOL)
    for m in meshes:
        if m.closed:
            assert mc.solid_problems(m) == [], m.id
    terrain = check_terrain(model, meshes, TOL)
    for a in terrain:
        for b in meshes:
            if b.closed and mc.boxes_overlap(a, b):
                assert mc.t_junctions(a, b) == 0 and mc.t_junctions(b, a) == 0, b.id
    # Relief keeps fine cells; planar blocks still merge where the surface is flat enough.
    n = sum(len(m.faces) for m in terrain)
    t = model.terrain
    assert 0 < n < 2 * (t.nx - 1) * (t.ny - 1)


def test_curved_determinism_ac_h5(model, meshes):
    again = generate(model, chord_tol_m=TOL)
    for a, b in zip(meshes, again, strict=True):
        assert a.id == b.id and np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def test_coarser_tolerance_uses_fewer_triangles(model, meshes):
    coarse = generate(model, chord_tol_m=0.25)
    fine = sum(len(m.faces) for m in meshes if m.closed)
    assert sum(len(m.faces) for m in coarse if m.closed) < fine
    assert all(mc.solid_problems(m) == [] for m in coarse if m.closed)


def test_skew_follows_supports_fr_st_4(model, meshes):
    road, st = model.roads[0], model.structures[0]
    t = math.tan(st.supports[0].skew_rad)
    abut = next(m for m in meshes if m.id == "sp_a1:abutment")
    plan = np.unique(abut.vertices[:, :2], axis=0)
    su = np.array([road.alignment.project(float(x), float(y), step=0.5) for x, y in plan])
    on_back = np.abs(su[:, 0] - t * su[:, 1] - st.s_start) < 1e-6
    assert on_back.sum() >= 3
    assert np.ptp(su[on_back, 1]) > 10.0  # the skewed back face spans the deck width


def test_fr_sg_2_only_the_inside_of_a_curve_can_fold():
    """A reach beyond the radius on the outside of a curve is buildable; on the inside it is refused."""
    from roader.solids.stations import max_step

    road = curved_model().roads[0]
    a, b = 100.0, 140.0  # on the left-turning 120 m arc: the centre is on the left
    far = 150.0
    assert max_step(road, a, b, 0.01, far, lambda _a, _b: (10.0, far)) > 0.0
    with pytest.raises(ValueError, match="centre of curvature"):
        max_step(road, a, b, 0.01, far, lambda _a, _b: (far, 10.0))
    with pytest.raises(ValueError, match="centre of curvature"):
        max_step(road, a, b, 0.01, far)


def test_fr_sg_2_a_width_kink_refines_only_the_station_lines_that_meet_its_side():
    """Skewed lines s0 + t u meet a left-edge kink only from u >= 0 (AC-P4, D-122).

    Refining every line within the full skew reach also refined the lines on the far
    side of the road, which at Woburn cut a 25 m embankment into 0.2 m needles.
    """
    from roader.solids.stations import SkewKnot, stations

    road = curved_model().roads[0]  # left width kink at s = 140 (0.025 m/m)
    t, umax, tol = 0.15, 30.0, 0.01
    knots = [SkewKnot(0.0, t, False), SkewKnot(road.length, t, False)]
    out = np.array(stations(road, [120.0, 130.0, 140.0, 150.0, 160.0], tol, umax, knots))
    fine = 4.0 * tol / 0.025
    meets = np.diff(out[(out >= 130.0) & (out <= 140.0)])  # left-side lines reach s = 140 from here
    behind = np.diff(out[(out >= 140.0) & (out <= 150.0)])  # these cross s = 140 only on the right side
    assert meets.max() <= fine + 1e-9
    assert behind.max() > fine
