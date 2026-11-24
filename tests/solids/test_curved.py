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
    """A reach beyond the radius on the outside of a curve is buildable; on the inside it only shortens the step.

    The reach ``sides`` gives is padded, so it no longer refuses the sweep (D-142): lines that
    really reach the centre are held short of it in the bodies, and the grid still refuses a fold.
    """
    from roader.solids.stations import max_step

    road = curved_model().roads[0]
    a, b = 100.0, 140.0  # on the left-turning 120 m arc: the centre is on the left
    far = 150.0
    outside = max_step(road, a, b, 0.01, far, lambda _a, _b: (10.0, far))
    inside = max_step(road, a, b, 0.01, far, lambda _a, _b: (far, 10.0))
    assert 0.0 < inside < outside
    assert 0.0 < max_step(road, a, b, 0.01, far) <= inside


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


def test_padded_reach_never_aborts_a_sweep_that_does_not_fold_fr_sg_2():
    """Toes at 0.8 R on the inside of the curve sweep; only the padded reach used to come near the centre (D-142)."""
    m = curved_model(ground=lambda x, y: _tilted(x, y) - 25.0)
    notes = []
    meshes = generate(m, chord_tol_m=TOL, failures=notes)
    assert notes == []
    assert all(mc.solid_problems(x) == [] for x in meshes if x.closed)


def test_toe_reaching_the_centre_of_curvature_is_held_and_flagged_fr_sg_2():
    """A toe past the centre of curvature is held at 1 - k u = FOLD_LIMIT and flagged, not a crash (D-142)."""
    import dataclasses

    from roader.solids.bodies import FOLD_LIMIT

    m = curved_model(ground=lambda x, y: _tilted(x, y) - 45.0)
    st = m.structures[0]
    m.structures = [dataclasses.replace(st, supports=[dataclasses.replace(sp, skew_rad=0.0) for sp in st.supports])]
    notes = []
    meshes = generate(m, chord_tol_m=TOL, failures=notes)
    assert {(f.code, f.element_id) for f in notes} == {("FR-SG-2:fold_limited", "ss_l")}
    assert all(mc.solid_problems(x) == [] for x in meshes if x.closed)
    road = m.roads[0]
    emb = [x for x in meshes if x.element_id == "ss_l"]
    for x in emb:
        s, u = road.alignment.project_many(x.vertices[:, :2])
        k = np.array([road.alignment.evaluate(float(v))[3] for v in s])
        assert np.all(1.0 - k * u >= FOLD_LIMIT - 1e-6)


def _ridged(x: float, y: float) -> float:
    return _tilted(x, y) - 1.0 + 1.5 * math.sin(y / 4.0)


def _skewed_over_ridges(skew: float) -> RoadModel:
    import dataclasses

    m = curved_model(_ridged)
    st = m.structures[0]
    abutments = [dataclasses.replace(sp, skew_rad=skew, footing_z=85.0) for sp in st.supports if sp.kind == "abutment"]
    m.structures = [dataclasses.replace(st, supports=abutments)]
    return m


def test_fr_sg_2_a_57_degree_deck_over_a_steep_toe_is_swept_skewed_and_flagged():
    """Embankment toes on ridged ground run nearly parallel to 57-degree station lines (D-161).

    Their intersections jumped between rows, the station lines crossed and the road fell
    back to square structures. Before that fallback, |tan skew| is now limited where an
    active toe's |du/ds| would make |tan skew| |du/ds| exceed TOE_SKEW_MAX, tapering like
    a skew ramp.
    """
    from roader.solids.stations import TOE_SKEW_MAX

    m = _skewed_over_ridges(1.0)
    notes = []
    meshes = generate(m, chord_tol_m=TOL, failures=notes)
    assert [f for f in notes if f.code == "FR-EX-1:element_failed"] == []
    # D-161 retry chain: toes pulled in alone (D-156) do not hold, so the skew is limited as well.
    assert {(f.code, f.element_id) for f in notes} == {
        ("FR-SG-2:toe_limited", "ss_l"), ("FR-SG-2:toe_limited", "ss_r"), ("FR-SG-2:skew_limited", "st_curve"),
    }
    assert all(mc.solid_problems(x) == [] for x in meshes if x.closed)
    assert TOE_SKEW_MAX == 0.5
    road = m.roads[0]
    # The abutments stay skewed (FR-ST-4), less than the supports where the toes bind: a square
    # 1 m abutment spans 1 m of station, the full 1.56 skew over the 14 m deck about 23 m.
    for sp_id in ("sp_a1", "sp_a2"):
        abut = next(x for x in meshes if x.id == f"{sp_id}:abutment")
        s, _u = road.alignment.project_many(np.unique(abut.vertices[:, :2], axis=0))
        assert 3.0 < np.ptp(s) < 23.0, sp_id


def test_fr_sg_2_a_road_that_sweeps_at_full_skew_keeps_it_over_steep_toes():
    """The toe limit only replaces the square fallback (D-161): a 26-degree deck over the same
    ridged toes, whose |tan skew| |du/ds| exceeds TOE_SKEW_MAX in places but whose grid holds,
    keeps the support skew. At Woburn the blanket limit squared a 33-degree pier into the road below.
    """
    from roader.solids.bodies import _toe
    from roader.solids.stations import TOE_SKEW_MAX, toe_samples
    from roader.solids.terrain import TerrainSampler

    skew = 0.45
    m = _skewed_over_ridges(skew)
    road, t = m.roads[0], math.tan(skew)
    sampler = TerrainSampler(m.terrain)
    toes = [(_toe(road, sign, 3.0, sampler, "ss"), a, b) for sign in (1.0, -1.0) for a, b in ((0.0, 105.0), (145.0, road.length))]
    assert toe_samples(toes, [(0.0, road.length)], t), "the limit would bind here"
    assert TOE_SKEW_MAX / t < 2.0
    notes = []
    meshes = generate(m, chord_tol_m=TOL, failures=notes)
    assert notes == []
    abut = next(x for x in meshes if x.id == "sp_a1:abutment")
    su = road.alignment.project_many(np.unique(abut.vertices[:, :2], axis=0))
    on_back = np.abs(su[0] - t * su[1] - m.structures[0].s_start) < 1e-6
    assert on_back.sum() >= 3


def test_fr_sg_2_skew_limit_binds_only_lines_that_meet_the_steep_toe_and_tapers():
    """A line with |t| <= L meets a toe sample at distance d only if L |u| >= d: L <= max(0.5 / rate, d / |u|) (D-161).

    Only rows where the toe is active solve it; the bound tapers over the lateral reach of each pair of rows.
    """
    from roader.solids.stations import RAMP_FACTOR, toe_skew_limit

    tmax = 1.6
    sweep = [float(s) for s in range(101)]
    wide = [100.0] * 100
    toe = (50.0, 2.0, 5.0, 0.0, 100.0)  # (station, |du/ds|, |u|, toe active from, to)
    caps = toe_skew_limit([toe], sweep, wide, tmax)
    assert caps[50] == pytest.approx(0.25)
    assert caps[51] == pytest.approx(0.25)  # lines within 0.25 |u| meet the steep sample at any allowed skew
    assert caps[53] == pytest.approx(0.25 + 2.0 / (RAMP_FACTOR * 100.0))  # past that the ramp taper from s = 51 binds
    assert caps[30] == pytest.approx(0.25 + 19.0 / (RAMP_FACTOR * 100.0))
    narrow = toe_skew_limit([toe], sweep, [1.0] * 100, tmax)  # a steep taper: the meeting distance binds
    assert narrow[52] == pytest.approx(2.0 / 5.0)
    assert narrow[59] >= tmax  # beyond tmax |u| no line meets the sample
    assert toe_skew_limit([(50.0, 0.2, 5.0, 0.0, 100.0)], sweep, wide, tmax) == [math.inf] * len(sweep)  # 1.6 * 0.2 <= 0.5
    # A toe that ends at s = 50 (a deck end) binds only its own rows; the deck's rows taper over their own reach.
    ended = toe_skew_limit([(49.5, 2.0, 5.0, 0.0, 50.0)], sweep, [100.0] * 50 + [10.0] * 50, tmax)
    assert ended[50] == pytest.approx(0.25)
    assert ended[55] == pytest.approx(0.25 + 5.0 / (RAMP_FACTOR * 10.0))


def test_ac_p4_skewed_rows_take_the_uv_segment_that_keeps_their_turning_smallest():
    """A skewed deck row reaches stations past its UV0 segment's end; the face takes the neighbouring segment
    whose reference heading keeps its turning smallest, so deck, cap and pavement tops stay well under AC-P4 (D-148).

    Before, faces kept the segment of their mean station and reached about 3 % across here, and over 5 % on the
    55-degree ramp deck on a 40 m radius at New Orleans.
    """
    import dataclasses

    from roader.validate.assets import uv_stretch

    m = curved_model()
    st = m.structures[0]
    m.structures = [dataclasses.replace(st, supports=[dataclasses.replace(sp, skew_rad=0.5) for sp in st.supports])]
    worst: dict[str, float] = {}
    for x in generate(m, chord_tol_m=TOL):
        if x.uv0 is not None and len(x.faces):
            worst[x.part] = max(worst.get(x.part, 0.0), float(uv_stretch(x)[1].max()))
    for part in ("deck", "pier_cap", "pavement", "parapet"):
        assert worst[part] < 0.02, (part, worst[part])


def test_ac_p4_face_distortion_measures_the_uv_mapping_as_the_gate_does():
    """D-148: the per-face fallback to an isometric plane mapping is judged by the AC-P4 measure."""
    from roader.solids.attributes import _distortion

    tri = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    assert _distortion(tri, np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])) < 1e-12
    assert abs(_distortion(tri, np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0 / 1.06]])) - 0.06) < 1e-9
