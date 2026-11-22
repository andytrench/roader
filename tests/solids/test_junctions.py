"""L6 junction areas on the synthetic diverge (FR-SG-3, AC-H4, FR-SV-1..7, AC-H1, AC-H5, AC-H6, D-046)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from roader.complete.junctions import build_junctions
from roader.model import dumps, loads
from roader.model.geometry import Alignment, Profile, VerticalElement
from roader.model.synthetic import DIVERGE_GRADE, TRUTH, diverge_model, overpass_model
from roader.solids import generate
from roader.validate.gates import check_continuity, check_provenance, check_solids
from roader.validate.profile import check_profile
from roader.validate.seams import junction_violations
from roader.validate.val3dity import find_val3dity

from . import meshcheck as mc


@pytest.fixture(scope="module")
def model():
    return diverge_model()


@pytest.fixture(scope="module")
def meshes(model):
    return generate(model)


def _plan(mesh) -> Polygon:
    v = mesh.vertices
    return unary_union([Polygon(v[f, :2]) for f in mesh.faces if abs(Polygon(v[f, :2]).area) > 1e-9])


# ---------------------------------------------------------------- construction (FR-SG-3)


def test_fr_sg_3_one_diverge_with_three_seams(model):
    assert model.flags == []
    assert len(model.junctions) == 1
    jn = model.junctions[0]
    fwy, ramp = model.roads
    assert jn.kind == "diverge"
    assert jn.road_ids == sorted([fwy.id, ramp.id])
    by_road = sorted((s.road_id == ramp.id, s.side) for s in jn.seams)
    assert by_road == [(False, -1), (False, 1), (True, -1)]
    lo, hi = sorted(s.s for s in jn.seams if s.road_id == fwy.id)
    # The ramp starts at s = 200 and clears the freeway near x = 72 (s = 272).
    assert 190.0 < lo < 200.0
    assert 260.0 < hi < 290.0
    (ramp_seam,) = [s for s in jn.seams if s.road_id == ramp.id]
    assert 60.0 < ramp_seam.s < 90.0
    for s in jn.seams:
        road = model.road(s.road_id)
        assert s.u0 == pytest.approx(-road.cross_section.width_right(s.s))
        assert s.u1 == pytest.approx(road.cross_section.width_left(s.s))
    assert jn.sections and all(sec.z is not None for sec in jn.sections)


def test_fr_sg_3_outline_is_a_planar_partition(model):
    """The outline is one valid counter-clockwise polygon covering both pavements in their intervals."""
    jn = model.junctions[0]
    poly = Polygon(jn.outline, jn.holes)
    assert poly.is_valid and poly.exterior.is_ccw
    ramp = model.roads[1]
    (seam,) = [s for s in jn.seams if s.road_id == ramp.id]
    for s in np.linspace(0.0, seam.s - 0.01, 40):
        for u in (-2.4, 0.0, 2.4):
            x, y = ramp.alignment.offset_point(float(s), u)
            assert jn.contains(x, y), (s, u)


def test_fr_sg_3_no_junction_where_roads_only_cross():
    """The M1 overpass: roads cross at different levels, so nothing is built (M1 unchanged)."""
    areas, flags = build_junctions(overpass_model())
    assert areas == [] and flags == []


# ---------------------------------------------------------------- lifted surface (AC-H4, D-046)


def test_ac_h4_junction_seams_match_elevation_and_slope(model):
    result = check_continuity(model)
    assert result.passed, result.violations
    assert result.metrics["n_junctions"] == 1.0
    assert result.metrics["n_seam_samples"] >= 25


def test_ac_h4_surface_follows_the_overlapping_roads(model):
    jn = model.junctions[0]
    fwy = model.roads[0]
    worst = max(
        abs(jn.surface_z(x, y) - fwy.surface_z(x + 200.0, y))
        for x in np.linspace(0.0, 60.0, 61)
        for y in (-5.0, -2.0, 1.0, 4.0)
    )
    assert worst < 0.01


def test_ac_h4_road_edited_after_lifting_fails_its_seam(model):
    """The surface is stored in the model: a road changed after lifting no longer matches its seam."""
    edited = loads(dumps(model))
    ramp = edited.roads[1]
    ramp.profile = Profile([VerticalElement(e.s0, e.length, e.z0 + 0.03, e.g0, e.g1, e.prov) for e in ramp.profile.elements])
    violations, _n = junction_violations(edited.junctions[0], {r.id: r for r in edited.roads})
    assert any("elevation" in v.message and ramp.id in v.message for v in violations), violations


def test_surface_z_is_nan_outside_the_area(model):
    jn = model.junctions[0]
    assert math.isnan(jn.surface_z(-50.0, 0.0))
    assert math.isnan(jn.surface_z(30.0, 30.0))
    assert math.isfinite(jn.surface_z(30.0, 0.0))


def test_ac_h5_junctions_round_trip_canonically(model):
    data = dumps(model)
    back = loads(data)
    assert dumps(back) == data
    a, b = model.junctions[0], back.junctions[0]
    for x, y in ((10.0, -3.0), (50.0, -6.0), (70.0, 2.0)):
        assert a.surface_z(x, y) == pytest.approx(b.surface_z(x, y), abs=1e-6)


def test_old_model_without_patches_loads():
    """Models written before D-046: seams without patch fields load; the surface says it is missing."""
    plain = json.loads(dumps(diverge_model()))
    for jn in plain["junctions"]:
        jn.pop("sections")
        jn["seams"] = [{k: s[k] for k in ("road_id", "s", "u0", "u1")} for s in jn["seams"]]
    back = loads(json.dumps(plain))
    jn = back.junctions[0]
    assert jn.seams[0].z is None and jn.sections == []
    with pytest.raises(ValueError, match="no lifted surface"):
        jn.surface_z(30.0, 0.0)


# ---------------------------------------------------------------- solids (FR-SG-3, FR-SV)


def test_junction_bodies_and_clipped_sweeps_fr_dm_3(model, meshes):
    jn = model.junctions[0]
    fwy, ramp = model.roads
    ids = {m.id for m in meshes}
    assert len(ids) == len(meshes)
    assert f"{jn.id}:pavement" in ids and f"{jn.id}:embankment" in ids
    for m in meshes:
        if m.element_id == jn.id:
            assert m.layer == "L6" and m.closed
    assert sorted(m.id for m in meshes if m.element_id == fwy.id) == [f"{fwy.id}:pavement:0", f"{fwy.id}:pavement:1"]
    assert [m.id for m in meshes if m.element_id == ramp.id] == [f"{ramp.id}:pavement"]


def test_fr_sg_3_sweeps_do_not_overlap_the_area(model, meshes):
    area = Polygon(model.junctions[0].outline)
    for m in meshes:
        if m.closed and m.layer in ("L3", "L4"):
            assert _plan(m).intersection(area).area < 1e-6, m.id


@pytest.mark.parametrize("part", ["pavement", "embankment"])
def test_fr_sv_1_to_6_junction_solids(model, meshes, part):
    (mesh,) = [m for m in meshes if m.element_id == model.junctions[0].id and m.part == part]
    assert mc.solid_problems(mesh) == [], mesh.id


def test_fr_sv_7_conforming_with_road_sweeps(model, meshes):
    """Analytic profile: no overlap and conforming contact between the area and the sweeps."""
    jn = model.junctions[0]
    near = [m for m in meshes if m.closed and (m.element_id == jn.id or m.layer in ("L3", "L4"))]
    assert check_profile(near) == []


def test_seam_vertices_are_shared_with_the_road(model, meshes):
    jn = model.junctions[0]
    slab = next(m for m in meshes if m.id == f"{jn.id}:pavement")
    keys = {tuple(np.round(v, 9)) for v in slab.vertices}
    for seam in jn.seams:
        road = model.road(seam.road_id)
        pave = [m for m in meshes if m.element_id == road.id and m.part == "pavement"]
        for u in (seam.u0, 0.0, seam.u1):
            x, y = road.alignment.offset_point(seam.s, u)
            top = (round(x, 9), round(y, 9), round(road.surface_z(seam.s, u), 9))
            assert top in keys, (seam.road_id, seam.s, u)
            assert any(top in {tuple(np.round(v, 9)) for v in m.vertices} for m in pave)


def test_ac_h1_ac_h6_with_val3dity(model, meshes):
    if find_val3dity() is None:
        pytest.skip("val3dity not installed")
    result = check_solids(meshes)
    assert result.passed, [f"{v.element_id}: {v.message}" for v in result.violations[:10]]
    assert check_provenance(model, meshes).passed


def test_ac_h5_deterministic(model):
    def digest(ms):
        h = hashlib.sha256()
        for m in ms:
            h.update(m.id.encode())
            h.update(np.ascontiguousarray(m.vertices).tobytes())
            h.update(np.ascontiguousarray(m.faces).tobytes())
        return h.hexdigest()

    assert digest(generate(model)) == digest(generate(loads(dumps(diverge_model()))))


def _split_freeway(model):
    """The vector shape of OSM: the freeway way ends at the ramp node and a new way starts there."""
    base = diverge_model(with_junctions=False)
    fwy, ramp = base.roads
    halves = []
    for name, x0 in (("fwy_a", -200.0), ("fwy_b", 0.0)):
        road = dataclasses.replace(fwy, id=f"rd_{name}", name=name)
        road.alignment = Alignment([dataclasses.replace(fwy.alignment.elements[0], s0=0.0, length=200.0, x0=x0)])
        z0 = fwy.profile.z(x0 + 200.0)
        road.profile = Profile([VerticalElement(0.0, 200.0, z0, DIVERGE_GRADE, DIVERGE_GRADE, TRUTH)])
        halves.append(road)
    base.roads = [*halves, ramp]
    base.side_slopes = [
        dataclasses.replace(ss, road_id=halves[0].id, s1=200.0) if ss.road_id == fwy.id and ss.side == "left" else ss
        for ss in base.side_slopes
        if ss.road_id != fwy.id or ss.side == "left"
    ]
    base.side_slopes.append(dataclasses.replace(base.side_slopes[0], id="ss_fwy_b_left", road_id=halves[1].id, s0=0.0, s1=200.0))
    base = loads(dumps(base))
    base.junctions, base.flags = build_junctions(base)
    return loads(dumps(base))


def test_fr_sg_3_freeway_split_at_the_ramp_node(model):
    split = _split_freeway(model)
    assert split.flags == []
    (jn,) = split.junctions
    assert jn.kind == "diverge" and len(jn.road_ids) == 3
    sides = sorted((s.road_id, s.side) for s in jn.seams)
    assert sides == sorted([("rd_fwy_a", 1), ("rd_fwy_b", -1), (split.roads[2].id, -1)])
    assert check_continuity(split).passed
    meshes = generate(split)
    closed = [m for m in meshes if m.closed]
    assert all(mc.solid_problems(m) == [] for m in closed)
    assert check_profile(closed) == []


def test_no_terrain_gives_a_slab_only(model):
    bare = dataclasses.replace(model, terrain=None, side_slopes=[])
    jn = model.junctions[0]
    meshes = generate(bare)
    assert [m.part for m in meshes if m.element_id == jn.id] == ["pavement"]
    assert all(mc.solid_problems(m) == [] for m in meshes)


def test_fr_sv_7_realtime_junction_bodies_inset_and_embedded(model):
    """Real-time profile (D-047): the area's slab and fill are inset and embedded like the sweeps, so no coplanar overlap."""
    from roader.validate.realtime import check_realtime

    rt = generate(model, profile="realtime")
    jn = model.junctions[0]
    slab = next(m for m in rt if m.id == f"{jn.id}:pavement")
    analytic = next(m for m in generate(model) if m.id == slab.id)
    assert slab.vertices[:, 2].min() < analytic.vertices[:, 2].min() - 0.2
    assert _plan(slab).area < _plan(analytic).area
    near = [m for m in rt if m.closed and (m.element_id == jn.id or m.layer in ("L3", "L4"))]
    assert check_realtime(near) == []
