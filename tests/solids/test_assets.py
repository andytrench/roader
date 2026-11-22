"""Asset outputs from the model: real-time profile, LOD chain, normals, UV sets (FR-SV-7, FR-AS-1..5)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import shapely

from roader.model.mesh import corner_values
from roader.model.synthetic import overpass_model
from roader.solids import LOD_CHORD_TOL, generate, generate_lods
from roader.solids.attributes import PSI_MAX
from roader.solids.bodies import REALTIME_EMBED_M
from roader.validate.assets import check_budgets, check_uv_stretch
from roader.validate.gates import check_solids, check_voids
from roader.validate.lod import check_lods, deck_top_error
from roader.validate.val3dity import find_val3dity

from .test_curved import _rolling, curved_model


@pytest.fixture(scope="module")
def overpass():
    model = overpass_model()
    return model, generate(model), generate(model, profile="realtime")


@pytest.fixture(scope="module")
def curved():
    model = curved_model(_rolling)
    return model, generate(model), generate(model, profile="realtime")


def _by_id(meshes):
    return {m.id: m for m in meshes}


def _native(result):
    return [f"{v.req} {v.element_id}: {v.message}" for v in result.violations if not v.message.startswith("val3dity")]


# ---------------------------------------------------------------- FR-SV-7 real-time profile


@pytest.mark.parametrize("scene", ["overpass", "curved"])
def test_realtime_profile_passes_fr_sv_7(scene, request):
    _model, _analytic, realtime = request.getfixturevalue(scene)
    result = check_solids(realtime, profile="realtime", val3dity_exe=find_val3dity())
    assert _native(result) == []
    if not result.skipped_reason:
        assert result.passed, [v.message for v in result.violations][:5]


def test_analytic_contacts_fail_the_realtime_rule_fr_sv_7(overpass):
    """The shared faces of the analytic profile are exactly what z-fights in real time."""
    _model, analytic, _realtime = overpass
    result = check_solids(analytic, profile="realtime", val3dity_exe="/nonexistent")
    assert any("coplanar" in m for m in _native(result))


def test_realtime_resting_bodies_embed_d_038(overpass):
    model, analytic, realtime = overpass
    a, r = _by_id(analytic), _by_id(realtime)
    road = model.roads[0]
    pav = f"{road.id}:pavement"
    assert r[pav].vertices[:, 2].min() == pytest.approx(a[pav].vertices[:, 2].min() - REALTIME_EMBED_M, abs=1e-9)
    st = model.structures[0]
    for sp in st.supports:
        mid = f"{sp.id}:abutment" if sp.kind == "abutment" else f"{sp.id}:pier_cap"
        assert r[mid].vertices[:, 2].max() == pytest.approx(a[mid].vertices[:, 2].max() + REALTIME_EMBED_M, abs=1e-9)
    for b in model.road(st.carried_road).cross_section.barriers:
        mid = f"{b.id}:parapet"
        assert r[mid].vertices[:, 2].min() <= a[mid].vertices[:, 2].min() - REALTIME_EMBED_M + 1e-9


def test_realtime_keeps_deck_and_voids_ac_h3(overpass):
    model, _analytic, realtime = overpass
    assert check_voids(model, realtime).passed
    err, missing, n = deck_top_error(model, model.structures[0], realtime)
    assert n > 0 and missing == 0 and err < 0.005


def test_profiles_share_ids_and_are_deterministic_ac_h5(overpass):
    model, analytic, realtime = overpass
    assert [m.id for m in analytic] == [m.id for m in realtime]
    again = generate(model, profile="realtime")
    for x, y in zip(realtime, again, strict=True):
        assert np.array_equal(x.vertices, y.vertices) and np.array_equal(x.faces, y.faces)
        for name in ("normals", "uv0", "uv1"):
            assert (getattr(x, name) is None and getattr(y, name) is None) or np.array_equal(getattr(x, name), getattr(y, name))


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        generate(overpass_model(), profile="lod")


# ---------------------------------------------------------------- FR-AS-1 LODs


@pytest.mark.parametrize("profile", ["analytic", "realtime"])
def test_lod_chain_fr_as_1(profile):
    model = curved_model(_rolling)
    lods = generate_lods(model, profile=profile)
    assert sorted(lods) == sorted(LOD_CHORD_TOL)
    result = check_lods(model, lods)
    assert result.passed, [v.message for v in result.violations][:10]
    counts = [result.metrics[f"n_triangles_lod{k}"] for k in sorted(lods)]
    assert all(b < a for a, b in itertools.pairwise(counts)), counts
    assert not any(m.part == "proxy" for level in lods.values() for m in level)
    assert [m.id for m in lods[3]] == [m.id for m in lods[0]]


def test_structures_keep_their_finest_tessellation_fr_as_1():
    """A deck on a crest would sag below the model at coarse chords and close the void."""
    model = overpass_model()
    lods = generate_lods(model)
    deck = [next(m for m in lods[k] if m.part == "deck") for k in sorted(lods)]
    assert all(len(d.faces) == len(deck[0].faces) for d in deck)


# ---------------------------------------------------------------- FR-AS-2 budgets


@pytest.mark.parametrize("profile", ["analytic", "realtime"])
def test_budgets_every_level_fr_as_2(profile):
    model = overpass_model()
    for level, meshes in generate_lods(model, profile=profile).items():
        result = check_budgets(model, meshes)
        assert result.passed, (level, [v.message for v in result.violations])


# ---------------------------------------------------------------- FR-AS-3 normals


def _corner_normals(mesh):
    n = corner_values(mesh, mesh.normals, 3)
    assert np.allclose(np.linalg.norm(n, axis=2), 1.0)
    return n


def _face_normals(mesh):
    t = mesh.vertices[mesh.faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return fn / np.linalg.norm(fn, axis=1)[:, None]


def _distinct_by_vertex(mesh, normals, mask):
    out: dict[int, set] = {}
    for f in np.nonzero(mask)[0]:
        for k, v in enumerate(mesh.faces[f]):
            out.setdefault(int(v), set()).add(tuple(np.round(normals[f, k], 9)))
    return out


def test_every_closed_mesh_has_attributes(overpass, curved):
    for meshes in (overpass[1], overpass[2], curved[1], curved[2]):
        for m in meshes:
            assert m.normals is not None and m.uv1 is not None
            if m.closed:
                assert m.normals.shape == (len(m.faces), 3, 3) and m.uv0.shape == (len(m.faces), 3, 2)
            else:
                assert m.uv0 is None  # terrain: textured in world space, not a road surface


def test_pavement_smooth_across_crown_hard_at_slab_edge_fr_as_3(curved):
    _model, analytic, _rt = curved
    pav = _by_id(analytic)["rd_curve:pavement:0"]
    n, fn = _corner_normals(pav), _face_normals(pav)
    top, wall = fn[:, 2] > 0.9, np.abs(fn[:, 2]) < 1e-6
    top_sets = _distinct_by_vertex(pav, n, top)
    assert all(len(s) == 1 for s in top_sets.values()), "pavement top must be one smooth surface (crown included)"
    # Smoothing really averaged across the crown and the banking changes.
    assert np.abs(n[top] - fn[top][:, None, :]).max() > 1e-4
    wall_sets = _distinct_by_vertex(pav, n, wall)
    edge = set(top_sets) & set(wall_sets)
    assert edge
    for v in edge:
        (nt,) = top_sets[v]
        assert nt[2] > 0.9 and all(abs(nw[2]) < 1e-6 for nw in wall_sets[v])


def test_curved_side_wall_is_smooth_end_walls_are_hard_fr_as_3(curved):
    _model, analytic, _rt = curved
    pav = _by_id(analytic)["rd_curve:pavement:0"]
    n, fn = _corner_normals(pav), _face_normals(pav)
    wall = np.abs(fn[:, 2]) < 1e-6
    smoothed = np.abs(n[wall] - fn[wall][:, None, :]).max(axis=2) > 1e-6
    assert smoothed.any(), "walls along the curve must share normals across plan edges"
    # The slab's end-wall corners keep two wall normals (along wall and end wall).
    sets = _distinct_by_vertex(pav, n, wall)
    assert any(len(s) == 2 for s in sets.values())


def test_barrier_arris_is_a_crease_fr_as_3(curved):
    _model, analytic, _rt = curved
    bar = _by_id(analytic)["guard_r:barrier"]
    n, fn = _corner_normals(bar), _face_normals(bar)
    up = fn[:, 2] > 1e-6
    sets = _distinct_by_vertex(bar, n, up)
    assert any(len(s) >= 2 for s in sets.values()), "top flat and sloped faces of a jersey barrier meet at a hard arris"


# ---------------------------------------------------------------- FR-AS-4 uv0


def test_uv0_is_station_and_offset_on_a_straight_road_fr_as_4(overpass):
    model, analytic, _rt = overpass
    eb = model.roads[0]
    pav = _by_id(analytic)[f"{eb.id}:pavement"]
    uv, fn = corner_values(pav, pav.uv0, 2), _face_normals(pav)
    top = fn[:, 2] > 0.9
    p = pav.vertices[pav.faces[top]]
    s_true, u_true = p[..., 0] + 300.0, p[..., 1] + 9.0
    assert np.allclose(uv[top][..., 0], s_true, atol=1e-6)
    assert np.allclose(uv[top][..., 1], u_true, atol=1.2e-3)  # arc length across a 2% crown


def test_uv0_continuous_from_pavement_onto_deck_fr_as_4(overpass):
    model, analytic, _rt = overpass
    st = model.structures[0]
    meshes = _by_id(analytic)
    seam = {}
    for mid in (f"{st.carried_road}:pavement:0", f"{st.id}:deck"):
        m = meshes[mid]
        uv, fn = corner_values(m, m.uv0, 2), _face_normals(m)
        for f in np.nonzero(fn[:, 2] > 0.9)[0]:
            for k, v in enumerate(m.faces[f]):
                seam.setdefault(tuple(np.round(m.vertices[v], 6)), {})[mid] = uv[f, k]
    shared = [d for d in seam.values() if len(d) == 2]
    assert len(shared) >= 3  # pavement edges and crown at the deck end
    for d in shared:
        a, b = d.values()
        assert np.allclose(a, b, atol=1e-4)


def test_uv0_does_not_stretch_on_curves_ac_p4(curved):
    """Centre-line station would stretch by u/R (R = 120 m, toes ~15 m out: >10%)."""
    _model, analytic, realtime = curved
    for meshes in (analytic, realtime):
        result = check_uv_stretch(meshes)
        assert result.passed, [v.message for v in result.violations][:5]
        assert result.metrics["max_stretch_along"] < 0.01
        assert result.metrics["max_stretch_across"] < 1.0 - np.cos(PSI_MAX) + 0.01


# ---------------------------------------------------------------- FR-AS-5 uv1


@pytest.mark.parametrize("scene", ["overpass", "curved"])
def test_uv1_atlas_in_unit_square_without_overlap_fr_as_5(scene, request):
    _model, analytic, realtime = request.getfixturevalue(scene)
    for m in analytic + realtime:
        uv = corner_values(m, m.uv1, 2)
        assert uv.min() >= 0.0 and uv.max() <= 1.0, m.id
        if m.closed and len(m.faces) <= 4000:
            polys = [shapely.Polygon(t) for t in uv]
            total = sum(p.area for p in polys)
            assert total > 0.0 and shapely.union_all(polys).area == pytest.approx(total, rel=1e-6), m.id
