"""Solid generation on the synthetic overpass (FR-SG-1..6, FR-SV-1..5, FR-SV-7, AC-H3, AC-H5, AC-H6)."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from roader.model import Mesh
from roader.model.synthetic import overpass_model
from roader.solids import generate
from roader.solids.terrain import GRID_SNAP_M

from . import meshcheck as mc


@pytest.fixture(scope="module")
def model():
    return overpass_model()


@pytest.fixture(scope="module")
def meshes(model) -> list[Mesh]:
    return generate(model)


def _closed(meshes: list[Mesh]) -> list[Mesh]:
    return [m for m in meshes if m.closed]


def _by_id(meshes: list[Mesh]) -> dict[str, Mesh]:
    return {m.id: m for m in meshes}


def test_body_list_and_provenance_ac_h6_fr_dm_3(model, meshes):
    st = model.structures[0]
    over = model.road(st.carried_road)
    ids = _by_id(meshes)
    assert len(ids) == len(meshes)
    for road in model.roads:
        n = 2 if road is over else 1
        found = [m for m in meshes if m.element_id == road.id and m.part == "pavement"]
        assert len(found) == n
    assert f"{st.id}:deck" in ids
    for b in over.cross_section.barriers:
        assert f"{b.id}:parapet" in ids
    for sp in st.supports:
        if sp.kind == "abutment":
            assert f"{sp.id}:abutment" in ids
        else:
            assert f"{sp.id}:pier_cap" in ids
            assert sum(1 for m in meshes if m.element_id == sp.id and m.part == "pier_column") == sp.n_columns
    for ss in model.side_slopes:
        assert f"{ss.id}:embankment" in ids
    layers = {"pavement": "L3", "embankment": "L4", "deck": "L5", "parapet": "L5", "abutment": "L5",
              "pier_cap": "L5", "pier_column": "L5", "terrain": "L7"}
    for m in meshes:
        assert m.layer == layers[m.part]
        assert m.material
        assert m.source is not None
        assert m.id.startswith(f"{m.element_id}:{m.part}")
        assert m.closed == (m.part != "terrain")
    assert meshes[-1].part == "terrain"
    assert all(m.element_id == "terrain" for m in meshes if m.layer == "L7")


@pytest.mark.parametrize("index", range(20))
def test_every_solid_is_valid_fr_sv_1_to_6(meshes, index):
    closed = _closed(meshes)
    if index >= len(closed):
        pytest.skip("fewer solids")
    mesh = closed[index]
    assert mc.solid_problems(mesh) == [], mesh.id


def test_solid_count(meshes):
    assert len(_closed(meshes)) == 20


def test_no_crossing_triangles_fr_sv_4_fr_sv_7(meshes):
    assert mc.crossing_pairs(_closed(meshes)) == []


def test_interiors_do_not_overlap_fr_sv_7(meshes):
    closed = _closed(meshes)
    for mesh in closed:
        pts = mc.interior_samples(mesh)
        assert len(pts) > 5, mesh.id
        for other in closed:
            if other is mesh:
                continue
            lo, hi = mc.bbox(other)
            near = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
            if len(near):
                assert not mc.inside(near, other).any(), (mesh.id, other.id)


def test_contacts_are_conforming_fr_sv_7(meshes):
    for a in meshes:
        for b in meshes:
            if a is not b and mc.boxes_overlap(a, b):
                assert mc.t_junctions(a, b) == 0, (a.id, b.id)


def test_pavement_deck_and_parapets_share_faces_fr_sg_2(model, meshes):
    st = model.structures[0]
    over = model.road(st.carried_road)
    ids = _by_id(meshes)
    deck = ids[f"{st.id}:deck"]
    for k in (0, 1):
        pav = ids[f"{over.id}:pavement:{k}"]
        assert mc.shared_faces(pav, deck) > 0
        assert mc.t_junctions(pav, deck) == 0 and mc.t_junctions(deck, pav) == 0
    for b in over.cross_section.barriers:
        par = ids[f"{b.id}:parapet"]
        normals = np.cross(par.vertices[par.faces[:, 1]] - par.vertices[par.faces[:, 0]],
                           par.vertices[par.faces[:, 2]] - par.vertices[par.faces[:, 0]])
        bottom = int(np.sum(normals[:, 2] < -1e-9))
        assert mc.shared_faces(par, deck) == bottom > 0
    for sp in st.supports:
        top = ids[f"{sp.id}:abutment"] if sp.kind == "abutment" else ids[f"{sp.id}:pier_cap"]
        assert mc.shared_faces(top, deck) > 0


def test_embankment_top_is_pavement_underside_fr_sg_2(model, meshes):
    ids = _by_id(meshes)
    for road in model.roads[:2]:
        pav = ids[f"{road.id}:pavement"]
        n = pav.faces.shape[0]
        normals = np.cross(pav.vertices[pav.faces[:, 1]] - pav.vertices[pav.faces[:, 0]],
                           pav.vertices[pav.faces[:, 2]] - pav.vertices[pav.faces[:, 0]])
        underside = int(np.sum(normals[:, 2] < -1e-9))
        shared = sum(mc.shared_faces(pav, ids[f"{ss.id}:embankment"]) for ss in model.side_slopes if ss.road_id == road.id)
        assert shared == underside > 0 and n > underside


def test_structure_union_is_one_solid_fr_sv_7(model, meshes):
    st = model.structures[0]
    over = model.road(st.carried_road)
    parts = {st.id} | {sp.id for sp in st.supports} | {b.id for b in over.cross_section.barriers}
    members = [m for m in meshes if m.element_id in parts]
    assert len(members) == 8
    verts, faces = mc.merge_union(members)
    union = Mesh("u", "u", "L5", "deck", "x", "rule", verts, faces)
    assert mc.closed_and_oriented(union) == []
    assert mc.components(union) == 1
    assert mc.vertex_fans(union) == 0
    total = sum(float(np.sum(m.vertices[m.faces][:, 0] * np.cross(m.vertices[m.faces][:, 1], m.vertices[m.faces][:, 2]))) / 6.0
                for m in members)
    from roader.model import signed_volume
    assert math.isclose(signed_volume(union), total, rel_tol=1e-9)


def _surface_hits(mesh: Mesh, xy: np.ndarray) -> list[np.ndarray]:
    """Elevations where vertical lines at xy cross the mesh."""
    v = mesh.vertices
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    out = []
    for x, y in xy:
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-14
        l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / np.where(ok, d, 1.0)
        l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / np.where(ok, d, 1.0)
        l3 = 1.0 - l1 - l2
        hit = ok & (l1 >= -1e-12) & (l2 >= -1e-12) & (l3 >= -1e-12)
        out.append(l1[hit] * a[hit, 2] + l2[hit] * b[hit, 2] + l3[hit] * c[hit, 2])
    return out


def test_void_under_deck_is_open_ac_h3_g3(model, meshes):
    st = model.structures[0]
    deck = _by_id(meshes)[f"{st.id}:deck"]
    checked = 0
    for road_id in st.crossed_roads:
        road = model.road(road_id)
        xs = road.cross_section
        pts, zs = [], []
        for s in np.arange(0.0, road.length, 0.5):
            for u in np.linspace(-xs.width_right(s) + 0.05, xs.width_left(s) - 0.05, 9):
                x, y = road.alignment.offset_point(float(s), float(u))
                pts.append((x, y))
                zs.append(road.surface_z(float(s), float(u)))
        pts = np.array(pts)
        lo, hi = mc.bbox(deck)
        sel = np.all((pts >= lo[:2]) & (pts <= hi[:2]), axis=1)
        pts, zs = pts[sel], np.array(zs)[sel]
        soffits = _surface_hits(deck, pts)
        under = [(p, z, h.min()) for p, z, h in zip(pts, zs, soffits, strict=True) if len(h)]
        assert len(under) > 50
        for mesh in _closed(meshes):
            hits = _surface_hits(mesh, np.array([p for p, _, _ in under]))
            for (_p, z, soffit), h in zip(under, hits, strict=True):
                assert not np.any((h > z + 0.05) & (h < soffit - 1e-6)), mesh.id
        checked += len(under)
    assert checked > 100


def check_terrain(model, meshes: list[Mesh], tol: float) -> list[Mesh]:
    """FR-SG-6 on any scene: one open, connected L7 mesh per piece; every open edge is on
    the patch border or is a solid edge; no terrain inside a ground-contact footprint;
    terrain plus footprints cover the patch with no gap; surface within tol of the grid."""
    terrain = [m for m in meshes if m.part == "terrain"]
    assert terrain and meshes[-len(terrain):] == terrain
    ids = [m.id for m in terrain]
    assert ids == (["terrain:terrain"] if len(terrain) == 1 else [f"terrain:terrain:{k}" for k in range(len(terrain))])
    keys = [(round(float(m.vertices[:, 0].min()), 3), round(float(m.vertices[:, 1].min()), 3)) for m in terrain]
    assert keys == sorted(keys)
    t = model.terrain
    x0, y0 = t.origin_x, t.origin_y
    x1, y1 = x0 + (t.nx - 1) * t.cell_m, y0 + (t.ny - 1) * t.cell_m
    solid_edges = set()
    for m in _closed(meshes):
        for a, b in mc.edges(m.faces).tolist():
            pa, pb = tuple(m.vertices[a]), tuple(m.vertices[b])
            solid_edges.add((pa, pb))
            solid_edges.add((pb, pa))
    on_border = 0
    for piece in terrain:
        assert not piece.closed and piece.layer == "L7" and piece.element_id == "terrain"
        assert mc.components(piece) == 1 and mc.vertex_fans(piece) == 0
        assert mc.degeneracy(piece) == []
        keys, counts = np.unique(np.sort(mc.edges(piece.faces), axis=1), axis=0, return_counts=True)
        assert counts.max() <= 2
        _, dcounts = np.unique(mc.edges(piece.faces), axis=0, return_counts=True)
        assert dcounts.max() == 1  # consistent winding
        for a, b in keys[counts == 1].tolist():
            pa, pb = piece.vertices[a], piece.vertices[b]
            border = any(
                abs(pa[k] - v) <= GRID_SNAP_M and abs(pb[k] - v) <= GRID_SNAP_M for k, v in ((0, x0), (0, x1), (1, y0), (1, y1))
            )
            if border:
                on_border += 1
                continue
            assert (tuple(pa), tuple(pb)) in solid_edges
        v = piece.vertices
        n = np.cross(v[piece.faces[:, 1]] - v[piece.faces[:, 0]], v[piece.faces[:, 2]] - v[piece.faces[:, 0]])
        assert np.all(n[:, 2] > 0.0)
    assert on_border > 0
    # No terrain inside a ground-contact footprint; the ground under a span stays.
    tv = {tuple(r) for m in terrain for r in m.vertices.tolist()}
    footprints = []
    for m in _closed(meshes):
        if any(tuple(r) in tv for r in m.vertices.tolist()):
            footprints.append(unary_union([Polygon(tri[:, :2]) for tri in m.vertices[m.faces] if Polygon(tri[:, :2]).area > 1e-9]))
    fp = unary_union(footprints)
    inner = fp.buffer(-1e-6)
    cent = np.vstack([m.vertices[m.faces].mean(axis=1) for m in terrain])
    lo, hi = np.array(inner.bounds[:2]), np.array(inner.bounds[2:])
    sel = np.all((cent[:, :2] >= lo) & (cent[:, :2] <= hi), axis=1)
    assert not any(inner.contains(Point(p)) for p in cent[sel, :2])
    area = 0.0
    for m in terrain:
        u = m.vertices[m.faces[:, 1], :2] - m.vertices[m.faces[:, 0], :2]
        w = m.vertices[m.faces[:, 2], :2] - m.vertices[m.faces[:, 0], :2]
        area += 0.5 * float(np.sum(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]))
    domain = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    assert math.isclose(area + fp.intersection(domain).area, domain.area, rel_tol=1e-8)
    # Merged far-field blocks stay within tol of the bilinear grid surface.
    zgrid = np.asarray(t.z, dtype=float).reshape(t.ny, t.nx)
    twist = 0.25 * np.abs(zgrid[:-1, :-1] - zgrid[:-1, 1:] - zgrid[1:, :-1] + zgrid[1:, 1:]).max()
    for m in terrain:
        gx, gy = (m.vertices[:, 0] - x0) / t.cell_m, (m.vertices[:, 1] - y0) / t.cell_m
        on_grid = (np.abs(gx - np.round(gx)) < 1e-9) & (np.abs(gy - np.round(gy)) < 1e-9)
        c = m.vertices[m.faces[on_grid[m.faces].all(axis=1)]].mean(axis=1)  # band triangles at footprints excluded
        if len(c) == 0:
            continue
        fx, fy = (c[:, 0] - x0) / t.cell_m, (c[:, 1] - y0) / t.cell_m
        i = np.clip(np.floor(fx).astype(int), 0, t.nx - 2)
        j = np.clip(np.floor(fy).astype(int), 0, t.ny - 2)
        tx, ty = fx - i, fy - j
        zb = (zgrid[j, i] * (1 - tx) + zgrid[j, i + 1] * tx) * (1 - ty) + (zgrid[j + 1, i] * (1 - tx) + zgrid[j + 1, i + 1] * tx) * ty
        assert np.abs(c[:, 2] - zb).max() <= tol + twist  # twist: error of splitting one grid cell
    return terrain


def test_terrain_conforms_fr_sg_6(model, meshes):
    terrain = check_terrain(model, meshes, 0.01)
    assert len(terrain) == 3  # the freeway embankments cross the whole patch
    assert sum(len(m.faces) for m in terrain) < 20_000
    over = model.road(model.structures[0].carried_road)
    under_span = np.array([over.alignment.offset_point(250.9, 0.0), over.alignment.offset_point(230.0, 0.0)])
    hits = [sum(len(h) for h in hs) for hs in zip(*[_surface_hits(m, under_span) for m in terrain], strict=True)]
    assert hits == [1, 1]


def test_determinism_ac_h5(model, meshes):
    again = generate(overpass_model())
    assert [m.id for m in again] == [m.id for m in meshes]
    for a, b in zip(meshes, again, strict=True):
        assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def _digest_in_subprocess(seed: str) -> str:
    code = (
        "import hashlib\n"
        "from roader.model.synthetic import overpass_model\n"
        "from roader.solids import generate\n"
        "h = hashlib.sha256()\n"
        "for m in generate(overpass_model()):\n"
        "    h.update(m.id.encode()); h.update(m.vertices.tobytes()); h.update(m.faces.tobytes())\n"
        "print(h.hexdigest())\n"
    )
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=os.path.abspath(src))
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip()


def test_determinism_across_processes_ac_h5(meshes):
    h = hashlib.sha256()
    for m in meshes:
        h.update(m.id.encode())
        h.update(m.vertices.tobytes())
        h.update(m.faces.tobytes())
    assert _digest_in_subprocess("1") == _digest_in_subprocess("2") == h.hexdigest()


def test_triangle_budget_report_fr_as_2(model, meshes, capsys):
    st = model.structures[0]
    over = model.road(st.carried_road)
    parts = {st.id} | {sp.id for sp in st.supports} | {b.id for b in over.cross_section.barriers}
    overpass = sum(len(m.faces) for m in meshes if m.element_id in parts)
    assert overpass < 60_000
    with capsys.disabled():
        print()
        for m in meshes:
            print(f"  {m.id:42s} {m.part:12s} {len(m.faces):7d}")
        print(f"  overpass (structure parts): {overpass} triangles")


@pytest.mark.parametrize("end", ["start", "end", "at_end"])
def test_skewed_deck_at_a_road_end_stays_on_the_road_fr_sg_2_fr_st_4(end):
    """A skewed deck end at (or its skew ramp past) the road's end: station lines square off inside the road (D-118).

    ``at_end``: the deck ends on the road end, its station a rounding step past the summed element lengths.
    """
    m = overpass_model()
    over, st = m.roads[2], m.structures[0]
    length = over.length
    if end == "start":
        st.s_start = st.supports[0].station = 0.5
        for b in over.cross_section.barriers:
            b.s0 = 0.5
        m.side_slopes = [s for s in m.side_slopes if not (s.road_id == over.id and s.s0 == 0.0)]
    else:
        s_end = length - 0.5 if end == "end" else length + 1e-7
        st.s_end = st.supports[-1].station = s_end
        for b in over.cross_section.barriers:
            b.s1 = s_end
        m.side_slopes = [s for s in m.side_slopes if not (s.road_id == over.id and s.s1 == length)]
    over.support = []
    meshes = generate(m)
    for mesh in _closed(meshes):
        assert not mc.solid_problems(mesh), (mesh.id, mc.solid_problems(mesh))
    assert any(x.part == "deck" for x in meshes)


def test_deck_edge_never_crosses_its_own_pavement_lines_fr_sg_2():
    """A twin joint that leaves a negative per-side overhang past the pavement (FR-CR-3) stops at the edge limit (D-141).

    -6 m beside 5.45 m of pavement and parapet would put the right deck edge left of the centre line.
    """
    from roader.solids.bodies import DECK_EDGE_MIN_M

    m = overpass_model()
    st = m.structures[0]
    st.deck.overhang_right_m = -6.0
    notes = []
    meshes = generate(m, failures=notes)
    deck = next(x for x in meshes if x.part == "deck")
    assert not mc.solid_problems(deck)
    assert [(f.code, f.element_id, f.req) for f in notes] == [("FR-SG-2:deck_edge_limited", st.id, "FR-CR-3")]
    over = m.roads[2]
    _s, u = over.alignment.project_many(deck.vertices[:, :2])
    assert abs(u.min() + DECK_EDGE_MIN_M) < 1e-3
    assert generate(m)  # strict mode renders the limited deck too, without a flag sink
