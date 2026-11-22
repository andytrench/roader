"""Native solid validity checks, FR-SV-1 to FR-SV-6 (spec section 7).

Each check returns Violations tagged with its requirement ID and the mesh id.
Meshes with ``closed=False`` (the L7 terrain surface) skip FR-SV-1, FR-SV-3 and
FR-SV-6 but still get FR-SV-2, FR-SV-4 and FR-SV-5. Nothing is repaired.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from roader.model.mesh import Mesh, signed_volume
from roader.validate._geom import (
    PLANE_EPS,
    candidate_pairs,
    distance_point_segment,
    edge_fold,
    tri_tri_intersect_batch,
    unit_normals,
    vertex_overlap,
)
from roader.validate.report import Violation

AREA_TOL = 1e-8  # m^2, FR-SV-5 zero-area face
DUP_TOL = 1e-6  # m, FR-SV-5 duplicate vertex
EDGE_TOL = 1e-3  # m, FR-SV-5 edge under 1 mm
TJ_TOL = 1e-6  # m, FR-SV-5 vertex on another edge's interior


def check_solid(
    mesh: Mesh,
    *,
    area_tol: float = AREA_TOL,
    dup_tol: float = DUP_TOL,
    edge_tol: float = EDGE_TOL,
    tj_tol: float = TJ_TOL,
) -> list[Violation]:
    """All native FR-SV-1..6 checks for one mesh."""
    v, f = mesh.vertices, mesh.faces
    out: list[Violation] = []

    def add(req: str, message: str) -> None:
        out.append(Violation(req, mesh.id, message))

    if len(f) == 0:
        add("FR-SV-1", "mesh has no faces")
        return out
    if not np.all(np.isfinite(v)):
        add("FR-SV-5", "non-finite vertex coordinates")
        return out
    if f.min() < 0 or f.max() >= len(v):
        add("FR-SV-5", "face references a vertex index out of range")
        return out

    topo = _Topology(f, len(v))
    if mesh.closed:
        _check_closed(topo, add)
    _check_manifold(topo, add)
    if mesh.closed:
        _check_orientation(mesh, topo, add)
    _check_degeneracy(v, f, topo, add, area_tol, dup_tol, edge_tol, tj_tol)
    _check_self_intersection(v, f, add)
    if mesh.closed:
        _check_components(topo, add)
    return out


class _Topology:
    """Directed edges e = 3 * face + k run from corner k to corner k + 1."""

    def __init__(self, faces: np.ndarray, n_vertices: int) -> None:
        self.faces = faces
        self.n_vertices = n_vertices
        self.directed = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
        a, b = self.directed[:, 0], self.directed[:, 1]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        self.key = lo * n_vertices + hi
        self.ukey, self.inverse, self.count = np.unique(self.key, return_inverse=True, return_counts=True)
        self.inverse = self.inverse.reshape(-1)
        order = np.argsort(self.key, kind="stable")
        same = self.key[order][1:] == self.key[order][:-1]
        # Consecutive entries of one undirected edge: the faces they join.
        self.pair_a = order[:-1][same]
        self.pair_b = order[1:][same]
        self.degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])

    def edge_vertices(self, mask: np.ndarray) -> np.ndarray:
        k = self.ukey[mask]
        return np.column_stack([k // self.n_vertices, k % self.n_vertices])


def _check_closed(topo: _Topology, add) -> None:
    """FR-SV-1: every undirected edge has exactly two faces (>2 is FR-SV-2)."""
    boundary = topo.count == 1
    if boundary.any():
        ex = topo.edge_vertices(boundary)[:3].tolist()
        add("FR-SV-1", f"{int(boundary.sum())} boundary edges with one face, e.g. vertex pairs {ex}")


def _check_manifold(topo: _Topology, add) -> None:
    """FR-SV-2: no edge with more than two faces; faces around each vertex form one fan."""
    over = topo.count > 2
    if over.any():
        ex = topo.edge_vertices(over)[:3].tolist()
        add("FR-SV-2", f"{int(over.sum())} edges shared by more than two faces, e.g. {ex}")

    # Corners (face, vertex) are joined when their faces share an edge at that vertex.
    m = len(topo.faces)
    corner_from = np.arange(3 * m)
    corner_to = 3 * (corner_from // 3) + (corner_from % 3 + 1) % 3
    a, b = topo.directed[:, 0], topo.directed[:, 1]
    corner_lo = np.where(a < b, corner_from, corner_to)
    corner_hi = np.where(a < b, corner_to, corner_from)
    i, j = topo.pair_a, topo.pair_b
    rows = np.concatenate([corner_lo[i], corner_hi[i]])
    cols = np.concatenate([corner_lo[j], corner_hi[j]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * m, 3 * m))
    _n, labels = connected_components(graph, directed=False)
    corner_vertex = topo.faces.reshape(-1)
    valid = ~np.repeat(topo.degenerate, 3)
    fans = np.unique(np.column_stack([corner_vertex[valid], labels[valid]]), axis=0)
    per_vertex = np.bincount(fans[:, 0], minlength=topo.n_vertices)
    bad = np.nonzero(per_vertex > 1)[0]
    if len(bad):
        add("FR-SV-2", f"{len(bad)} vertices where faces form more than one fan, e.g. vertices {bad[:3].tolist()}")


def _check_orientation(mesh: Mesh, topo: _Topology, add) -> None:
    """FR-SV-3: every directed edge used once; outward normals (positive volume)."""
    dkey = topo.directed[:, 0] * topo.n_vertices + topo.directed[:, 1]
    _u, dcount = np.unique(dkey, return_counts=True)
    twice = int((dcount > 1).sum())
    if twice:
        add("FR-SV-3", f"{twice} directed edges used by two faces (inconsistent winding)")
        return
    if (topo.count != 2).any():
        return  # volume of an open or non-manifold shell is meaningless
    volume = signed_volume(mesh)
    if volume <= 0.0:
        add("FR-SV-3", f"signed volume {volume:.6g} m^3 is not positive (normals point inward)")


def _check_degeneracy(v, f, topo: _Topology, add, area_tol, dup_tol, edge_tol, tj_tol) -> None:
    """FR-SV-5: zero-area faces, duplicate vertices, edges under 1 mm, T-junctions."""
    tri = v[f]
    _n, double_area = unit_normals(tri)
    small = (0.5 * double_area < area_tol) | topo.degenerate
    if small.any():
        add("FR-SV-5", f"{int(small.sum())} zero-area faces (< {area_tol:g} m^2), e.g. faces {np.nonzero(small)[0][:3].tolist()}")

    used = np.unique(f)
    tree = cKDTree(v[used])
    pairs = tree.query_pairs(dup_tol, output_type="ndarray")
    if len(pairs):
        ex = [(int(used[a]), int(used[b])) for a, b in pairs[:3]]
        add("FR-SV-5", f"{len(pairs)} duplicate vertex pairs (< {dup_tol:g} m apart), e.g. {ex}")

    edges = topo.edge_vertices(np.ones(len(topo.ukey), dtype=bool))
    p, q = v[edges[:, 0]], v[edges[:, 1]]
    length = np.linalg.norm(q - p, axis=1)
    short = (length < edge_tol) & (edges[:, 0] != edges[:, 1])
    if short.any():
        add("FR-SV-5", f"{int(short.sum())} edges under 1 mm, e.g. {edges[short][:3].tolist()}")

    # T-junction: a used vertex inside the open interior of an edge it is not on.
    mid = 0.5 * (p + q)
    near = tree.query_ball_point(mid, length * 0.5 + tj_tol)
    counts = np.fromiter((len(n) for n in near), dtype=np.int64, count=len(near))
    if counts.sum() == 0:
        return
    edge_idx = np.repeat(np.arange(len(edges)), counts)
    vert_idx = used[np.concatenate([np.asarray(n, dtype=np.int64) for n in near if len(n)])]
    not_end = (vert_idx != edges[edge_idx, 0]) & (vert_idx != edges[edge_idx, 1])
    edge_idx, vert_idx = edge_idx[not_end], vert_idx[not_end]
    dist, t = distance_point_segment(v[vert_idx], p[edge_idx], q[edge_idx])
    along = t * length[edge_idx]
    hit = (dist <= tj_tol) & (along > tj_tol) & (along < length[edge_idx] - tj_tol)
    if hit.any():
        ex = [(int(a), edges[e].tolist()) for a, e in zip(vert_idx[hit][:3], edge_idx[hit][:3], strict=True)]
        add("FR-SV-5", f"{int(hit.sum())} T-junctions (vertex inside another edge), e.g. vertex/edge {ex}")


def self_intersecting_pairs(v: np.ndarray, f: np.ndarray, eps: float = PLANE_EPS) -> list[tuple[int, int]]:
    """FR-SV-4: face pairs that meet other than at shared vertices or a shared edge."""
    _u, canon = np.unique(v, axis=0, return_inverse=True)
    fc = canon.reshape(-1)[f]
    tri = v[f]
    normals, double_area = unit_normals(tri)
    i, j = candidate_pairs(tri)
    good = (double_area[i] > 0.0) & (double_area[j] > 0.0)
    i, j = i[good], j[good]
    if len(i) == 0:
        return []
    a, b = fc[i], fc[j]
    shared_b = (b[:, :, None] == a[:, None, :]).any(axis=2)  # which corners of B are shared
    shared_a = (a[:, :, None] == b[:, None, :]).any(axis=2)
    db = np.einsum("nj,nkj->nk", normals[i], tri[j] - tri[i][:, :1, :])
    da = np.einsum("nj,nkj->nk", normals[j], tri[i] - tri[j][:, :1, :])

    def separated(d: np.ndarray, shared: np.ndarray) -> np.ndarray:
        free = ~shared
        above = np.where(free, d > eps, True).all(axis=1)
        below = np.where(free, d < -eps, True).all(axis=1)
        return free.any(axis=1) & (above | below)

    nshared = shared_b.sum(axis=1)
    survive = ~(separated(db, shared_b) | separated(da, shared_a))
    hits: list[tuple[int, int]] = []
    free = np.nonzero(survive & (nshared == 0))[0]
    for k in free[tri_tri_intersect_batch(tri[i[free]], tri[j[free]], eps)]:
        hits.append((int(i[k]), int(j[k])))
    for k in np.nonzero(survive & (nshared > 0))[0]:
        ta, tb = tri[i[k]], tri[j[k]]
        if nshared[k] == 3:
            hit = True  # the same triangle twice
        elif nshared[k] == 2:
            hit = edge_fold(ta, tb, int(np.argmin(shared_a[k])), int(np.argmin(shared_b[k])), eps)
        else:
            hit = vertex_overlap(ta, tb, int(np.argmax(shared_a[k])), int(np.argmax(shared_b[k])), eps)
        if hit:
            hits.append((int(i[k]), int(j[k])))
    return sorted(hits)


def _check_self_intersection(v, f, add) -> None:
    hits = self_intersecting_pairs(v, f)
    if hits:
        add("FR-SV-4", f"{len(hits)} intersecting face pairs, e.g. {hits[:3]}")


def _check_components(topo: _Topology, add) -> None:
    """FR-SV-6: one shell connected through shared edges."""
    m = len(topo.faces)
    rows, cols = topo.pair_a // 3, topo.pair_b // 3
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m))
    n, labels = connected_components(graph, directed=False)
    if n > 1:
        sizes = np.bincount(labels)
        add("FR-SV-6", f"{n} connected components, face counts {sorted(sizes.tolist(), reverse=True)[:5]}")


__all__ = ["AREA_TOL", "DUP_TOL", "EDGE_TOL", "TJ_TOL", "check_solid", "self_intersecting_pairs"]
