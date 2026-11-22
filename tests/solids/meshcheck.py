"""Minimal native solid checks for the solids lane (FR-SV-1..5, FR-SV-7).

The validate lane owns the official checkers; these exist so solid generation can be
tested on its own.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from shapely.geometry import Polygon

from roader.model.mesh import Mesh, signed_volume

RAY_DIRS = np.array(
    [
        [0.2357022604, 0.4123105626, 0.8801937735],
        [-0.6123724357, 0.3535533906, 0.7071067812],
        [0.5345224838, -0.8017837257, 0.2672612419],
    ]
)


def edges(faces: np.ndarray) -> np.ndarray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def closed_and_oriented(mesh: Mesh) -> list[str]:
    """FR-SV-1, FR-SV-2 (edge part), FR-SV-3 winding."""
    problems = []
    e = edges(mesh.faces)
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2):
        problems.append(f"{int(np.sum(counts != 2))} edges not shared by exactly two faces")
    _, dcounts = np.unique(e, axis=0, return_counts=True)
    if np.any(dcounts != 1):
        problems.append(f"{int(np.sum(dcounts != 1))} directed edges repeated (inconsistent winding)")
    return problems


def vertex_fans(mesh: Mesh) -> int:
    """FR-SV-2: number of vertices whose incident faces do not form one fan."""
    bad = 0
    by_vertex: dict[int, list[int]] = {}
    for f, tri in enumerate(mesh.faces.tolist()):
        for v in tri:
            by_vertex.setdefault(v, []).append(f)
    faces = mesh.faces.tolist()
    for v, fs in by_vertex.items():
        parent = {f: f for f in fs}

        def find(a: int, parent: dict[int, int] = parent) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by_edge: dict[int, int] = {}
        for f in fs:
            for w in faces[f]:
                if w == v:
                    continue
                if w in by_edge:
                    parent[find(f)] = find(by_edge[w])
                else:
                    by_edge[w] = f
        if len({find(f) for f in fs}) != 1:
            bad += 1
    return bad


def components(mesh: Mesh) -> int:
    n = len(mesh.vertices)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges(mesh.faces).tolist():
        parent[find(a)] = find(b)
    used = np.unique(mesh.faces)
    return len({find(int(v)) for v in used})


def degeneracy(mesh: Mesh) -> list[str]:
    """FR-SV-5: zero-area faces, edges under 1 mm, duplicate vertices."""
    problems = []
    v = mesh.vertices
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if np.any(area < 1e-6):
        problems.append(f"{int(np.sum(area < 1e-6))} faces under 1 mm^2")
    e = edges(mesh.faces)
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    if np.any(length < 1e-3):
        problems.append(f"{int(np.sum(length < 1e-3))} edges under 1 mm (min {length.min():.2e})")
    if len(np.unique(v, axis=0)) != len(v):
        problems.append("duplicate vertices")
    return problems


def solid_problems(mesh: Mesh) -> list[str]:
    problems = closed_and_oriented(mesh) + degeneracy(mesh)
    if vertex_fans(mesh):
        problems.append("non-manifold vertex")
    if components(mesh) != 1:
        problems.append(f"{components(mesh)} components")
    if signed_volume(mesh) <= 0.0:
        problems.append(f"volume {signed_volume(mesh):.3f} not positive")
    return problems


def ray_hits(points: np.ndarray, mesh: Mesh, direction: np.ndarray) -> np.ndarray:
    """Number of crossings of the ray points + t * direction (t > 0) with the mesh."""
    v = mesh.vertices
    a = v[mesh.faces[:, 0]]
    e1 = v[mesh.faces[:, 1]] - a
    e2 = v[mesh.faces[:, 2]] - a
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-14
    a, e1, e2, pvec, det = a[ok], e1[ok], e2[ok], pvec[ok], det[ok]
    inv = 1.0 / det
    out = np.zeros(len(points), dtype=np.int64)
    for start in range(0, len(points), 64):
        p = points[start : start + 64]
        tvec = p[:, None, :] - a[None, :, :]
        u = np.einsum("pij,ij->pi", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None, :, :])
        w = np.einsum("j,pij->pi", direction, qvec) * inv
        t = np.einsum("pij,ij->pi", qvec, e2) * inv
        hit = (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0) & (t > 1e-9)
        out[start : start + 64] = hit.sum(axis=1)
    return out


def inside(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    votes = sum((ray_hits(points, mesh, d) % 2 == 1).astype(int) for d in RAY_DIRS)
    return votes >= 2


def interior_samples(mesh: Mesh, count: int = 40, eps: float = 1e-4) -> np.ndarray:
    v = mesh.vertices
    idx = np.linspace(0, len(mesh.faces) - 1, min(count, len(mesh.faces))).astype(int)
    f = mesh.faces[idx]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1)[:, None]
    pts = (a + b + c) / 3.0 - eps * n
    return pts[inside(pts, mesh)]


def bbox(mesh: Mesh, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return mesh.vertices.min(axis=0) - pad, mesh.vertices.max(axis=0) + pad


def boxes_overlap(a: Mesh, b: Mesh, pad: float = 1e-6) -> bool:
    lo1, hi1 = bbox(a, pad)
    lo2, hi2 = bbox(b, pad)
    return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1))


def shared_faces(a: Mesh, b: Mesh) -> int:
    """Faces of a that appear in b with opposite winding (a conforming contact)."""

    def canon(tri: list[tuple[float, ...]]) -> tuple:
        k = min(range(3), key=lambda i: tri[i])
        return tuple(tri[k:] + tri[:k])

    va, vb = [tuple(r) for r in a.vertices.tolist()], [tuple(r) for r in b.vertices.tolist()]
    rev = {canon([vb[i], vb[k], vb[j]]) for i, j, k in b.faces.tolist()}
    return sum(1 for i, j, k in a.faces.tolist() if canon([va[i], va[j], va[k]]) in rev)


def t_junctions(a: Mesh, b: Mesh, tol: float = 1e-7) -> int:
    """Vertices of a strictly inside an edge or a face of b (non-conforming contact)."""
    lo, hi = bbox(b, 1e-6)
    pts = a.vertices[np.all((a.vertices >= lo) & (a.vertices <= hi), axis=1)]
    if len(pts) == 0:
        return 0
    vb = {tuple(r) for r in b.vertices.tolist()}
    pts = np.array([p for p in pts.tolist() if tuple(p) not in vb]).reshape(-1, 3)
    if len(pts) == 0:
        return 0
    v = b.vertices
    e = np.unique(np.sort(edges(b.faces), axis=1), axis=0)
    p0, d = v[e[:, 0]], v[e[:, 1]] - v[e[:, 0]]
    dd = np.einsum("ij,ij->i", d, d)
    fa = v[b.faces[:, 0]]
    e1 = v[b.faces[:, 1]] - fa
    e2 = v[b.faces[:, 2]] - fa
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1)
    count = 0
    for start in range(0, len(pts), 32):
        p = pts[start : start + 32]
        w = p[:, None, :] - p0[None, :, :]
        t = np.einsum("pij,ij->pi", w, d) / dd
        dist = np.linalg.norm(w - t[..., None] * d[None, :, :], axis=2)
        on_edge = (dist < tol) & (t > 1e-9) & (t < 1.0 - 1e-9)
        q = p[:, None, :] - fa[None, :, :]
        h = np.einsum("pij,ij->pi", q, n) / nn
        d00 = np.einsum("ij,ij->i", e1, e1)
        d01 = np.einsum("ij,ij->i", e1, e2)
        d11 = np.einsum("ij,ij->i", e2, e2)
        d20 = np.einsum("pij,ij->pi", q, e1)
        d21 = np.einsum("pij,ij->pi", q, e2)
        den = d00 * d11 - d01 * d01
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        in_face = (np.abs(h) < tol) & (bv > 1e-9) & (bw > 1e-9) & (bv + bw < 1.0 - 1e-9)
        count += int(np.sum(on_edge.any(axis=1) | in_face.any(axis=1)))
    return count


def crossing_pairs(meshes: list[Mesh], cell: float = 2.0, eps: float = 1e-9) -> list[tuple[str, str]]:
    """FR-SV-4 and FR-SV-7: pairs of triangles (same or different solids) that cross, or that
    lie in one plane and overlap without being the same face (non-conforming contact).

    Touching along shared edges or vertices is allowed.
    """
    tris, owner = [], []
    for m, mesh in enumerate(meshes):
        tris.append(mesh.vertices[mesh.faces])
        owner += [m] * len(mesh.faces)
    tri = np.concatenate(tris)
    owner_arr = np.array(owner)
    allv = tri.reshape(-1, 3)
    _, gid = np.unique(allv, axis=0, return_inverse=True)
    gid = gid.reshape(-1, 3)
    lo = np.floor(tri.min(axis=1) / cell).astype(int)
    hi = np.floor(tri.max(axis=1) / cell).astype(int)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for t in range(len(tri)):
        for i in range(lo[t, 0], hi[t, 0] + 1):
            for j in range(lo[t, 1], hi[t, 1] + 1):
                for k in range(lo[t, 2], hi[t, 2] + 1):
                    grid.setdefault((i, j, k), []).append(t)
    pairs = set()
    for members in grid.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.add((members[x], members[y]))
    if not pairs:
        return []
    pa = np.array(sorted(pairs))
    i, j = pa[:, 0], pa[:, 1]
    tmin, tmax = tri.min(axis=1), tri.max(axis=1)
    keep = np.all((tmin[i] <= tmax[j] + eps) & (tmin[j] <= tmax[i] + eps), axis=1)
    i, j = i[keep], j[keep]
    share_count = (gid[i][:, :, None] == gid[j][:, None, :]).any(axis=2).sum(axis=1)

    def straddle(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        d = np.einsum("tvk,tk->tv", p - q[:, :1], n)
        return (d.max(axis=1) > eps) & (d.min(axis=1) < -eps), d

    s1, d1 = straddle(tri[i], tri[j])
    s2, d2 = straddle(tri[j], tri[i])
    bad = []
    cand = s1 & s2 & (share_count == 0)
    for a, b in zip(i[cand].tolist(), j[cand].tolist(), strict=True):
        if _intervals_overlap(tri[a], tri[b], eps):
            bad.append((meshes[owner_arr[a]].id, meshes[owner_arr[b]].id))
    coplanar = (np.abs(d1).max(axis=1) <= eps) & (np.abs(d2).max(axis=1) <= eps) & (share_count < 3)
    for a, b in zip(i[coplanar].tolist(), j[coplanar].tolist(), strict=True):
        if _coplanar_overlap(tri[a], tri[b]):
            bad.append((meshes[owner_arr[a]].id, meshes[owner_arr[b]].id))
    return bad


def _coplanar_overlap(t1: np.ndarray, t2: np.ndarray) -> bool:
    n = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    drop = int(np.argmax(np.abs(n)))
    axes = [k for k in range(3) if k != drop]
    p1, p2 = Polygon(t1[:, axes]), Polygon(t2[:, axes])
    return p1.intersection(p2).area > 1e-9


def _interval(t: np.ndarray, d: np.ndarray, line: np.ndarray, eps: float) -> tuple[float, float]:
    proj = t @ line
    pts = [proj[k] for k in range(3) if abs(d[k]) <= eps]
    for k, m in pairwise([0, 1, 2, 0]):
        if (d[k] > eps and d[m] < -eps) or (d[k] < -eps and d[m] > eps):
            pts.append(proj[k] + (proj[m] - proj[k]) * d[k] / (d[k] - d[m]))
    return min(pts), max(pts)


def _intervals_overlap(t1: np.ndarray, t2: np.ndarray, eps: float) -> bool:
    n1 = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    n2 = np.cross(t2[1] - t2[0], t2[2] - t2[0])
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    line = np.cross(n1, n2)
    if np.linalg.norm(line) < 1e-12:
        return False
    d1 = (t1 - t2[0]) @ n2
    d2 = (t2 - t1[0]) @ n1
    a0, a1 = _interval(t1, d1, line, eps)
    b0, b1 = _interval(t2, d2, line, eps)
    return min(a1, b1) - max(a0, b0) > 1e-7


def merge_union(meshes: list[Mesh]) -> tuple[np.ndarray, np.ndarray]:
    """Union of touching solids: weld vertices by coordinate, drop opposite face pairs."""
    verts = np.concatenate([m.vertices for m in meshes])
    offs = np.cumsum([0] + [len(m.vertices) for m in meshes])
    faces = np.concatenate([m.faces + o for m, o in zip(meshes, offs[:-1], strict=True)])
    uniq, inv = np.unique(verts, axis=0, return_inverse=True)
    faces = inv.reshape(-1)[faces]
    key = {}
    for n, tri in enumerate(faces.tolist()):
        k = min(range(3), key=lambda i: tri[i])
        key.setdefault(tuple(tri[k:] + tri[:k]), []).append(n)
    drop = set()
    for (a, b, c), ns in key.items():
        k = min(range(3), key=lambda i: (a, c, b)[i])
        rev = (a, c, b)[k:] + (a, c, b)[:k]
        if rev in key:
            drop.update(ns)
    keep = [n for n in range(len(faces)) if n not in drop]
    return uniq, faces[keep]
