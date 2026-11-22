"""Solid-to-solid checks for the real-time export profile (FR-SV-7, spec section 7).

Real-time profile: parts may embed into their neighbours so no seam shows, by at
least a declared minimum (default 0.2 m), and no two faces of different solids
are coplanar and overlapping (z-fighting).

Embed depth, as measured here: how far the two parts would have to be pulled
apart to stop overlapping. Where closed solids A and B pass through each other,
the faces of each that the other crosses are the entry faces, and their unit
normals are the candidate pull directions. Probe points are A's vertices and
face centroids strictly inside B, and B's strictly inside A. For a direction d,
the pull needed is the largest distance a ray from an A probe along d travels
before leaving B, or from a B probe along -d before leaving A. The embed depth
is the smallest pull over the candidate directions. So a barrier sunk 0.2 m
into a 0.3 m slab measures 0.2 m (pulled up through the slab top), a pavement
slab pushed 0.2 m into a deck end measures 0.2 m, and two boxes overlapping by
0.1 m measure 0.1 m whichever is larger. A pair that intersects with embed
depth under the minimum fails.

Solids that touch face to face without embedding produce coplanar overlapping
faces and fail on that rule. Open meshes (the L7 terrain surface) are checked
for coplanar overlap only, since "inside" is undefined for them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from roader.model.mesh import Mesh
from roader.validate._geom import (
    PLANE_EPS,
    candidate_pairs,
    coplanar_overlap_area,
    crossing_length,
    points_strictly_inside,
    unit_normals,
    vertical_hits,
)
from roader.validate.report import Violation

DEFAULT_MIN_EMBED_M = 0.2
COPLANAR_TOL = 1e-4  # m: faces this close to each other's plane z-fight
AREA_EPS = 1e-8  # m^2: overlap below this is an edge or vertex contact
CROSS_EPS = 1e-6  # m: shorter crossings are touching
EMBED_TOL = 1e-3  # m
DIRECTION_ROUND = 2  # decimals: entry-face normals this close are one pull direction


def check_realtime(meshes: Sequence[Mesh], min_embed_m: float = DEFAULT_MIN_EMBED_M) -> list[Violation]:
    """FR-SV-7 (real-time) over every pair of meshes whose bounding boxes overlap."""
    boxes = [(m.vertices.min(axis=0), m.vertices.max(axis=0)) if len(m.vertices) else None for m in meshes]
    out: list[Violation] = []
    for a in range(len(meshes)):
        for b in range(a + 1, len(meshes)):
            if boxes[a] is None or boxes[b] is None:
                continue
            (lo_a, hi_a), (lo_b, hi_b) = boxes[a], boxes[b]
            if np.any(lo_a > hi_b + 1e-6) or np.any(lo_b > hi_a + 1e-6):
                continue
            out.extend(check_pair_realtime(meshes[a], meshes[b], min_embed_m))
    return out


def check_pair_realtime(a: Mesh, b: Mesh, min_embed_m: float = DEFAULT_MIN_EMBED_M) -> list[Violation]:
    pair_id = f"{a.id}|{b.id}"
    out: list[Violation] = []
    ta, tb = a.vertices[a.faces], b.vertices[b.faces]
    na, area_a = unit_normals(ta)
    nb, area_b = unit_normals(tb)
    i, j = candidate_pairs(ta, tb, pad=max(COPLANAR_TOL, 1e-6))
    good = (area_a[i] > 0.0) & (area_b[j] > 0.0)
    i, j = i[good], j[good]
    if not len(i):
        return out
    db = np.einsum("nj,nkj->nk", na[i], tb[j] - ta[i][:, :1, :])
    da = np.einsum("nj,nkj->nk", nb[j], ta[i] - tb[j][:, :1, :])

    coplanar = (np.abs(db).max(1) <= COPLANAR_TOL) & (np.abs(da).max(1) <= COPLANAR_TOL)
    overlaps = [(int(i[k]), int(j[k])) for k in np.nonzero(coplanar)[0] if coplanar_overlap_area(ta[i[k]], tb[j[k]]) > AREA_EPS]
    if overlaps:
        out.append(Violation("FR-SV-7", pair_id, f"{len(overlaps)} coplanar overlapping face pairs (z-fighting), e.g. {overlaps[:3]}"))

    straddle = (db.max(1) > PLANE_EPS) & (db.min(1) < -PLANE_EPS) & (da.max(1) > PLANE_EPS) & (da.min(1) < -PLANE_EPS)
    crossing = np.array([k for k in np.nonzero(straddle)[0] if crossing_length(ta[i[k]], tb[j[k]]) > CROSS_EPS], dtype=np.int64)
    if not len(crossing) or not (a.closed and b.closed):
        return out
    normals = np.vstack([na[np.unique(i[crossing])], nb[np.unique(j[crossing])]])
    depth = embed_depth(a, b, normals)
    if depth < min_embed_m - EMBED_TOL:
        out.append(
            Violation(
                "FR-SV-7",
                pair_id,
                f"parts intersect but embed only {depth:.3f} m, under the declared minimum {min_embed_m:.3f} m",
            )
        )
    return out


def embed_depth(a: Mesh, b: Mesh, directions: np.ndarray) -> float:
    """Smallest pull along ``directions`` (and their opposites) that separates the probes of A and B."""
    pa = _probes_inside(a, b)
    pb = _probes_inside(b, a)
    if not len(pa) and not len(pb):
        return 0.0
    dirs = np.unique(np.round(directions, DIRECTION_ROUND), axis=0)
    dirs = dirs[np.linalg.norm(dirs, axis=1) > 0.5]
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    dirs = np.vstack([dirs, -dirs])
    best = np.inf
    for d in dirs:
        pull = 0.0
        if len(pa):
            pull = max(pull, float(exit_distance(b, pa, d).max()))
        if len(pb) and pull < best:
            pull = max(pull, float(exit_distance(a, pb, -d).max()))
        best = min(best, pull)
    return float(best)


def _probes_inside(inner: Mesh, outer: Mesh) -> np.ndarray:
    tri = inner.vertices[inner.faces]
    _n, area = unit_normals(tri)
    probes = np.vstack([inner.vertices[np.unique(inner.faces)], tri.mean(axis=1)[area > 0.0]])
    lo, hi = outer.vertices.min(axis=0), outer.vertices.max(axis=0)
    probes = probes[np.all((probes > lo) & (probes < hi), axis=1)]
    if not len(probes):
        return probes
    return probes[points_strictly_inside(outer.vertices, outer.faces, probes)]


def exit_distance(mesh: Mesh, points: np.ndarray, d: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Distance along unit direction d from each point to the first crossing of the mesh surface."""
    e3 = d / np.linalg.norm(d)
    helper = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(e3, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    rot = np.column_stack([e1, e2, e3])
    origin = points.mean(axis=0)
    v = (mesh.vertices - origin) @ rot
    p = (points - origin) @ rot
    pi, _ti, z = vertical_hits(v[mesh.faces], p[:, :2])
    dz = z - p[pi, 2]
    ahead = dz > tol
    out = np.full(len(points), np.inf)
    np.minimum.at(out, pi[ahead], dz[ahead])
    return out
