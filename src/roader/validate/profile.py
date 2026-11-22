"""Solid-to-solid checks for the analytic export profile (FR-SV-7) and tile seams (FR-SV-8).

Analytic profile: interiors of different solids never overlap, and touching
solids share conforming faces. A pair of solids fails when

- two faces pass through each other (a crossing of positive length),
- two coplanar faces overlap with the same orientation (both bodies on one side),
- a vertex or an inward-offset face centroid of one lies strictly inside the other,
- or, where faces touch back to back, a vertex of one lies on the other's face
  without coinciding with one of its vertices ("non-conforming contact").
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
    fmt_points,
    point_in_triangle,
    points_strictly_inside,
    unit_normals,
)
from roader.validate.report import Violation

AREA_EPS = 1e-8  # m^2: coplanar overlap below this is an edge or vertex contact
CROSS_EPS = 1e-6  # m: crossing segments shorter than this are touching
COINCIDE_EPS = 1e-6  # m: two vertices this close are the same vertex
INWARD_OFFSET = 1e-3  # m: face centroids pushed this far into their own solid


def check_profile(meshes: Sequence[Mesh]) -> list[Violation]:
    """FR-SV-7 over every pair of meshes whose bounding boxes overlap."""
    boxes = [(m.vertices.min(axis=0), m.vertices.max(axis=0)) if len(m.vertices) else None for m in meshes]
    out: list[Violation] = []
    for a in range(len(meshes)):
        for b in range(a + 1, len(meshes)):
            if boxes[a] is None or boxes[b] is None:
                continue
            (lo_a, hi_a), (lo_b, hi_b) = boxes[a], boxes[b]
            if np.any(lo_a > hi_b + 1e-6) or np.any(lo_b > hi_a + 1e-6):
                continue
            out.extend(check_pair(meshes[a], meshes[b]))
    return out


def check_pair(a: Mesh, b: Mesh) -> list[Violation]:
    pair_id = f"{a.id}|{b.id}"
    out: list[Violation] = []
    ta, tb = a.vertices[a.faces], b.vertices[b.faces]
    na, area_a = unit_normals(ta)
    nb, area_b = unit_normals(tb)
    i, j = candidate_pairs(ta, tb)
    good = (area_a[i] > 0.0) & (area_b[j] > 0.0)
    i, j = i[good], j[good]

    crossings: list[tuple[int, int]] = []
    same_side: list[tuple[int, int]] = []
    loose: list[np.ndarray] = []
    if len(i):
        db = np.einsum("nj,nkj->nk", na[i], tb[j] - ta[i][:, :1, :])
        da = np.einsum("nj,nkj->nk", nb[j], ta[i] - tb[j][:, :1, :])
        straddle = (db.max(1) > PLANE_EPS) & (db.min(1) < -PLANE_EPS) & (da.max(1) > PLANE_EPS) & (da.min(1) < -PLANE_EPS)
        coplanar = (np.abs(db).max(1) <= PLANE_EPS) & (np.abs(da).max(1) <= PLANE_EPS)
        for k in np.nonzero(straddle)[0]:
            if crossing_length(ta[i[k]], tb[j[k]]) > CROSS_EPS:
                crossings.append((int(i[k]), int(j[k])))
        for k in np.nonzero(coplanar)[0]:
            fa, fb = ta[i[k]], tb[j[k]]
            if coplanar_overlap_area(fa, fb) <= AREA_EPS:
                continue
            if float(na[i[k]] @ nb[j[k]]) > 0.0:
                same_side.append((int(i[k]), int(j[k])))
                continue
            loose.extend(_unmatched_contact_vertices(fa, fb))
            loose.extend(_unmatched_contact_vertices(fb, fa))

    if crossings:
        out.append(Violation("FR-SV-7", pair_id, f"{len(crossings)} face pairs pass through each other, e.g. {crossings[:3]}"))
    if same_side:
        out.append(
            Violation("FR-SV-7", pair_id, f"{len(same_side)} coplanar faces overlap with the same orientation (interiors overlap), e.g. {same_side[:3]}")
        )
    for inner, outer in ((a, b), (b, a)):
        if not outer.closed:
            continue
        pts = _probe_points(inner, outer)
        inside = points_strictly_inside(outer.vertices, outer.faces, pts)
        if inside.any():
            out.append(
                Violation(
                    "FR-SV-7",
                    pair_id,
                    f"{int(inside.sum())} points of {inner.id} inside {outer.id}, e.g. {fmt_points(pts[inside])}",
                )
            )
    if loose:
        unique = np.unique(np.round(np.array(loose), 9), axis=0)
        out.append(
            Violation(
                "FR-SV-7",
                pair_id,
                f"non-conforming contact: {len(unique)} contact vertices without a partner vertex, e.g. {fmt_points(unique)}",
            )
        )
    return out


def _unmatched_contact_vertices(fa: np.ndarray, fb: np.ndarray) -> list[np.ndarray]:
    """Vertices of face fa lying on face fb but not on one of fb's vertices (a T-junction across solids)."""
    out = []
    for p in fa:
        if np.min(np.linalg.norm(fb - p, axis=1)) <= COINCIDE_EPS:
            continue
        if point_in_triangle(p, fb, PLANE_EPS):
            out.append(p)
    return out


def _probe_points(inner: Mesh, outer: Mesh) -> np.ndarray:
    """Vertices of ``inner`` plus its face centroids pushed slightly into ``inner``, within outer's box."""
    tri = inner.vertices[inner.faces]
    normals, area = unit_normals(tri)
    centroids = tri.mean(axis=1)[area > 0.0] - INWARD_OFFSET * normals[area > 0.0]
    pts = np.vstack([inner.vertices[np.unique(inner.faces)], centroids]) if inner.closed else inner.vertices
    lo, hi = outer.vertices.min(axis=0), outer.vertices.max(axis=0)
    keep = np.all((pts > lo) & (pts < hi), axis=1)
    return pts[keep]


# ---------------------------------------------------------------- FR-SV-8 tile seams


def seam_plane_from_line(p0: tuple[float, float], p1: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Vertical seam plane through a plan line: (origin, unit normal)."""
    d = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
    d /= np.linalg.norm(d)
    return np.array([p0[0], p0[1], 0.0]), np.array([-d[1], d[0], 0.0])


def check_seam(
    side_a: Sequence[Mesh],
    side_b: Sequence[Mesh],
    *,
    origin: Sequence[float],
    normal: Sequence[float],
    tol: float = 1e-6,
    near: float = 1e-3,
    seam_id: str = "seam",
) -> list[Violation]:
    """FR-SV-8: every vertex within ``tol`` of the seam plane has a bit-identical twin across it."""
    o = np.asarray(origin, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    def on_seam(meshes: Sequence[Mesh]) -> np.ndarray:
        parts = [m.vertices[np.abs((m.vertices - o) @ n) <= tol] for m in meshes]
        pts = np.vstack(parts) if parts else np.empty((0, 3))
        return np.unique(pts, axis=0)

    pa, pb = on_seam(side_a), on_seam(side_b)
    out: list[Violation] = []
    for name, mine, other in (("A", pa, pb), ("B", pb, pa)):
        exact = {p.tobytes() for p in other}
        missing = np.array([p for p in mine if p.tobytes() not in exact]).reshape(-1, 3)
        if not len(missing):
            continue
        if len(other):
            gap = np.array([np.min(np.linalg.norm(other - p, axis=1)) for p in missing])
        else:
            gap = np.full(len(missing), np.inf)
        close = missing[gap <= near]
        lost = missing[gap > near]
        if len(close):
            out.append(
                Violation("FR-SV-8", seam_id, f"{len(close)} seam vertices on side {name} are not bit-identical across the seam, e.g. {fmt_points(close)}")
            )
        if len(lost):
            out.append(Violation("FR-SV-8", seam_id, f"{len(lost)} seam vertices on side {name} have no partner across the seam, e.g. {fmt_points(lost)}"))
    return out
