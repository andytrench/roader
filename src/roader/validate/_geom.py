"""Geometry kernels for the validity checks: triangle boxes, candidate pairs,
triangle-triangle tests and vertical ray casting.

Tolerances are absolute meters. Coordinates are reduced to a local origin per
triangle before any product, so unit-frame magnitudes (1e4 m) keep ~1e-12 m
precision.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

PLANE_EPS = 1e-7  # a vertex within this distance of a plane is on it
SHARE_EPS = 1e-6  # an intersection point this close to a shared vertex is that vertex
PARALLEL_EPS = 1e-6  # unit normals closer than this (sine of the angle) are treated as coplanar
BOX_PAD = 1e-6


def triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(M, 3, 3) corner coordinates."""
    return vertices[faces]


def unit_normals(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (M, 3) and doubled areas (M,). Degenerate faces get a zero normal."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    safe = np.where(length > 0.0, length, 1.0)
    return n / safe[:, None] * (length > 0.0)[:, None], length


def candidate_pairs(
    tri_a: np.ndarray, tri_b: np.ndarray | None = None, pad: float = BOX_PAD
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of triangles whose padded 3D boxes overlap.

    With one argument the pairs are within one set and returned with i < j.
    """
    lo_a, hi_a = tri_a.min(axis=1) - pad, tri_a.max(axis=1) + pad
    same = tri_b is None
    if same:
        lo_b, hi_b = lo_a, hi_a
    else:
        lo_b, hi_b = tri_b.min(axis=1) - pad, tri_b.max(axis=1) + pad
    if len(lo_a) == 0 or len(lo_b) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    tree = STRtree(shapely.box(lo_b[:, 0], lo_b[:, 1], hi_b[:, 0], hi_b[:, 1]))
    i, j = tree.query(shapely.box(lo_a[:, 0], lo_a[:, 1], hi_a[:, 0], hi_a[:, 1]), predicate="intersects")
    keep = (lo_a[i, 2] <= hi_b[j, 2]) & (lo_b[j, 2] <= hi_a[i, 2])
    if same:
        keep &= i < j
    return i[keep].astype(np.int64), j[keep].astype(np.int64)


# ---------------------------------------------------------------- triangle tests


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for two 3-vectors, same arithmetic, without np.cross's per-call overhead."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _seg_tri(p: np.ndarray, q: np.ndarray, tri: np.ndarray, eps: float) -> tuple[float, float] | None:
    """Parameter interval [t0, t1] of segment p->q inside the closed triangle, or None."""
    a = tri[0]
    n = _cross3(tri[1] - a, tri[2] - a)
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        return None
    n = n / nn
    dp = float(n @ (p - a))
    dq = float(n @ (q - a))
    if (dp > eps and dq > eps) or (dp < -eps and dq < -eps):
        return None
    t0, t1 = 0.0, 1.0
    if abs(dp) <= eps and abs(dq) <= eps:
        # Coplanar: clip the segment by the triangle's three inward half-planes.
        d = q - p
        for k in range(3):
            e0, e1 = tri[k], tri[(k + 1) % 3]
            m = _cross3(n, e1 - e0)
            mm = float(np.linalg.norm(m))
            f0 = float(m @ (p - e0)) + eps * mm
            df = float(m @ d)
            if abs(df) < 1e-30:
                if f0 < 0.0:
                    return None
                continue
            t = -f0 / df
            if df > 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
        return t0, t1
    t = min(max(dp / (dp - dq), 0.0), 1.0)
    x = p + t * (q - p)
    for k in range(3):
        e0, e1 = tri[k], tri[(k + 1) % 3]
        edge = e1 - e0
        if float(_cross3(edge, x - e0) @ n) < -eps * float(np.linalg.norm(edge)):
            return None
    return t, t


def tri_tri_touch(
    tri_a: np.ndarray, tri_b: np.ndarray, shared: list[np.ndarray], eps: float = PLANE_EPS, share_eps: float = SHARE_EPS
) -> bool:
    """True if the closed triangles meet anywhere other than at the shared points.

    Two triangles meet iff an edge of one meets the other. Each such contact is a
    segment interval; it is allowed only if both its ends are shared points.
    """
    for tri, other in ((tri_a, tri_b), (tri_b, tri_a)):
        for k in range(3):
            p, q = tri[k], tri[(k + 1) % 3]
            hit = _seg_tri(p, q, other, eps)
            if hit is None:
                continue
            for t in hit:
                x = p + t * (q - p)
                if all(float(np.linalg.norm(x - s)) > share_eps for s in shared):
                    return True
    return False


FLAT_SINE = 1e-6  # sine of the angle under which two face planes count as parallel
SNAPPED_SINE = 1e-3  # under this, a vertex within PLANE_EPS of the other plane need not be near the common line


def tri_tri_intersect_batch(ta: np.ndarray, tb: np.ndarray, eps: float = PLANE_EPS) -> np.ndarray:
    """Vectorized closed-triangle intersection for pairs sharing no vertex (touching counts).

    Non-coplanar pairs: each triangle meets the other's plane in a segment on the
    planes' common line; they intersect when those segments overlap. Coplanar or
    near-parallel pairs fall back to ``tri_tri_touch``.
    """
    n = len(ta)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    origin = ta[:, :1, :]
    a = ta - origin
    b = tb - origin
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    na /= np.linalg.norm(na, axis=1)[:, None]
    nb /= np.linalg.norm(nb, axis=1)[:, None]
    db = np.einsum("nj,nkj->nk", na, b - a[:, :1, :])
    da = np.einsum("nj,nkj->nk", nb, a - b[:, :1, :])
    db = np.where(np.abs(db) <= eps, 0.0, db)
    da = np.where(np.abs(da) <= eps, 0.0, da)
    apart = (db > 0).all(1) | (db < 0).all(1) | (da > 0).all(1) | (da < 0).all(1)
    line = np.cross(na, nb)
    ll = np.linalg.norm(line, axis=1)
    # Below FLAT_SINE the planes' common line is too ill-conditioned to intersect on.
    # Near-parallel planes with a vertex snapped onto the other plane also take the
    # exact path: the snapped vertex can lie far from the common line.
    snapped = (db == 0).any(1) | (da == 0).any(1)
    flat = ~apart & ((db == 0).all(1) | (da == 0).all(1) | (ll < FLAT_SINE) | (snapped & (ll < SNAPPED_SINE)))
    live = ~apart & ~flat
    if live.any():
        line = line[live] / ll[live][:, None]
        ia = _interval(a[live], da[live], line)
        ib = _interval(b[live], db[live], line)
        out[live] = np.minimum(ia[1], ib[1]) - np.maximum(ia[0], ib[0]) >= -eps
    for k in np.nonzero(flat)[0]:
        out[k] = tri_tri_touch(ta[k], tb[k], [], eps)
    return out


def _interval(tri: np.ndarray, d: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Range along ``line`` of where each triangle meets the plane its distances ``d`` refer to."""
    cand = []
    for i in range(3):
        j = (i + 1) % 3
        on = d[:, i] == 0.0
        cand.append(np.where(on, np.einsum("nj,nj->n", tri[:, i], line), np.nan))
        cross = d[:, i] * d[:, j] < 0.0
        w = np.where(cross, d[:, i] / np.where(cross, d[:, i] - d[:, j], 1.0), 0.0)
        p = tri[:, i] + (tri[:, j] - tri[:, i]) * w[:, None]
        cand.append(np.where(cross, np.einsum("nj,nj->n", p, line), np.nan))
    c = np.column_stack(cand)
    return np.nanmin(c, axis=1), np.nanmax(c, axis=1)


def edge_fold(tri_a: np.ndarray, tri_b: np.ndarray, free_a: int, free_b: int, eps: float = PLANE_EPS) -> bool:
    """Faces sharing an edge overlap only if coplanar and folded onto the same side."""
    p = tri_a[(free_a + 1) % 3]
    q = tri_a[(free_a + 2) % 3]
    n = _cross3(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
    n /= np.linalg.norm(n)
    if abs(float(n @ (tri_b[free_b] - p))) > eps:
        return False
    edge = q - p
    side_a = float(_cross3(edge, tri_a[free_a] - p) @ n)
    side_b = float(_cross3(edge, tri_b[free_b] - p) @ n)
    return side_a * side_b > 0.0


def vertex_overlap(tri_a: np.ndarray, tri_b: np.ndarray, ca: int, cb: int, eps: float = PLANE_EPS) -> bool:
    """Faces sharing only corner ca of A (= corner cb of B): do they overlap beyond it?

    Coplanar faces overlap when their angular sectors at the shared vertex do.
    Otherwise both faces meet the planes' intersection line in a segment starting
    at the shared vertex; they overlap when the two segments point the same way.
    """
    v = tri_a[ca]
    da_dirs = [tri_a[(ca + 1) % 3] - v, tri_a[(ca + 2) % 3] - v]
    db_dirs = [tri_b[(cb + 1) % 3] - v, tri_b[(cb + 2) % 3] - v]
    na = _cross3(da_dirs[0], da_dirs[1])
    nb = _cross3(db_dirs[0], db_dirs[1])
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    off_b = [float(na @ d) for d in db_dirs]
    off_a = [float(nb @ d) for d in da_dirs]
    near_parallel = float(np.linalg.norm(_cross3(na, nb))) < PARALLEL_EPS
    if near_parallel or (max(abs(x) for x in off_b) <= eps and max(abs(x) for x in off_a) <= eps):
        # Planes this close to parallel have no meaningful intersection line (FR-SV-4).
        return _sectors_overlap(da_dirs, db_dirs, na)
    sa = _ray_section(da_dirs, off_a, eps)
    sb = _ray_section(db_dirs, off_b, eps)
    if sa is None or sb is None:
        return False
    line = _cross3(na, nb)
    ll = float(np.linalg.norm(line))
    if ll < 1e-12:
        return False
    line /= ll
    ta, tb = float(line @ sa), float(line @ sb)
    return (ta > eps and tb > eps) or (ta < -eps and tb < -eps)


def _ray_section(dirs: list[np.ndarray], off: list[float], eps: float) -> np.ndarray | None:
    """Farthest point (relative to the shared vertex) where a face meets the other plane."""
    d0, d1 = dirs
    o0, o1 = off
    if abs(o0) <= eps and abs(o1) <= eps:
        return None
    if abs(o0) <= eps:
        return d0
    if abs(o1) <= eps:
        return d1
    if (o0 > 0.0) == (o1 > 0.0):
        return None
    return d0 + (d1 - d0) * (o0 / (o0 - o1))


def _sectors_overlap(da: list[np.ndarray], db: list[np.ndarray], n: np.ndarray, tol: float = 1e-12) -> bool:
    def cross(a: np.ndarray, b: np.ndarray) -> float:
        return float(_cross3(a, b) @ n) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))

    def ordered(d: list[np.ndarray]) -> list[np.ndarray]:
        return d if cross(d[0], d[1]) > 0.0 else [d[1], d[0]]

    da, db = ordered(da), ordered(db)

    def inside(w: np.ndarray, sector: list[np.ndarray]) -> bool:
        return cross(sector[0], w) > tol and cross(w, sector[1]) > tol

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return abs(cross(a, b)) <= tol and float(a @ b) > 0.0

    if any(inside(w, da) for w in db) or any(inside(w, db) for w in da):
        return True
    return same(da[0], db[0]) and same(da[1], db[1])


def _plane_section(tri: np.ndarray, d: np.ndarray, eps: float) -> list[np.ndarray]:
    pts = [tri[k] for k in range(3) if abs(d[k]) <= eps]
    for k in range(3):
        i, j = k, (k + 1) % 3
        if (d[i] > eps and d[j] < -eps) or (d[i] < -eps and d[j] > eps):
            pts.append(tri[i] + (tri[j] - tri[i]) * (d[i] / (d[i] - d[j])))
    return pts


def crossing_length(tri_a: np.ndarray, tri_b: np.ndarray, eps: float = PLANE_EPS) -> float:
    """Length along which two non-coplanar triangles pass through each other.

    Zero unless each triangle has vertices strictly on both sides of the other's
    plane; touching along an edge or at a vertex is not a crossing.
    """
    na = _cross3(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
    nb = _cross3(tri_b[1] - tri_b[0], tri_b[2] - tri_b[0])
    la, lb = float(np.linalg.norm(na)), float(np.linalg.norm(nb))
    if la == 0.0 or lb == 0.0:
        return 0.0
    na, nb = na / la, nb / lb
    db = (tri_b - tri_a[0]) @ na
    da = (tri_a - tri_b[0]) @ nb
    if not (db.max() > eps and db.min() < -eps and da.max() > eps and da.min() < -eps):
        return 0.0
    line = _cross3(na, nb)
    ll = float(np.linalg.norm(line))
    if ll < 1e-12:
        return 0.0
    line /= ll
    sa = [float(line @ (p - tri_a[0])) for p in _plane_section(tri_a, da, eps)]
    sb = [float(line @ (p - tri_a[0])) for p in _plane_section(tri_b, db, eps)]
    if not sa or not sb:
        return 0.0
    return max(0.0, min(max(sa), max(sb)) - max(min(sa), min(sb)))


def coplanar_overlap_area(tri_a: np.ndarray, tri_b: np.ndarray) -> float:
    """Area of overlap of two triangles that lie in one plane (tri_a's plane)."""
    origin = tri_a[0]
    n = _cross3(tri_a[1] - origin, tri_a[2] - origin)
    n /= np.linalg.norm(n)
    ax = tri_a[1] - origin
    ax /= np.linalg.norm(ax)
    ay = _cross3(n, ax)

    def poly(tri: np.ndarray) -> shapely.Polygon:
        rel = tri - origin
        return shapely.Polygon(np.column_stack([rel @ ax, rel @ ay]))

    a, b = poly(tri_a), poly(tri_b)
    if not a.is_valid or not b.is_valid or a.area == 0.0 or b.area == 0.0:
        return 0.0
    return float(a.intersection(b).area)


def point_in_triangle(x: np.ndarray, tri: np.ndarray, eps: float) -> bool:
    """x (assumed in the triangle's plane) inside the closed triangle, with tolerance."""
    n = _cross3(tri[1] - tri[0], tri[2] - tri[0])
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        return False
    n /= nn
    if abs(float(n @ (x - tri[0]))) > eps:
        return False
    for k in range(3):
        e0, e1 = tri[k], tri[(k + 1) % 3]
        edge = e1 - e0
        if float(_cross3(edge, x - e0) @ n) < -eps * float(np.linalg.norm(edge)):
            return False
    return True


# ---------------------------------------------------------------- rays


def vertical_hits(
    tri: np.ndarray, xy: np.ndarray, *, tree: STRtree | None = None, bary_tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where vertical lines through xy cross the triangles.

    Returns (point index, triangle index, z). Triangles seen edge-on (vertical
    faces) are skipped; a line through a shared edge may report both faces.
    """
    if len(tri) == 0 or len(xy) == 0:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0)
    if tree is None:
        tree = xy_tree(tri)
    pi, ti = tree.query(shapely.points(xy[:, 0], xy[:, 1]), predicate="intersects")
    if len(pi) == 0:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty(0)
    t = tri[ti]
    p = xy[pi]
    a = t[:, 0, :2]
    v0 = t[:, 1, :2] - a
    v1 = t[:, 2, :2] - a
    v2 = p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    scale = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    ok = np.abs(den) > 1e-12 * np.maximum(scale, 1e-300)
    den = np.where(ok, den, 1.0)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    l0 = 1.0 - l1 - l2
    inside = ok & (l0 >= -bary_tol) & (l1 >= -bary_tol) & (l2 >= -bary_tol)
    z = l0 * t[:, 0, 2] + l1 * t[:, 1, 2] + l2 * t[:, 2, 2]
    return pi[inside].astype(np.int64), ti[inside].astype(np.int64), z[inside]


def xy_tree(tri: np.ndarray, pad: float = BOX_PAD) -> STRtree:
    lo, hi = tri.min(axis=1) - pad, tri.max(axis=1) + pad
    return STRtree(shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]))


# Two tilted ray directions (dx/dz, dy/dz). Irrational-looking so rays rarely pass
# exactly through mesh edges or vertices of axis-aligned or road-aligned geometry.
_RAY_TILTS = ((0.0137, 0.0071), (-0.0093, 0.0113))


def points_strictly_inside(
    vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, tol: float = 1e-6
) -> np.ndarray:
    """Ray-parity test against a closed mesh; points on or near the surface are not inside.

    A point is inside only if two tilted upward rays both cross the surface an odd
    number of times and neither meets the surface within ``tol`` of the point.
    """
    inside = np.ones(len(points), dtype=bool)
    if len(points) == 0 or len(faces) == 0:
        return np.zeros(len(points), dtype=bool)
    for tx, ty in _RAY_TILTS:
        # Shear so the tilted ray becomes vertical: x' = x - tx z, y' = y - ty z.
        shear = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-tx, -ty, 1.0]])
        v = vertices @ shear
        p = points @ shear
        pi, _ti, z = vertical_hits(v[faces], p[:, :2])
        dz = z - p[pi, 2]
        near = np.zeros(len(points), dtype=bool)
        near[pi[np.abs(dz) <= tol]] = True
        above = np.bincount(pi[dz > tol], minlength=len(points))
        inside &= (above % 2 == 1) & ~near
    return inside


def distance_point_segment(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized: distance from x to segment p-q and the segment parameter t (unclamped)."""
    d = q - p
    dd = np.einsum("ij,ij->i", d, d)
    t = np.einsum("ij,ij->i", x - p, d) / np.where(dd > 0.0, dd, 1.0)
    foot = p + np.clip(t, 0.0, 1.0)[:, None] * d
    return np.linalg.norm(x - foot, axis=1), t


def fmt_points(points: list[tuple[float, ...]] | np.ndarray, limit: int = 3) -> str:
    items = [tuple(round(float(c), 4) for c in p) for p in list(points)[:limit]]
    return ", ".join(str(i) for i in items)
