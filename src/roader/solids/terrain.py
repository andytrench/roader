"""L7 terrain: bilinear sampling and re-triangulation around solid footprints (FR-SG-6).

Grid cells away from any footprint are merged into planar quadtree blocks. A band of cells
around the footprints is re-triangulated by constrained Delaunay with the footprint
outlines as constraint edges, so terrain boundary vertices are exactly the solid
vertices where the solids meet the ground. Nothing is generated inside a footprint;
the ground under a bridge span is not a footprint and stays.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from roader.model.mesh import Mesh
from roader.model.schema import TerrainPatch

SNAP_M = 1e-7
GRID_SNAP_M = 1e-4  # footprint vertices this close to a grid line (or the patch border) sit on it


class TerrainSampler:
    def __init__(self, patch: TerrainPatch) -> None:
        if patch.nx < 2 or patch.ny < 2 or len(patch.z) != patch.nx * patch.ny:
            raise ValueError("terrain grid must be at least 2 x 2 with nx * ny values")
        if any(v is None for v in patch.z):
            raise ValueError("FR-SG-6: terrain has unobserved cells; complete them (stage 8) before solid generation")
        self.patch = patch
        self.z = np.asarray(patch.z, dtype=np.float64).reshape(patch.ny, patch.nx)
        self.xs = patch.origin_x + np.arange(patch.nx, dtype=np.float64) * patch.cell_m
        self.ys = patch.origin_y + np.arange(patch.ny, dtype=np.float64) * patch.cell_m

    def __call__(self, x: float, y: float) -> float:
        p = self.patch
        fx = min(max((x - p.origin_x) / p.cell_m, 0.0), p.nx - 1.0)
        fy = min(max((y - p.origin_y) / p.cell_m, 0.0), p.ny - 1.0)
        i = min(math.floor(fx), p.nx - 2)
        j = min(math.floor(fy), p.ny - 2)
        tx, ty = fx - i, fy - j
        z = self.z
        a = float(z[j, i]) * (1.0 - tx) + float(z[j, i + 1]) * tx
        b = float(z[j + 1, i]) * (1.0 - tx) + float(z[j + 1, i + 1]) * tx
        return a * (1.0 - ty) + b * ty


def _key(x: float, y: float) -> tuple[int, int]:
    return round(x / SNAP_M), round(y / SNAP_M)


def _snap(x: float, y: float) -> tuple[float, float]:
    """Lattice point used for the planar overlay; exact coordinates are restored by key."""
    kx, ky = _key(x, y)
    return kx * SNAP_M, ky * SNAP_M


def _overlay(lines: np.ndarray, v: float) -> float:
    """Overlay coordinate: the nearest grid line within GRID_SNAP_M, else the lattice point.

    Model values round-tripped through canonical JSON (D-011) move by micrometres; this
    keeps a footprint that ends on the patch border, or on a grid node, from leaving a
    sliver or a stray intersection vertex.
    """
    k = int(np.clip(np.searchsorted(lines, v), 1, len(lines) - 1))
    near = lines[k - 1] if abs(v - lines[k - 1]) <= abs(lines[k] - v) else lines[k]
    if abs(v - near) <= GRID_SNAP_M:
        return float(near)
    return round(v / SNAP_M) * SNAP_M


def terrain_meshes(
    sampler: TerrainSampler,
    footprints: list[list[tuple[float, float]]],
    ring: list[tuple[float, float, float]],
    element_id: str,
    source: str,
    tol: float,
) -> list[Mesh]:
    """Open L7 surface, one Mesh per edge-connected component (a CompositeSurface must be
    connected). footprints: plan loops of every ground-contact cell; ring: exact xyz of
    every footprint boundary vertex (the solid vertices at ground level). Away from the
    footprints, grid blocks that are planar within tol / 2 merge into fans (quadtree)."""
    p = sampler.patch
    nx, ny = p.nx, p.ny
    xs, ys, zg = sampler.xs, sampler.ys, sampler.z
    band = np.zeros((ny - 1, nx - 1), dtype=bool)
    region = None
    sx = np.array([_snap(float(x), 0.0)[0] for x in xs])
    sy = np.array([_snap(0.0, float(y))[1] for y in ys])

    def overlay(x: float, y: float) -> tuple[float, float]:
        return _overlay(sx, x), _overlay(sy, y)

    if footprints:
        fp = shapely.union_all([Polygon([overlay(x, y) for x, y in loop]) for loop in footprints])
        buf = fp.buffer(p.cell_m)
        shapely.prepare(buf)
        x0, y0 = np.meshgrid(sx[:-1], sy[:-1])
        x1, y1 = np.meshgrid(sx[1:], sy[1:])
        boxes = shapely.box(x0, y0, x1, y1)
        band = shapely.intersects(buf, boxes)
        if band.any():
            region = shapely.difference(shapely.coverage_union_all(boxes[band]), fp)

    ring_index: dict[tuple[int, int], int] = {}
    for n, (x, y, _z) in enumerate(ring):
        ring_index.setdefault(_key(*overlay(x, y)), n)
    n_grid = nx * ny

    def gref(j: int, i: int) -> int:
        return j * nx + i

    faces: list[tuple[int, int, int]] = []
    ring_used: set[int] = set()
    extra: list[tuple[float, float, float]] = []  # boundary vertices that are neither grid nor solid (D-072)
    extra_index: dict[tuple[int, int], int] = {}
    edges = _EdgeIndex(footprints, ring, ring_index, overlay, sampler)
    if region is not None and not region.is_empty:
        for part in shapely.get_parts(region):
            for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                refs = []
                for x, y in shapely.get_coordinates(tri)[:3]:
                    kind, a, b = _lookup(float(x), float(y), ring_index, sampler)
                    if kind == "r":
                        ring_used.add(a)
                        refs.append(n_grid + a)
                    elif kind == "g":
                        refs.append(gref(b, a))
                    else:
                        key = _key(float(x), float(y))
                        if key not in extra_index:
                            extra_index[key] = len(extra)
                            extra.append((float(x), float(y), edges.z(float(x), float(y))))
                        refs.append(-1 - extra_index[key])
                faces.append((refs[0], refs[1], refs[2]))

    leaves = _leaves(band, zg, tol)
    used = np.zeros((ny, nx), dtype=bool)
    bj, bi = np.nonzero(band)
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[bj + dj, bi + di] = True
    for j0, i0, size in leaves:
        used[[j0, j0, j0 + size, j0 + size], [i0, i0 + size, i0, i0 + size]] = True
    for j0, i0, size in leaves:
        if size == 1:
            a, b, c, d = gref(j0, i0), gref(j0, i0 + 1), gref(j0 + 1, i0 + 1), gref(j0 + 1, i0)
            faces += [(a, b, c), (a, c, d)]
            continue
        loop = [(j0, i) for i in range(i0, i0 + size)]
        loop += [(j, i0 + size) for j in range(j0, j0 + size)]
        loop += [(j0 + size, i) for i in range(i0 + size, i0, -1)]
        loop += [(j, i0) for j in range(j0 + size, j0, -1)]
        loop = [gref(j, i) for j, i in loop if used[j, i]]
        centre = gref(j0 + size // 2, i0 + size // 2)
        faces += [(centre, loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))]

    refs = np.array(faces, dtype=np.int64).reshape(-1, 3)
    ring_arr = np.array(ring, dtype=np.float64).reshape(-1, 3)
    refs = np.where(refs < 0, n_grid + len(ring_arr) - 1 - refs, refs)
    gj, gi = np.divmod(np.arange(n_grid), nx)
    allv = np.vstack([np.column_stack([xs[gi], ys[gj], zg[gj, gi]]), ring_arr, np.array(extra, dtype=np.float64).reshape(-1, 3)])
    pa, pb, pc = allv[refs[:, 0]], allv[refs[:, 1]], allv[refs[:, 2]]
    area2 = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
    # Overlay points within the snap distance of one vertex collapse a triangle; it carries no surface (D-072).
    keep = np.abs(area2) > 1e-12
    collapsed = (refs[:, 0] == refs[:, 1]) | (refs[:, 1] == refs[:, 2]) | (refs[:, 2] == refs[:, 0])
    if np.any(~keep & ~collapsed & (refs < n_grid).all(axis=1)):
        raise ValueError("FR-SG-6: degenerate terrain triangle")
    refs, area2 = refs[keep], area2[keep]
    flip = area2 < 0.0
    refs[flip] = refs[flip][:, [0, 2, 1]]

    x_lo, x_hi = xs[0] + GRID_SNAP_M, xs[-1] - GRID_SNAP_M
    y_lo, y_hi = ys[0] + GRID_SNAP_M, ys[-1] - GRID_SNAP_M
    for n, (x, y, _z) in enumerate(ring):
        if (
            x_lo < x < x_hi and y_lo < y < y_hi and ring_index[_key(*overlay(x, y))] == n and n not in ring_used
            and not edges.covered(x, y)
        ):
            raise ValueError(f"FR-SG-6: footprint vertex ({x:.3f}, {y:.3f}) is not on the terrain boundary")

    labels = _face_components(refs)
    parts = []
    for c in range(int(labels.max()) + 1 if len(labels) else 0):
        f = refs[labels == c]
        keep, local = np.unique(f, return_inverse=True)
        verts = allv[keep]
        # Millimetre keys: ids must not flip when the model moves by micrometres (D-011 round trip).
        parts.append((round(float(verts[:, 0].min()), 3), round(float(verts[:, 1].min()), 3), verts, local.reshape(-1, 3)))
    parts.sort(key=lambda item: (item[0], item[1]))
    return [
        Mesh(
            id=f"{element_id}:terrain" if len(parts) == 1 else f"{element_id}:terrain:{k}",
            element_id=element_id,
            layer="L7",
            part="terrain",
            material="terrain",
            source=source,
            vertices=verts,
            faces=f,
            closed=False,
        )
        for k, (_x, _y, verts, f) in enumerate(parts)
    ]


def _leaves(band: np.ndarray, zg: np.ndarray, tol: float) -> list[tuple[int, int, int]]:
    """Quadtree over grid cells: blocks free of the band and planar within tol / 2, else cells."""
    ncy, ncx = band.shape
    csum = np.zeros((ncy + 1, ncx + 1), dtype=np.int64)
    csum[1:, 1:] = np.cumsum(np.cumsum(band, axis=0), axis=1)
    size = 1
    while size < max(ncx, ncy):
        size *= 2
    leaves: list[tuple[int, int, int]] = []

    def planar(j0: int, i0: int, s: int) -> bool:
        z = zg[j0 : j0 + s + 1, i0 : i0 + s + 1]
        jj, ii = np.mgrid[0 : s + 1, 0 : s + 1]
        a = np.column_stack([np.ones(z.size), ii.ravel(), jj.ravel()])
        coef = np.linalg.lstsq(a, z.ravel(), rcond=None)[0]
        return bool(np.abs(a @ coef - z.ravel()).max() <= 0.5 * tol)

    stack = [(0, 0, size)]
    while stack:
        j0, i0, s = stack.pop()
        if j0 >= ncy or i0 >= ncx:
            continue
        if s == 1:
            if not band[j0, i0]:
                leaves.append((j0, i0, 1))
            continue
        if j0 + s <= ncy and i0 + s <= ncx:
            n_band = csum[j0 + s, i0 + s] - csum[j0, i0 + s] - csum[j0 + s, i0] + csum[j0, i0]
            if n_band == 0 and planar(j0, i0, s):
                leaves.append((j0, i0, s))
                continue
        h = s // 2
        stack += [(j0 + h, i0 + h, h), (j0 + h, i0, h), (j0, i0 + h, h), (j0, i0, h)]
    return leaves


def _face_components(faces: np.ndarray) -> np.ndarray:
    """Component label per face; faces connect through shared edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(faces)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(n), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[order], fid[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = fid[:-1][same], fid[1:][same]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _count, labels = connected_components(graph, directed=False)
    # Relabel by first face so labels follow face order (deterministic).
    _, first = np.unique(labels, return_index=True)
    rank = np.empty_like(first)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[labels]


def _lookup(x: float, y: float, ring_index: dict[tuple[int, int], int], sampler: TerrainSampler) -> tuple[str, int, int]:
    """("r", ring index, 0), ("g", column, row), or ("x", 0, 0) for a vertex that is neither."""
    n = ring_index.get(_key(x, y))
    if n is not None:
        return ("r", n, 0)
    p = sampler.patch
    i = round((x - p.origin_x) / p.cell_m)
    j = round((y - p.origin_y) / p.cell_m)
    if 0 <= i < p.nx and 0 <= j < p.ny and abs(sampler.xs[i] - x) <= SNAP_M and abs(sampler.ys[j] - y) <= SNAP_M:
        return ("g", i, j)
    return ("x", 0, 0)


class _EdgeIndex:
    """Footprint outline edges, for boundary vertices the overlay adds where footprints of
    different solids cross or a footprint leaves the patch (D-072): such a vertex takes the
    ground level interpolated along the outline edge it lies on, else the sampled terrain."""

    def __init__(self, footprints, ring, ring_index, overlay, sampler: TerrainSampler) -> None:
        self.footprints, self.ring, self.ring_index = footprints, ring, ring_index
        self.overlay, self.sampler = overlay, sampler
        self._tree = None
        self._union = None

    def _build(self) -> None:
        segs, ends = [], []
        for loop in self.footprints:
            pts = [self.overlay(x, y) for x, y in loop]
            for k in range(len(pts)):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                if a != b:
                    segs.append(shapely.linestrings([a, b]))
                    ends.append((a, b))
        self._segs, self._ends = segs, ends
        self._tree = shapely.STRtree(segs)

    def _level(self, x: float, y: float) -> float:
        n = self.ring_index.get(_key(x, y))
        return float(self.ring[n][2]) if n is not None else self.sampler(x, y)

    def z(self, x: float, y: float) -> float:
        if self._tree is None:
            self._build()
        pt = shapely.points(x, y)
        best = None
        for k in self._tree.query(pt, predicate="dwithin", distance=1e-6):
            d = shapely.distance(self._segs[int(k)], pt)
            if best is None or d < best[0]:
                best = (d, int(k))
        if best is None:
            return self.sampler(x, y)
        (ax, ay), (bx, by) = self._ends[best[1]]
        length2 = (bx - ax) ** 2 + (by - ay) ** 2
        t = min(max(((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / length2, 0.0), 1.0)
        return (1.0 - t) * self._level(ax, ay) + t * self._level(bx, by)

    def covered(self, x: float, y: float) -> bool:
        """A footprint vertex strictly inside the union of footprints (solids overlap there)."""
        if self._union is None:
            self._union = shapely.union_all([Polygon([self.overlay(a, b) for a, b in loop]) for loop in self.footprints])
            shapely.prepare(self._union)
        p = shapely.points(*self.overlay(x, y))
        return bool(self._union.contains(p)) and shapely.distance(self._union.boundary, p) > 1e-6
