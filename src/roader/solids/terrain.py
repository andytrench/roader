"""L7 terrain: bilinear sampling and re-triangulation around solid footprints (FR-SG-6).

Grid cells away from any footprint keep their two grid triangles. A band of cells
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


def terrain_mesh(
    sampler: TerrainSampler,
    footprints: list[list[tuple[float, float]]],
    ring: list[tuple[float, float, float]],
    element_id: str,
    source: str,
) -> Mesh:
    """Open L7 surface. footprints: plan loops of every ground-contact cell; ring: exact
    xyz of every footprint boundary vertex (the solid vertices at ground level)."""
    p = sampler.patch
    nx, ny = p.nx, p.ny
    xs, ys, zg = sampler.xs, sampler.ys, sampler.z
    band = np.zeros((ny - 1, nx - 1), dtype=bool)
    region = None
    if footprints:
        fp = shapely.union_all([Polygon([_snap(x, y) for x, y in loop]) for loop in footprints])
        buf = fp.buffer(p.cell_m)
        shapely.prepare(buf)
        sx = np.array([_snap(float(x), 0.0)[0] for x in xs])
        sy = np.array([_snap(0.0, float(y))[1] for y in ys])
        x0, y0 = np.meshgrid(sx[:-1], sy[:-1])
        x1, y1 = np.meshgrid(sx[1:], sy[1:])
        boxes = shapely.box(x0, y0, x1, y1)
        band = shapely.intersects(buf, boxes)
        if band.any():
            region = shapely.difference(shapely.coverage_union_all(boxes[band]), fp)

    ring_index: dict[tuple[int, int], int] = {}
    for n, (x, y, _z) in enumerate(ring):
        ring_index.setdefault(_key(x, y), n)

    used = np.zeros((ny, nx), dtype=bool)
    kept = ~band
    used[:-1, :-1] |= kept
    used[:-1, 1:] |= kept
    used[1:, :-1] |= kept
    used[1:, 1:] |= kept

    band_tris: list[tuple[tuple[str, int, int], ...]] = []
    if region is not None and not region.is_empty:
        for part in shapely.get_parts(region):
            for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                refs = []
                for x, y in shapely.get_coordinates(tri)[:3]:
                    refs.append(_lookup(float(x), float(y), ring_index, sampler))
                band_tris.append(tuple(refs))
                for kind, a, b in refs:
                    if kind == "g":
                        used[b, a] = True

    grid_id = np.full((ny, nx), -1, dtype=np.int64)
    n_grid = int(used.sum())
    grid_id[used] = np.arange(n_grid)
    gj, gi = np.nonzero(used)
    verts = [np.column_stack([xs[gi], ys[gj], zg[gj, gi]])]

    ring_used: dict[int, int] = {}
    for tri in band_tris:
        for kind, a, _b in tri:
            if kind == "r" and a not in ring_used:
                ring_used[a] = n_grid + len(ring_used)
    if ring_used:
        verts.append(np.array([ring[n] for n in ring_used], dtype=np.float64))
    vertices = np.vstack(verts)

    kj, ki = np.nonzero(kept)
    v00, v10 = grid_id[kj, ki], grid_id[kj, ki + 1]
    v11, v01 = grid_id[kj + 1, ki + 1], grid_id[kj + 1, ki]
    faces = [np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])]
    if band_tris:
        bt = np.array(
            [[grid_id[b, a] if kind == "g" else ring_used[a] for kind, a, b in tri] for tri in band_tris],
            dtype=np.int64,
        )
        pa, pb, pc = vertices[bt[:, 0]], vertices[bt[:, 1]], vertices[bt[:, 2]]
        area2 = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
        if np.any(np.abs(area2) <= 1e-12):
            raise ValueError("FR-SG-6: degenerate terrain triangle at a footprint")
        flip = area2 < 0.0
        bt[flip] = bt[flip][:, [0, 2, 1]]
        faces.append(bt)

    x_lo, x_hi = xs[0] + 1e-6, xs[-1] - 1e-6
    y_lo, y_hi = ys[0] + 1e-6, ys[-1] - 1e-6
    for n, (x, y, _z) in enumerate(ring):
        if x_lo < x < x_hi and y_lo < y < y_hi and ring_index[_key(x, y)] == n and n not in ring_used:
            raise ValueError(f"FR-SG-6: footprint vertex ({x:.3f}, {y:.3f}) is not on the terrain boundary")

    return Mesh(
        id=f"{element_id}:terrain",
        element_id=element_id,
        layer="L7",
        part="terrain",
        material="terrain",
        source=source,
        vertices=vertices,
        faces=np.vstack(faces),
        closed=False,
    )


def _lookup(x: float, y: float, ring_index: dict[tuple[int, int], int], sampler: TerrainSampler) -> tuple[str, int, int]:
    n = ring_index.get(_key(x, y))
    if n is not None:
        return ("r", n, 0)
    p = sampler.patch
    i = round((x - p.origin_x) / p.cell_m)
    j = round((y - p.origin_y) / p.cell_m)
    if 0 <= i < p.nx and 0 <= j < p.ny and abs(sampler.xs[i] - x) < 1e-6 and abs(sampler.ys[j] - y) < 1e-6:
        return ("g", i, j)
    raise ValueError(f"FR-SG-6: terrain re-triangulation produced a vertex ({x:.4f}, {y:.4f}) that is neither grid nor solid")
