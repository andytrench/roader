"""Vertical layering of returns and surface sheets (FR-RS-2, FR-PN-3).

Returns are binned into square cells. Inside a cell they are split into layers
wherever two consecutive heights differ by more than ``LAYER_GAP_M``; a deck and
the road under it, or a car roof and the lane under it, land in different
layers. Smooth layers in neighbouring cells whose heights agree are joined into
sheets, smallest height difference first, and a join is refused when the two
sheets already occupy a common cell. Each sheet is therefore a height field, and
two sheets may overlap in plan at different heights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYER_GAP_M = 0.3  # a larger vertical gap inside one cell starts a new layer
SMOOTH_SIGMA_M = 0.08  # robust spread of a smooth (paveable) layer
SMOOTH_RANGE_M = 0.4  # full height range of a smooth layer in a 1 m cell
SMOOTH_MIN_POINTS = 3
JOIN_DZ_M = 0.25  # height step allowed between neighbouring cells of one sheet (per 1 m cell)
MIN_SHEET_CELLS = 20


@dataclass
class Layers:
    """Per layer-cell statistics, sorted by (cell key, z). Points map in via ``point_layer``."""

    cell_m: float
    x0: float
    y0: float
    nx: int
    ny: int
    ix: np.ndarray
    iy: np.ndarray
    key: np.ndarray  # iy * nx + ix
    n: np.ndarray
    z: np.ndarray  # median height
    zmin: np.ndarray
    zmax: np.ndarray
    sigma: np.ndarray  # half of the 16-84 percentile range
    smooth: np.ndarray
    point_layer: np.ndarray  # (N,) layer index of each point
    sheet: np.ndarray | None = None  # (L,) sheet index or -1

    def __len__(self) -> int:
        return int(self.n.size)

    def center_xy(self, idx: np.ndarray | None = None) -> np.ndarray:
        ix = self.ix if idx is None else self.ix[idx]
        iy = self.iy if idx is None else self.iy[idx]
        return np.column_stack([self.x0 + (ix + 0.5) * self.cell_m, self.y0 + (iy + 0.5) * self.cell_m])

    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.x0) / self.cell_m).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.cell_m).astype(np.int64)
        return ix, iy

    def cell_ranges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(unique keys, first layer index, layer count) per occupied cell."""
        keys, first, count = np.unique(self.key, return_index=True, return_counts=True)
        return keys, first, count


def build_layers(xyz: np.ndarray, cell_m: float = 1.0, origin: tuple[float, float] | None = None) -> Layers:
    xyz = np.asarray(xyz, dtype=np.float64)
    if origin is None:
        origin = (float(np.floor(xyz[:, 0].min())) - cell_m, float(np.floor(xyz[:, 1].min())) - cell_m)
    x0, y0 = origin
    ix = np.floor((xyz[:, 0] - x0) / cell_m).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / cell_m).astype(np.int64)
    nx, ny = int(ix.max()) + 2, int(iy.max()) + 2
    key = iy * nx + ix
    z = xyz[:, 2]
    order = np.lexsort((z, key))
    ks, zs = key[order], z[order]
    new = np.ones(len(ks), dtype=bool)
    new[1:] = (ks[1:] != ks[:-1]) | (np.diff(zs) > LAYER_GAP_M)
    lid_sorted = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    n = np.diff(np.append(starts, len(ks)))

    def at(q: float) -> np.ndarray:
        return zs[starts + np.floor(q * (n - 1)).astype(np.int64)]

    zmed = 0.5 * (at(0.5) + zs[starts + np.ceil(0.5 * (n - 1)).astype(np.int64)])
    zmin, zmax = zs[starts], zs[starts + n - 1]
    sigma = 0.5 * (at(0.84) - at(0.16))
    smooth = (n >= SMOOTH_MIN_POINTS) & (sigma <= SMOOTH_SIGMA_M) & (zmax - zmin <= SMOOTH_RANGE_M * cell_m)
    point_layer = np.empty(len(z), dtype=np.int64)
    point_layer[order] = lid_sorted
    lk = ks[starts]
    return Layers(
        cell_m=cell_m,
        x0=x0,
        y0=y0,
        nx=nx,
        ny=ny,
        ix=lk % nx,
        iy=lk // nx,
        key=lk,
        n=n,
        z=zmed,
        zmin=zmin,
        zmax=zmax,
        sigma=sigma,
        smooth=smooth,
        point_layer=point_layer,
    )


def neighbour_pairs(layers: Layers, candidates: np.ndarray, tol: float, *, diagonal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (a, b) of candidate layers in edge-adjacent cells with |z_a - z_b| < tol.

    Each unordered pair appears once (b is in the +x or +y neighbour cell, or a
    diagonal neighbour when ``diagonal``).
    """
    idx = np.flatnonzero(candidates)
    if idx.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    zref = float(layers.z[idx].min()) - 1.0
    span = float(layers.z[idx].max() - zref) + 10.0
    comp = layers.key[idx].astype(np.float64) * span + (layers.z[idx] - zref)
    order = np.argsort(comp, kind="stable")
    comp_sorted = comp[order]
    idx_sorted = idx[order]
    steps = [(1, 0), (0, 1)]
    if diagonal:
        steps += [(1, 1), (-1, 1)]
    out_a, out_b = [], []
    for dx, dy in steps:
        tx = layers.ix[idx] + dx
        ty = layers.iy[idx] + dy
        valid = (tx >= 0) & (tx < layers.nx) & (ty >= 0) & (ty < layers.ny)
        tkey = (ty * layers.nx + tx).astype(np.float64)
        lo = np.searchsorted(comp_sorted, tkey * span + (layers.z[idx] - zref) - tol, side="left")
        hi = np.searchsorted(comp_sorted, tkey * span + (layers.z[idx] - zref) + tol, side="right")
        cnt = np.where(valid, hi - lo, 0)
        if cnt.sum() == 0:
            continue
        a = np.repeat(idx, cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        b = idx_sorted[np.repeat(lo, cnt) + offs]
        out_a.append(a)
        out_b.append(b)
    if not out_a:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_a), np.concatenate(out_b)


def build_sheets(layers: Layers, candidates: np.ndarray | None = None) -> np.ndarray:
    """Sheet index per layer (-1 for none). Sheets are numbered by their first cell key, then height."""
    cand = layers.smooth if candidates is None else candidates
    tol = JOIN_DZ_M * layers.cell_m
    a, b = neighbour_pairs(layers, cand, tol)
    dz = np.abs(layers.z[a] - layers.z[b])
    order = np.lexsort((b, a, dz))
    parent = list(range(len(layers)))
    cells: dict[int, set[int]] = {}
    keys = layers.key.tolist()

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    refused: set[tuple[int, int]] = set()
    for i, j in zip(a[order].tolist(), b[order].tolist(), strict=True):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        pair = (ri, rj) if ri < rj else (rj, ri)
        if pair in refused:
            continue
        ci = cells.get(ri) or {keys[ri]}
        cj = cells.get(rj) or {keys[rj]}
        small, large = (ci, cj) if len(ci) <= len(cj) else (cj, ci)
        if not large.isdisjoint(small):
            refused.add(pair)
            continue
        large |= small
        keep, drop = (ri, rj) if large is ci else (rj, ri)
        parent[drop] = keep
        cells[keep] = large
        cells.pop(drop, None)

    roots = np.array([find(i) for i in range(len(layers))], dtype=np.int64)
    roots[~cand] = -1
    sheet = np.full(len(layers), -1, dtype=np.int64)
    valid = roots >= 0
    if not valid.any():
        return sheet
    uniq, inverse, counts = np.unique(roots[valid], return_inverse=True, return_counts=True)
    # Deterministic numbering: by the smallest layer index in the sheet (layers are sorted by key, z).
    first = np.full(len(uniq), np.iinfo(np.int64).max)
    np.minimum.at(first, inverse, np.flatnonzero(valid))
    big = counts >= MIN_SHEET_CELLS
    rank = np.full(len(uniq), -1)
    big_idx = np.flatnonzero(big)
    rank[big_idx[np.argsort(first[big_idx], kind="stable")]] = np.arange(big_idx.size)
    sheet[np.flatnonzero(valid)] = rank[inverse]
    return sheet
