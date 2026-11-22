"""Stage 2 metrics and stage 3 evidence for one unit corridor.

``build_evidence`` runs on the points that fitting may use (holdout already
removed, D-018). Output is per point (probability, transient flag, sheet) and per
layer-cell; nothing here fits a road.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from roader.evidence.features import (
    BIG_SHEET_CELLS,
    RaisedFeature,
    pavement_probability,
    raised_features,
)
from roader.evidence.layers import Layers, build_layers, build_sheets
from roader.evidence.strips import StripReport, correct_strips, strip_disagreement
from roader.io.points import PointSet

CORRIDOR_M = 45.0  # half width of the evidence corridor around input centerlines
CORE_M = 3.0  # layer-cells this close to a centerline set the pavement intensity reference
PAVEMENT_P = 0.5
SPARSE_PTS_M2 = 2.0  # FR-PN-3
UNOBSERVED_PTS_M2 = 0.5  # FR-PN-3


@dataclass
class SheetInfo:
    index: int
    n_cells: int
    z_min: float
    z_max: float
    centroid: tuple[float, float]
    pavement_cells: int


@dataclass
class DensityReport:
    """FR-PN-3 on candidate pavement cells."""

    median_pts_m2: float | None
    sparse_fraction: float
    unobserved_fraction: float
    n_cells: int


@dataclass
class Evidence:
    index: np.ndarray  # indices into the input PointSet
    xyz: np.ndarray  # heights strip-corrected (FR-PN-2)
    classification: np.ndarray
    intensity: np.ndarray
    point_source_id: np.ndarray
    layers: Layers
    layer_prob: np.ndarray  # FR-RS-1 per layer-cell
    layer_label: np.ndarray  # 0 surface or unlabelled, 1 linear raised, 2 transient, 3 other raised
    features: list[RaisedFeature]
    sheets: list[SheetInfo]
    strips: StripReport
    density: DensityReport
    intensity_stats: dict[str, float] = field(default_factory=dict)
    _top: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def point_layer(self) -> np.ndarray:
        return self.layers.point_layer

    @property
    def point_prob(self) -> np.ndarray:
        return self.layer_prob[self.layers.point_layer]

    @property
    def point_sheet(self) -> np.ndarray:
        return self.layers.sheet[self.layers.point_layer]

    @property
    def transient(self) -> np.ndarray:
        """FR-RS-3: points excluded from surface fitting and never meshed."""
        return self.layer_label[self.layers.point_layer] == 2

    def pavement_mask(self, p: float = PAVEMENT_P) -> np.ndarray:
        return (self.point_prob >= p) & ~self.transient

    def cell_top(self, xy: np.ndarray, radius: int = 0) -> np.ndarray:
        """Highest return height in the cell of each xy (and neighbours within radius); NaN if empty."""
        if self._top is None:
            keys, first, _count = self.layers.cell_ranges()
            top = np.maximum.reduceat(self.layers.zmax, first)
            self._top = (keys, top)
        keys, top = self._top
        ix, iy = self.layers.cell_of(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        out = np.full(len(ix), -np.inf)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                k = (iy + dy) * self.layers.nx + ix + dx
                pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
                hit = keys[pos] == k
                out = np.where(hit, np.maximum(out, top[pos]), out)
        return np.where(np.isfinite(out), out, np.nan)


def corridor_mask(xy: np.ndarray, lines: list[np.ndarray], buffer_m: float, cell_m: float = 1.0) -> np.ndarray:
    """Points within ``buffer_m`` of any line, tested per cell for speed."""
    if not lines:
        return np.ones(len(xy), dtype=bool)
    verts = []
    for line in lines:
        line = np.asarray(line, dtype=np.float64)
        for a, b in itertools.pairwise(line):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / (0.5 * cell_m))))
            t = np.arange(n)[:, None] / n
            verts.append(a + t * (b - a))
        verts.append(line[-1:])
    tree = cKDTree(np.vstack(verts))
    cells = np.floor(xy / cell_m).astype(np.int64)
    uniq, inverse = np.unique(cells, axis=0, return_inverse=True)
    centers = (uniq + 0.5) * cell_m
    d, _ = tree.query(centers, distance_upper_bound=buffer_m + cell_m)
    return (d <= buffer_m + 0.71 * cell_m)[inverse.reshape(-1)]


def build_evidence(
    points: PointSet,
    lines: list[np.ndarray],
    *,
    cell_m: float = 1.0,
    corridor_m: float = CORRIDOR_M,
) -> Evidence:
    """Evidence for the corridor around ``lines`` (input centerlines in the unit frame)."""
    xy = points.xyz[:, :2]
    index = np.flatnonzero(corridor_mask(xy, lines, corridor_m, cell_m))
    xyz = points.xyz[index].copy()
    cls = points.classification[index]
    inten = points.intensity[index]
    psid = points.point_source_id[index]
    origin = (float(np.floor(xyz[:, 0].min())) - cell_m, float(np.floor(xyz[:, 1].min())) - cell_m)

    layers, prob, stats = _surface(xyz, cls, inten, lines, cell_m, origin)
    strips = strip_disagreement(layers, xyz[:, 2], psid, prob >= PAVEMENT_P)
    corrected = correct_strips(xyz[:, 2], psid, strips)
    if strips.corrected:
        xyz[:, 2] = corrected
        layers, prob, stats = _surface(xyz, cls, inten, lines, cell_m, origin)

    road_like = (prob >= PAVEMENT_P) | ((layers.sheet >= 0) & (_sheet_sizes(layers) >= BIG_SHEET_CELLS))
    features, label = raised_features(layers, road_like)
    prob = np.where(label > 0, 0.0, prob)

    sheets = _sheet_info(layers, prob)
    pav = (prob >= PAVEMENT_P) & (label == 0)
    dens = layers.n[pav] / (cell_m * cell_m)
    density = DensityReport(
        median_pts_m2=float(np.median(dens)) if dens.size else None,
        sparse_fraction=float(np.mean(dens < SPARSE_PTS_M2)) if dens.size else 0.0,
        unobserved_fraction=float(np.mean(dens < UNOBSERVED_PTS_M2)) if dens.size else 0.0,
        n_cells=int(dens.size),
    )
    return Evidence(
        index=index,
        xyz=xyz,
        classification=cls,
        intensity=inten,
        point_source_id=psid,
        layers=layers,
        layer_prob=prob,
        layer_label=label,
        features=features,
        sheets=sheets,
        strips=strips,
        density=density,
        intensity_stats=stats,
    )


def _surface(xyz, cls, inten, lines, cell_m, origin) -> tuple[Layers, np.ndarray, dict[str, float]]:
    layers = build_layers(xyz, cell_m, origin)
    layers.sheet = build_sheets(layers)
    core = corridor_mask(layers.center_xy(), lines, CORE_M, cell_m)
    prob, stats = pavement_probability(layers, cls, inten, core)
    return layers, prob, stats


def _sheet_sizes(layers: Layers) -> np.ndarray:
    sheet = layers.sheet
    out = np.zeros(len(layers), dtype=np.int64)
    if (sheet >= 0).any():
        counts = np.bincount(sheet[sheet >= 0])
        out[sheet >= 0] = counts[sheet[sheet >= 0]]
    return out


def _sheet_info(layers: Layers, prob: np.ndarray) -> list[SheetInfo]:
    sheet = layers.sheet
    out = []
    if not (sheet >= 0).any():
        return out
    xy = layers.center_xy()
    order = np.argsort(sheet, kind="stable")
    order = order[sheet[order] >= 0]
    s_sorted = sheet[order]
    bounds = np.flatnonzero(np.diff(s_sorted)) + 1
    for g in np.split(order, bounds):
        out.append(
            SheetInfo(
                index=int(sheet[g[0]]),
                n_cells=len(g),
                z_min=float(layers.z[g].min()),
                z_max=float(layers.z[g].max()),
                centroid=(float(xy[g, 0].mean()), float(xy[g, 1].mean())),
                pavement_cells=int((prob[g] >= PAVEMENT_P).sum()),
            )
        )
    return out
