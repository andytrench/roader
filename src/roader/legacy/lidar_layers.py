"""LiDAR terrain and elevated samples via laspy.

PDAL is optional (`pip install -e ".[pdal]"`) for heavier pipelines later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from laspy import LasData, read as las_read
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union


@dataclass
class LidarLayers:
    ground_xy: np.ndarray  # (N, 2)
    ground_z: np.ndarray
    bridge_xy: np.ndarray
    bridge_z: np.ndarray
    all_xy: np.ndarray
    all_z: np.ndarray
    all_class: np.ndarray
    crs_epsg: int | None
    ground_intensity: np.ndarray | None = None
    bridge_intensity: np.ndarray | None = None
    all_intensity: np.ndarray | None = None
    has_intensity: bool = False


def _load_las(path: Path) -> LasData:
    return las_read(str(path))


def _intensity_values(las: LasData) -> np.ndarray | None:
    """Return per-point intensity, or None when the dimension is absent.

    Missing intensity is a cue gap, not a failed read. Callers keep going on height.
    """
    names = {str(name).lower() for name in las.point_format.dimension_names}
    if "intensity" not in names:
        return None
    try:
        values = np.asarray(las.intensity, dtype=np.float64)
    except (AttributeError, ValueError):
        return None
    return values


def _layers_from_las(
    las: LasData,
    bbox_projected: tuple[float, float, float, float] | None = None,
) -> LidarLayers:
    """Split a LAS dataset into ground (2) and bridge deck (17)."""
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification)
    intensity = _intensity_values(las)
    if intensity is not None and len(intensity) != len(x):
        intensity = None

    if bbox_projected is not None and len(x):
        west, south, east, north = bbox_projected
        keep = (x >= west) & (x <= east) & (y >= south) & (y <= north)
        x, y, z, cls = x[keep], y[keep], z[keep], cls[keep]
        if intensity is not None:
            intensity = intensity[keep]

    xy = np.column_stack([x, y]) if len(x) else np.zeros((0, 2))
    ground = cls == 2
    bridge = cls == 17
    epsg = None
    try:
        crs = las.header.parse_crs()
        if crs is not None and crs.to_epsg():
            epsg = int(crs.to_epsg())
    except Exception:
        epsg = None

    def _take(mask: np.ndarray) -> np.ndarray | None:
        if intensity is None:
            return None
        return intensity[mask]

    return LidarLayers(
        ground_xy=xy[ground],
        ground_z=z[ground],
        bridge_xy=xy[bridge],
        bridge_z=z[bridge],
        all_xy=xy,
        all_z=z,
        all_class=cls,
        crs_epsg=epsg,
        ground_intensity=_take(ground),
        bridge_intensity=_take(bridge),
        all_intensity=intensity,
        has_intensity=intensity is not None,
    )


def load_lidar_layers(
    path: Path,
    bbox_projected: tuple[float, float, float, float] | None = None,
) -> LidarLayers:
    """Load LAS/LAZ and split ground (2) vs bridge deck (17).

    bbox_projected is in the same CRS as the point cloud coordinates when set.
    Intensity is kept when the point format has that dimension.
    """
    return _layers_from_las(_load_las(path), bbox_projected)


def estimate_ground_z(
    layers: LidarLayers,
    query_xy: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Nearest-neighbour ground height at query points (NaN if no ground)."""
    if len(layers.ground_xy) == 0:
        return np.full(len(query_xy), np.nan)
    from scipy.spatial import cKDTree

    tree = cKDTree(layers.ground_xy)
    dist, idx = tree.query(query_xy, k=min(k, len(layers.ground_xy)))
    if dist.ndim == 1:
        return layers.ground_z[idx]
    # weighted mean of k neighbours
    w = 1.0 / np.maximum(dist, 0.05)
    w /= w.sum(axis=1, keepdims=True)
    return (layers.ground_z[idx] * w).sum(axis=1)


def elevated_candidate_points(
    layers: LidarLayers,
    height_above_ground_m: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-ground points that sit above local ground — deck candidates.

    Prefers class 17 when present; otherwise uses unclassified / other
    elevated returns (not vegetation classes 3–5 by default).
    """
    if len(layers.bridge_xy):
        return layers.bridge_xy, layers.bridge_z

    # Fallback: points not ground/veg that clear local ground
    mask = ~np.isin(layers.all_class, [2, 3, 4, 5, 7, 18])
    if not mask.any():
        return np.zeros((0, 2)), np.zeros((0,))
    xy = layers.all_xy[mask]
    z = layers.all_z[mask]
    gz = estimate_ground_z(layers, xy)
    keep = np.isfinite(gz) & ((z - gz) >= height_above_ground_m)
    return xy[keep], z[keep]


def points_to_alpha_outline(
    xy: np.ndarray,
    cell_m: float = 1.0,
) -> MultiPoint | None:
    """Coarse outline via gridded occupancy union (no CGAL dependency)."""
    if len(xy) < 10:
        return None
    ix = np.floor(xy[:, 0] / cell_m).astype(np.int64)
    iy = np.floor(xy[:, 1] / cell_m).astype(np.int64)
    cells = {}
    for x_i, y_i in zip(ix, iy, strict=False):
        cells[(int(x_i), int(y_i))] = True
    polys = [
        box(i * cell_m, j * cell_m, (i + 1) * cell_m, (j + 1) * cell_m)
        for i, j in cells
    ]
    return unary_union(polys)
