"""Pavement probability, transients and raised linear features (FR-RS-1, FR-RS-3, FR-RS-4).

All scores are geometric; vendor class codes are one feature among others
(FR-IN-3, FR-RS-5). Scores are computed per layer-cell and inherited by its points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from roader.evidence.layers import Layers, neighbour_pairs

# Pavement probability shape parameters (tolerances, not design conventions).
SIGMA_MID_M = 0.06
SIGMA_SCALE_M = 0.015
GRADIENT_MID = 0.15  # local surface gradient (grade and cross slope combined)
GRADIENT_SCALE = 0.025
INTENSITY_SPREAD = 3.0  # robust standard deviations of corridor pavement intensity
BIG_SHEET_CELLS = 200  # lateral extent: a paved surface is a large continuous sheet

# FR-RS-3 transient window above the road sheet, and vehicle footprint.
TRANSIENT_LOW_M = 0.3
TRANSIENT_HIGH_M = 4.5
VEHICLE_MAX_AREA_M2 = 60.0
VEHICLE_MAX_LENGTH_M = 24.0
# FR-RS-4 raised linear features (barriers, parapets, guardrail, curbs).
LINEAR_MIN_LENGTH_M = 10.0
LINEAR_MAX_WIDTH_M = 2.5  # a one-cell line at 45 degrees spans about two cells across
LINEAR_MIN_ASPECT = 4.0
LINEAR_MAX_HEIGHT_M = 1.8  # taller linear runs are fences or walls, not barriers
REFERENCE_RADIUS_CELLS = 2

CLASS_SCORE = {2: 1.0, 11: 1.0, 17: 1.0, 1: 0.7, 3: 0.3, 4: 0.2, 5: 0.15, 6: 0.15, 9: 0.1}
CLASS_DEFAULT = 0.5


@dataclass
class RaisedFeature:
    """FR-RS-4 evidence: a narrow raised run beside or on a sheet."""

    id: int
    kind: str  # linear | vehicle | other
    layer_idx: np.ndarray  # layer-cells in the feature
    xy: np.ndarray  # cell centers
    height_m: float  # median height above the reference sheet
    top_height_m: float  # 90th percentile of layer tops above the reference sheet
    top_z: np.ndarray  # per layer-cell top height
    ref_sheet: int
    length_m: float
    width_m: float


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def layer_class_score(layers: Layers, classification: np.ndarray) -> np.ndarray:
    lut = np.full(256, CLASS_DEFAULT)
    for code, score in CLASS_SCORE.items():
        lut[code] = score
    per_point = lut[classification.astype(np.int64)]
    total = np.bincount(layers.point_layer, weights=per_point, minlength=len(layers))
    return total / np.maximum(layers.n, 1)


def layer_mean(layers: Layers, values: np.ndarray) -> np.ndarray:
    ok = np.isfinite(values)
    total = np.bincount(layers.point_layer[ok], weights=values[ok], minlength=len(layers))
    count = np.bincount(layers.point_layer[ok], minlength=len(layers))
    out = np.full(len(layers), np.nan)
    has = count > 0
    out[has] = total[has] / count[has]
    return out


def layer_fraction(layers: Layers, flag: np.ndarray) -> np.ndarray:
    total = np.bincount(layers.point_layer, weights=flag.astype(np.float64), minlength=len(layers))
    return total / np.maximum(layers.n, 1)


def sheet_gradient(layers: Layers) -> np.ndarray:
    """Local surface gradient magnitude from same-sheet neighbours (NaN without neighbours)."""
    sheet = layers.sheet
    a, b = neighbour_pairs(layers, sheet >= 0, tol=1.0 * layers.cell_m)
    same = sheet[a] == sheet[b]
    a, b = a[same], b[same]
    gx = np.zeros(len(layers))
    gy = np.zeros(len(layers))
    cx = np.zeros(len(layers))
    cy = np.zeros(len(layers))
    dz = (layers.z[b] - layers.z[a]) / layers.cell_m
    horiz = layers.iy[b] == layers.iy[a]
    for arr, cnt, mask in ((gx, cx, horiz), (gy, cy, ~horiz)):
        np.add.at(arr, a[mask], dz[mask])
        np.add.at(arr, b[mask], dz[mask])
        np.add.at(cnt, a[mask], 1.0)
        np.add.at(cnt, b[mask], 1.0)
    out = np.full(len(layers), np.nan)
    has = (cx > 0) | (cy > 0)
    gxm = np.where(cx > 0, gx / np.maximum(cx, 1), 0.0)
    gym = np.where(cy > 0, gy / np.maximum(cy, 1), 0.0)
    out[has] = np.hypot(gxm[has], gym[has])
    return out


def pavement_probability(
    layers: Layers,
    classification: np.ndarray,
    intensity: np.ndarray,
    core: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """FR-RS-1 per layer-cell probability of a paved travel surface.

    ``core`` marks layer-cells near input centerlines; their intensity sets the
    corridor's pavement reference, so intensity enters as consistency, not as an
    absolute threshold.
    """
    sheet = layers.sheet
    counts = np.bincount(sheet[sheet >= 0], minlength=max(int(sheet.max()) + 1, 1)) if (sheet >= 0).any() else np.zeros(1)
    big = np.zeros(len(layers), dtype=bool)
    big[sheet >= 0] = counts[sheet[sheet >= 0]] >= BIG_SHEET_CELLS
    f_smooth = logistic(-(layers.sigma - SIGMA_MID_M) / SIGMA_SCALE_M) * layers.smooth
    grad = sheet_gradient(layers)
    f_grad = np.where(np.isfinite(grad), logistic(-(grad - GRADIENT_MID) / GRADIENT_SCALE), 0.5)
    f_class = layer_class_score(layers, classification)
    f_ext = np.where(big, 1.0, 0.3)
    inten = layer_mean(layers, intensity)
    ref_mask = core & big & np.isfinite(inten) & (f_smooth > 0.5)
    stats = {"intensity_ref": float("nan"), "intensity_mad": float("nan")}
    f_int = np.ones(len(layers))
    if ref_mask.sum() >= 50:
        ref = float(np.median(inten[ref_mask]))
        mad = float(np.median(np.abs(inten[ref_mask] - ref))) * 1.4826
        mad = max(mad, 1e-6 * max(abs(ref), 1.0))
        stats = {"intensity_ref": ref, "intensity_mad": mad}
        dev = np.where(np.isfinite(inten), (inten - ref) / (INTENSITY_SPREAD * mad), 0.0)
        f_int = np.exp(-0.5 * dev**2)
    # Scores are combined as a product: each feature can veto, none can confirm alone.
    prob = f_smooth * f_grad * f_class * f_ext * np.sqrt(f_int)
    prob = np.clip(prob / 0.7, 0.0, 1.0)
    return prob, stats


def reference_below(layers: Layers, mask_ref: np.ndarray, query: np.ndarray, radius: int = REFERENCE_RADIUS_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """For each query layer, the highest reference layer below it in the same or nearby cells.

    Returns (reference layer index or -1, height above it).
    """
    q = np.flatnonzero(query)
    best = np.full(len(layers), -1, dtype=np.int64)
    height = np.full(len(layers), np.nan)
    ref = np.flatnonzero(mask_ref)
    if q.size == 0 or ref.size == 0:
        return best, height
    zref = float(min(layers.z[ref].min(), layers.z[q].min())) - 1.0
    span = float(max(layers.z[ref].max(), layers.z[q].max()) - zref) + 10.0
    comp = layers.key[ref].astype(np.float64) * span + (layers.z[ref] - zref)
    order = np.argsort(comp, kind="stable")
    comp_sorted, ref_sorted = comp[order], ref[order]
    zq = layers.z[q]
    best_z = np.full(q.size, -np.inf)
    best_i = np.full(q.size, -1, dtype=np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tkey = ((layers.iy[q] + dy) * layers.nx + layers.ix[q] + dx).astype(np.float64)
            # Last reference layer in the target cell strictly below the query height.
            pos = np.searchsorted(comp_sorted, tkey * span + (zq - zref), side="left") - 1
            ok = pos >= 0
            cand = ref_sorted[np.clip(pos, 0, None)]
            ok &= layers.key[cand] == tkey.astype(np.int64)
            zc = layers.z[cand]
            better = ok & (zc < zq) & (zc > best_z)
            best_z = np.where(better, zc, best_z)
            best_i = np.where(better, cand, best_i)
    found = best_i >= 0
    best[q[found]] = best_i[found]
    height[q[found]] = zq[found] - best_z[found]
    return best, height


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected-component label per node: the smallest node index in its component."""
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _count, labels = connected_components(graph, directed=False)
    first = np.full(labels.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n))
    return first[labels]


def raised_features(layers: Layers, road_like: np.ndarray) -> tuple[list[RaisedFeature], np.ndarray]:
    """FR-RS-3 and FR-RS-4: group layers sitting 0.3 to 4.5 m above a road-like sheet.

    Returns the features and a per-layer label: 0 none, 1 linear raised feature,
    2 transient (vehicle-sized), 3 other.
    """
    label = np.zeros(len(layers), dtype=np.int8)
    query = ~road_like
    ref, h = reference_below(layers, road_like, query)
    cand = query & (ref >= 0) & (h >= TRANSIENT_LOW_M) & (h <= TRANSIENT_HIGH_M)
    a, b = neighbour_pairs(layers, cand, tol=1.0, diagonal=True)
    n = len(layers)
    roots_all = _components(n, a, b)
    idx = np.flatnonzero(cand)
    roots = roots_all[idx]
    features: list[RaisedFeature] = []
    if idx.size == 0:
        return features, label
    order = np.lexsort((idx, roots))
    idx, roots = idx[order], roots[order]
    bounds = np.flatnonzero(np.diff(roots)) + 1
    groups = np.split(idx, bounds)
    c = layers.cell_m
    for g in groups:
        xy = layers.center_xy(g)
        if len(g) >= 2:
            centered = xy - xy.mean(axis=0)
            _w, vecs = np.linalg.eigh(centered.T @ centered)
            major, minor = centered @ vecs[:, 1], centered @ vecs[:, 0]
            length = float(major.max() - major.min()) + c
            width = float(np.percentile(minor, 97.5) - np.percentile(minor, 2.5)) + c
        else:
            length = width = c
        area = len(g) * c * c
        hmed = float(np.median(h[g]))
        linear = length >= LINEAR_MIN_LENGTH_M and width <= LINEAR_MAX_WIDTH_M and length >= LINEAR_MIN_ASPECT * width
        top_h = layers.zmax[g] - layers.z[ref[g]]
        if linear and float(np.percentile(top_h, 90.0)) <= LINEAR_MAX_HEIGHT_M:
            kind, code = "linear", 1
        elif area <= VEHICLE_MAX_AREA_M2 and length <= VEHICLE_MAX_LENGTH_M:
            kind, code = "vehicle", 2
        else:
            kind, code = "other", 3
        label[g] = code
        ref_sheet = int(np.bincount(layers.sheet[ref[g]][layers.sheet[ref[g]] >= 0]).argmax()) if (layers.sheet[ref[g]] >= 0).any() else -1
        features.append(
            RaisedFeature(
                id=len(features),
                kind=kind,
                layer_idx=g,
                xy=xy,
                height_m=hmed,
                top_height_m=float(np.percentile(top_h, 90.0)),
                top_z=layers.zmax[g],
                ref_sheet=ref_sheet,
                length_m=length,
                width_m=width,
            )
        )
    return features, label
