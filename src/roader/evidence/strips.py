"""Flight-strip vertical disagreement on pavement (FR-PN-2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roader.evidence.layers import Layers

STRIP_TOL_M = 0.05  # FR-PN-2: above this, correct per strip
STRIP_BLOCK_M = 10.0  # cell differences are pooled in blocks so per-cell noise does not count as disagreement
STRIP_MIN_POINTS = 3  # per strip per cell
STRIP_MIN_BLOCKS = 5


@dataclass
class StripReport:
    reference_strip: int | None
    offsets_m: dict[int, float] = field(default_factory=dict)  # strip z minus reference strip z
    p95_m: float | None = None  # 95th percentile of |block median difference|
    n_blocks: int = 0
    corrected: list[int] = field(default_factory=list)


def strip_disagreement(layers: Layers, z: np.ndarray, strip: np.ndarray, pavement: np.ndarray) -> StripReport:
    """Per-strip height offsets against the most populous strip, on pavement layers only."""
    on = pavement[layers.point_layer]
    if not on.any():
        return StripReport(None)
    strips, counts = np.unique(strip[on], return_counts=True)
    if len(strips) < 2:
        return StripReport(int(strips[0]) if len(strips) else None)
    ref = int(strips[np.argmax(counts)])
    lid = layers.point_layer[on]
    zs = z[on]
    ss = strip[on].astype(np.int64)
    report = StripReport(ref)
    diffs_all = []
    for other in strips.tolist():
        if other == ref:
            continue
        med = {}
        for code in (ref, other):
            m = ss == code
            order = np.lexsort((zs[m], lid[m]))
            l_sorted, z_sorted = lid[m][order], zs[m][order]
            keys, start, cnt = np.unique(l_sorted, return_index=True, return_counts=True)
            mid = z_sorted[start + (cnt - 1) // 2]
            good = cnt >= STRIP_MIN_POINTS
            med[code] = dict(zip(keys[good].tolist(), mid[good].tolist(), strict=True))
        common = sorted(set(med[ref]) & set(med[other]))
        if not common:
            continue
        idx = np.array(common)
        d = np.array([med[other][i] - med[ref][i] for i in common])
        bx = np.floor(layers.ix[idx] * layers.cell_m / STRIP_BLOCK_M).astype(np.int64)
        by = np.floor(layers.iy[idx] * layers.cell_m / STRIP_BLOCK_M).astype(np.int64)
        block = by * 1_000_003 + bx
        block_med = group_median(d, block)
        if len(block_med) < STRIP_MIN_BLOCKS:
            continue
        report.offsets_m[int(other)] = float(np.median(block_med))
        diffs_all.append(block_med)
    if diffs_all:
        pooled = np.abs(np.concatenate(diffs_all))
        report.p95_m = float(np.percentile(pooled, 95.0))
        report.n_blocks = int(pooled.size)
    return report


def group_median(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Median of values per group, groups in ascending order."""
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    _keys, start, cnt = np.unique(g, return_index=True, return_counts=True)
    lo = v[start + (cnt - 1) // 2]
    hi = v[start + cnt // 2]
    return 0.5 * (lo + hi)


def correct_strips(z: np.ndarray, strip: np.ndarray, report: StripReport) -> np.ndarray:
    """Remove per-strip offsets above STRIP_TOL_M (FR-PN-2). Returns corrected heights."""
    out = z.copy()
    for code, offset in sorted(report.offsets_m.items()):
        if abs(offset) > STRIP_TOL_M:
            out[strip == code] -= offset
            report.corrected.append(code)
    return out
