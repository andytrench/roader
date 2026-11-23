"""Assign a road line to surface sheets station by station (FR-VC-2).

At every station the candidate surfaces are the smooth sheet layers near the
line, clustered by sheet and height. The road follows the candidate that
continues its own surface: tracking starts on the longest unambiguous run and
extends both ways with a local linear prediction of height. A deck seven meters
above the lower road is never a continuation of it, so the lower road is left
without support under the deck, and that gap is classified as occluded
(something above) or unsupported (nothing above).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roader.conflate.polyline import Polyline
from roader.evidence import Evidence
from roader.model.schema import Interval

STEP_M = 2.0
SEARCH_M = 5.0  # half width of the candidate window around the line
CLUSTER_GAP_M = 1.0  # height gap separating two candidate surfaces within one sheet
PREDICT_WINDOW_M = 30.0
TOL_BASE_M = 0.4
TOL_PER_M = 0.04  # extra height tolerance per meter since the last accepted station
CANDIDATE_P = 0.3
OCCLUDER_DZ_M = 2.0  # returns this far above the expected surface mean something is in the way
MIN_INTERVAL_M = 4.0
CURVE_GAP_M = 6.0  # beyond this gap a station no straight prediction reaches is tried on a curve
CURVE_MIN_STATIONS = 8


@dataclass
class Track:
    s: np.ndarray
    z: np.ndarray  # tracked surface height near the line (NaN where none)
    sheet: np.ndarray  # sheet index (-1 where none)
    kind: np.ndarray  # "observed" | "occluded" | "unsupported"
    ambiguous: np.ndarray

    def z_at(self, s: np.ndarray) -> np.ndarray:
        ok = np.isfinite(self.z)
        if not ok.any():
            return np.full(np.shape(s), np.nan)
        return np.interp(s, self.s[ok], self.z[ok])

    def intervals(self, length: float) -> list[Interval]:
        return station_intervals(self.s, self.kind, length)


def station_intervals(s: np.ndarray, kind: np.ndarray, length: float, reason: dict[str, str] | None = None) -> list[Interval]:
    """Runs of equal kind as Intervals over [0, length]; runs shorter than MIN_INTERVAL_M are absorbed."""
    reason = reason or {"occluded": "no surface returns; returns above", "unsupported": "no surface returns"}
    if len(s) == 0:
        return [Interval(0.0, length, "unsupported", reason.get("unsupported", ""))]
    kinds = list(kind)
    # Absorb short runs into the preceding run (or the following one at the start).
    changed = True
    while changed:
        changed = False
        runs = _runs(kinds)
        for a, b in runs:
            span = (s[b - 1] - s[a]) + STEP_M
            if span < MIN_INTERVAL_M and len(runs) > 1:
                fill = kinds[a - 1] if a > 0 else kinds[b]
                if fill != kinds[a]:
                    kinds[a:b] = [fill] * (b - a)
                    changed = True
                    break
    out = []
    for a, b in _runs(kinds):
        s0 = 0.0 if a == 0 else 0.5 * (s[a - 1] + s[a])
        s1 = length if b == len(s) else 0.5 * (s[b - 1] + s[b])
        if s1 > s0:
            out.append(Interval(float(s0), float(s1), kinds[a], reason.get(kinds[a], "")))
    return out


def _runs(kinds: list[str]) -> list[tuple[int, int]]:
    runs, start = [], 0
    for i in range(1, len(kinds) + 1):
        if i == len(kinds) or kinds[i] != kinds[start]:
            runs.append((start, i))
            start = i
    return runs


def _candidates(line: Polyline, ev: Evidence, search_m: float):
    lay = ev.layers
    cand = lay.smooth & (lay.sheet >= 0) & (ev.layer_label == 0) & (ev.layer_prob >= CANDIDATE_P)
    idx = np.flatnonzero(cand)
    s, u, ok = line.project(lay.center_xy(idx), max_dist=search_m)
    return idx[ok], s[ok], u[ok]


def track_line(line: Polyline, ev: Evidence, *, search_m: float = SEARCH_M, step: float = STEP_M) -> Track:
    lay = ev.layers
    idx, s, u, = _candidates(line, ev, search_m)
    n_st = max(2, int(np.ceil(line.length / step)) + 1)
    stations = np.linspace(0.0, line.length, n_st)
    k = np.clip(np.rint(s / (line.length / (n_st - 1))).astype(np.int64), 0, n_st - 1)
    order = np.lexsort((lay.z[idx], lay.sheet[idx], k))
    k, idx, u = k[order], idx[order], u[order]
    bounds = np.searchsorted(k, np.arange(n_st + 1))
    clusters: list[list[tuple[float, int, int]]] = []  # per station: (z, sheet, n cells)
    for i in range(n_st):
        a, b = bounds[i], bounds[i + 1]
        found = []
        if b > a:
            sh = lay.sheet[idx[a:b]]
            zz = lay.z[idx[a:b]]
            uu = np.abs(u[a:b])
            for sheet in np.unique(sh).tolist():
                m = sh == sheet
                z_s, u_s = zz[m], uu[m]
                o = np.argsort(z_s, kind="stable")
                z_s, u_s = z_s[o], u_s[o]
                cuts = np.flatnonzero(np.diff(z_s) > CLUSTER_GAP_M) + 1
                for zc, uc in zip(np.split(z_s, cuts), np.split(u_s, cuts), strict=True):
                    near = uc <= np.percentile(uc, 50) if len(uc) > 4 else np.ones(len(uc), bool)
                    found.append((float(np.median(zc[near])), int(sheet), len(zc)))
        clusters.append(found)

    z_out = np.full(n_st, np.nan)
    sheet_out = np.full(n_st, -1, dtype=np.int64)
    ambiguous = np.zeros(n_st, dtype=bool)
    single = np.array([len(c) == 1 and c[0][2] >= 3 for c in clusters])
    seed = _longest_run_center(single)
    if seed is None:
        multi = [i for i, c in enumerate(clusters) if c]
        if not multi:
            return Track(stations, z_out, sheet_out, np.array(["unsupported"] * n_st), ambiguous)
        seed = multi[len(multi) // 2]
    best = max(clusters[seed], key=lambda c: (c[2], -c[0]))
    z_out[seed], sheet_out[seed] = best[0], best[1]
    for direction in (1, -1):
        i = seed + direction
        while 0 <= i < n_st:
            pred, gap = _predict(stations, z_out, i, direction)
            tol = TOL_BASE_M + TOL_PER_M * gap
            options = [c for c in clusters[i] if abs(c[0] - pred) <= tol]
            if not options and clusters[i] and gap > CURVE_GAP_M:
                # Across a gap on a vertical curve a straight extrapolation runs off the surface (D-129).
                pred = _predict(stations, z_out, i, direction, degree=2)[0]
                options = [c for c in clusters[i] if abs(c[0] - pred) <= tol]
            if options:
                options.sort(key=lambda c: (abs(c[0] - pred), c[1]))
                z_out[i], sheet_out[i] = options[0][0], options[0][1]
                if len(options) > 1 and options[1][1] != options[0][1] and abs(options[1][0] - options[0][0]) > 0.3:
                    ambiguous[i] = True
            i += direction

    kind = np.array(["observed"] * n_st, dtype=object)
    missing = ~np.isfinite(z_out)
    if missing.any():
        ok = ~missing
        expected = np.interp(stations, stations[ok], z_out[ok]) if ok.any() else np.full(n_st, np.nan)
        xy = line.point(stations[missing])
        top = ev.cell_top(xy, radius=1)
        above = np.isfinite(top) & (top > expected[missing] + OCCLUDER_DZ_M)
        kind[np.flatnonzero(missing)] = np.where(above, "occluded", "unsupported")
    return Track(stations, z_out, sheet_out, kind.astype(str), ambiguous)


def _longest_run_center(mask: np.ndarray) -> int | None:
    best_len, best_start, start = 0, None, None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = None
    if best_start is None:
        return None
    return best_start + best_len // 2


def _predict(stations: np.ndarray, z: np.ndarray, i: int, direction: int, degree: int = 1) -> tuple[float, float]:
    """Linear (or, with enough stations, quadratic) prediction of the surface at station i from accepted stations behind it."""
    if direction == 1:
        prev = np.flatnonzero(np.isfinite(z[:i]))
    else:
        prev = np.flatnonzero(np.isfinite(z[i + 1 :])) + i + 1
    last = prev[-1] if direction == 1 else prev[0]
    gap = abs(stations[i] - stations[last])
    window = prev[np.abs(stations[prev] - stations[last]) <= PREDICT_WINDOW_M]
    if degree == 2 and len(window) >= CURVE_MIN_STATIONS:
        c = np.polyfit(stations[window] - stations[last], z[window], 2)
        return float(np.polyval(c, stations[i] - stations[last])), float(gap)
    if len(window) >= 3:
        a, b = np.polyfit(stations[window], z[window], 1)
        return float(a * stations[i] + b), float(gap)
    return float(z[last]), float(gap)
