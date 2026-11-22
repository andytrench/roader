"""Cross-profiles, pavement edges and lateral registration (FR-VC-3, FR-XS-1).

Every ``PROFILE_STEP_M`` along a line the surface returns near the tracked road
height are binned across the road. From the bin nearest the expected center the
pavement is followed outward while the surface stays on a continuing plane and
the intensity stays consistent with this road's pavement; an elevation break,
roughness, a raised feature or an intensity change ends it. The two ends are the
pavement edges. The midpoint of the edges, smoothed along the road, is the
lateral correction applied to the input centerline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roader.conflate.polyline import Polyline
from roader.evidence import Evidence

PROFILE_STEP_M = 5.0  # FR-VC-3: cross-profiles every 5 to 10 m
PROFILE_HALF_WINDOW_M = 2.5
BIN_M = 0.5
SEARCH_U_M = 22.0
PLANE_TOL_M = 0.06  # a bin off the continuing pavement plane by more than this ends the pavement
INTENSITY_REL = 0.2  # relative intensity band, used when wider than INTENSITY_MADS robust deviations
INTENSITY_MADS = 3.0
GAP_BINS = 2  # failing bins bridged inside the pavement (paint, a vehicle shadow)
MIN_WIDTH_M = 3.0
MAX_WIDTH_M = 30.0
WIDTH_WINDOW_M = 50.0
WIDTH_DEV_M = 1.0
WIDTH_DEV_REL = 0.1
SMOOTH_SIGMA_M = 10.0  # along-road kernel for the lateral correction
XY_SIGMA_M = 15.0  # plan smoothing of measured midpoints; local quadratic, so arcs are not biased
HEIGHT_BAND_M = 0.3
# Layer probabilities blur at the edge, where one cell holds pavement and verge,
# so edges are decided on point height and intensity; the probability only
# rejects bins that are not surface at all.
MIN_BIN_P = 0.05


@dataclass
class LinePoints:
    """Evidence points near a line, with station and offset."""

    s: np.ndarray
    u: np.ndarray
    z: np.ndarray
    intensity: np.ndarray
    prob: np.ndarray
    index: np.ndarray  # into Evidence arrays


def near_points(line: Polyline, ev: Evidence, max_u: float = SEARCH_U_M) -> LinePoints:
    usable = (ev.layer_label[ev.point_layer] == 0) & ~ev.transient
    idx = np.flatnonzero(usable)
    s, u, ok = line.project(ev.xyz[idx, :2], max_dist=max_u)
    idx = idx[ok]
    return LinePoints(s[ok], u[ok], ev.xyz[idx, 2], ev.intensity[idx], ev.point_prob[idx], idx)


@dataclass
class Profiles:
    s: np.ndarray
    left: np.ndarray  # u of the left pavement edge (NaN when not found)
    right: np.ndarray  # u of the right pavement edge (negative)
    center: np.ndarray  # u of the start bin
    slope_left: np.ndarray
    slope_right: np.ndarray
    n: np.ndarray
    intensity_ref: float

    @property
    def valid(self) -> np.ndarray:
        """Both edges found, plausible width, and width close to the local median.

        A width far from its neighbours means one edge ran into a driveway, a gore
        or a parked vehicle; such stations are not used.
        """
        w = self.left - self.right
        ok = np.isfinite(w) & (w >= MIN_WIDTH_M) & (w <= MAX_WIDTH_M)
        if ok.sum() < 3:
            return ok
        ws, ss = w[ok], self.s[ok]
        lo = np.searchsorted(ss, self.s - WIDTH_WINDOW_M, side="left")
        hi = np.searchsorted(ss, self.s + WIDTH_WINDOW_M, side="right")
        local = np.array([np.median(ws[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])
        return ok & (np.abs(w - local) <= np.maximum(WIDTH_DEV_M, WIDTH_DEV_REL * local))

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured center offset and weight per station.

        Both edges valid: their midpoint, weight 1. One edge consistent with its
        local median position: that edge moved inward by half the local median
        width, weight 0.5 (a verge as smooth and bright as the shoulder hides the
        other edge). Otherwise NaN, weight 0.
        """
        valid = self.valid
        mid = np.where(valid, 0.5 * (self.left + self.right), np.nan)
        weight = valid.astype(np.float64)
        if valid.sum() < 3:
            return mid, weight
        sv = self.s[valid]
        lo = np.searchsorted(sv, self.s - WIDTH_WINDOW_M, side="left")
        hi = np.searchsorted(sv, self.s + WIDTH_WINDOW_M, side="right")
        lv, rv = self.left[valid], self.right[valid]

        def local(values: np.ndarray) -> np.ndarray:
            return np.array([np.median(values[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])

        med_l, med_r = local(lv), local(rv)
        half = 0.5 * (med_l - med_r)
        left_ok = ~valid & np.isfinite(self.left) & (np.abs(self.left - med_l) <= WIDTH_DEV_M)
        right_ok = ~valid & ~left_ok & np.isfinite(self.right) & (np.abs(self.right - med_r) <= WIDTH_DEV_M)
        mid = np.where(left_ok, self.left - half, mid)
        mid = np.where(right_ok, self.right + half, mid)
        weight = np.where(left_ok | right_ok, 0.5, weight)
        return mid, weight


def _group_stats(key: np.ndarray, values: np.ndarray, n_keys: int) -> np.ndarray:
    out = np.full(n_keys, np.nan)
    if key.size == 0:
        return out
    order = np.lexsort((values, key))
    k, v = key[order], values[order]
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    out[uk] = 0.5 * (v[start + (cnt - 1) // 2] + v[start + cnt // 2])
    return out


def cross_profiles(
    lp: LinePoints,
    surface_z,
    length: float,
    *,
    step: float = PROFILE_STEP_M,
    center_prior=None,
    intensity_ref: float | None = None,
    search_u: float = SEARCH_U_M,
) -> Profiles:
    """Pavement edges at stations every ``step``; ``surface_z(s)`` is the road's height near u = 0."""
    n_st = max(2, int(np.ceil(length / step)) + 1)
    stations = np.linspace(0.0, length, n_st)
    n_bins = round(2 * search_u / BIN_M)
    zt = surface_z(lp.s)
    dz = lp.z - zt
    keep = np.isfinite(dz) & (np.abs(dz) <= HEIGHT_BAND_M + 0.08 * np.abs(lp.u))
    k = np.rint(lp.s / (length / (n_st - 1))).astype(np.int64)
    keep &= np.abs(lp.s - stations[np.clip(k, 0, n_st - 1)]) <= PROFILE_HALF_WINDOW_M
    b = np.floor((lp.u + search_u) / BIN_M).astype(np.int64)
    keep &= (b >= 0) & (b < n_bins)
    k, b = k[keep], b[keep]
    pu, pdz, pint, pprob = lp.u[keep], dz[keep], lp.intensity[keep], lp.prob[keep]
    by_station = np.argsort(k, kind="stable")
    st_bounds = np.searchsorted(k[by_station], np.arange(n_st + 1))
    key = k * n_bins + b
    total = n_st * n_bins
    zb = _group_stats(key, dz[keep], total).reshape(n_st, n_bins)
    ib = _group_stats(key, lp.intensity[keep], total).reshape(n_st, n_bins)
    nb = np.bincount(key, minlength=total).reshape(n_st, n_bins)
    pb = (np.bincount(key, weights=lp.prob[keep], minlength=total) / np.maximum(np.bincount(key, minlength=total), 1)).reshape(n_st, n_bins)
    uc = -search_u + (np.arange(n_bins) + 0.5) * BIN_M
    prior = np.zeros(n_st) if center_prior is None else np.asarray(center_prior(stations), dtype=np.float64)

    if intensity_ref is None:
        near = (np.abs(uc[None, :] - prior[:, None]) <= 2.0) & (nb >= 2) & (pb >= 0.5) & np.isfinite(ib)
        vals = ib[near]
        intensity_ref = float(np.median(vals)) if vals.size else float("nan")
    vals = ib[(nb >= 2) & (pb >= 0.5) & np.isfinite(ib) & (np.abs(uc[None, :] - prior[:, None]) <= 2.0)]
    mad = float(np.median(np.abs(vals - intensity_ref))) * 1.4826 if vals.size and np.isfinite(intensity_ref) else float("nan")
    band = max(INTENSITY_MADS * mad, INTENSITY_REL * abs(intensity_ref)) if np.isfinite(mad) else np.inf
    prior_pts = np.interp(lp.s[keep], stations, prior)
    central = (np.abs(pu - prior_pts) <= 2.0) & (pprob >= 0.5) & np.isfinite(pint)
    if central.sum() >= 20 and np.isfinite(intensity_ref):
        mad_pt = float(np.median(np.abs(pint[central] - intensity_ref))) * 1.4826
        band_pt = max(INTENSITY_MADS * mad_pt, INTENSITY_REL * abs(intensity_ref))
    else:
        band_pt = np.inf

    left = np.full(n_st, np.nan)
    right = np.full(n_st, np.nan)
    center = np.full(n_st, np.nan)
    sl = np.full(n_st, np.nan)
    sr = np.full(n_st, np.nan)
    count = nb.sum(axis=1)
    for i in range(n_st):
        good = (nb[i] >= 2) & np.isfinite(zb[i]) & (pb[i] >= MIN_BIN_P)
        if np.isfinite(intensity_ref):
            good &= ~(np.abs(ib[i] - intensity_ref) > band)
        if not good.any():
            continue
        start = int(np.argmin(np.where(good, np.abs(uc - prior[i]), np.inf)))
        if abs(uc[start] - prior[i]) > 6.0:
            continue
        accepted = {start: zb[i, start]}
        edges = {}
        core = good & (np.abs(uc - uc[start]) <= 1.6)
        if core.sum() >= 3:
            a_core, c_core = np.polyfit(uc[core], zb[i, core], 1)
        else:
            a_core, c_core = 0.0, float(zb[i, start])
        for direction in (1, -1):
            j, misses, last, empty_run, censored = start, 0, start, 0, False
            side_u, side_z = [uc[start]], [zb[i, start]]
            while True:
                j += direction
                if j < 0 or j >= n_bins:
                    censored = True
                    break
                empty_run = empty_run + 1 if nb[i, j] < 2 else 0
                ok = good[j]
                if ok:
                    if len(side_u) >= 4:
                        a, c0 = np.polyfit(side_u[-8:], side_z[-8:], 1)
                        pred = a * uc[j] + c0
                    else:
                        pred = a_core * uc[j] + c_core
                    ok = abs(zb[i, j] - pred) <= PLANE_TOL_M + 0.02 * misses
                if ok:
                    side_u.append(uc[j])
                    side_z.append(zb[i, j])
                    accepted[j] = zb[i, j]
                    last, misses = j, 0
                else:
                    misses += 1
                    if misses > GAP_BINS:
                        # Ended by missing data (a held-out block, the data edge), not by a break.
                        censored = empty_run > GAP_BINS
                        break
            edge = uc[last] + direction * 0.5 * BIN_M
            fit_u = np.array(side_u)
            fit_z = np.array(side_z)
            if len(fit_u) >= 4:
                slope, c0 = np.polyfit(fit_u[-8:], fit_z[-8:], 1)
                (sl if direction == 1 else sr)[i] = np.polyfit(fit_u, fit_z, 1)[0]
            else:
                slope, c0 = a_core, c_core
            sel = by_station[st_bounds[i] : st_bounds[i + 1]]
            refined = _refine_edge(pu[sel], pdz[sel], pint[sel], pprob[sel], edge, direction, slope, c0, intensity_ref, band_pt)
            edges[direction] = np.nan if censored else refined
        left[i], right[i], center[i] = edges[1], edges[-1], uc[start]
    return Profiles(stations, left, right, center, sl, sr, count, float(intensity_ref))


def _refine_edge(u, dz, inten, prob, edge, direction, slope, c0, iref, band) -> float:
    """Sub-bin edge: the threshold that best separates pavement-like points inside from others outside."""
    near = np.abs(u - edge) <= 2.0 * BIN_M
    if near.sum() < 6:
        return float(edge)
    u, dz, inten, prob = u[near], dz[near], inten[near], prob[near]
    like = np.abs(dz - (slope * u + c0)) <= 2.0 * PLANE_TOL_M
    if np.isfinite(iref) and np.isfinite(band):
        like &= ~(np.abs(inten - iref) > band)
    t = direction * u  # increasing outward
    order = np.argsort(t, kind="stable")
    t, like = t[order], like[order]
    # errors(threshold after point m) = non-pavement inside + pavement outside
    inside_bad = np.cumsum(~like)
    outside_good = like[::-1].cumsum()[::-1]
    outside_good = np.append(outside_good[1:], 0)
    err = inside_bad + outside_good
    m = int(np.argmin(err))
    cut = 0.5 * (t[m] + t[m + 1]) if m + 1 < len(t) else t[m]
    return float(direction * cut)


def smooth_series(
    s: np.ndarray,
    values: np.ndarray,
    at: np.ndarray,
    sigma: float = SMOOTH_SIGMA_M,
    iterations: int = 3,
    weights: np.ndarray | None = None,
    degree: int = 1,
) -> np.ndarray:
    """Robust Gaussian-kernel local polynomial smoothing (Huber reweighting)."""
    ok = np.isfinite(values)
    base = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    s, v, base = s[ok], values[ok], base[ok]
    if s.size == 0:
        return np.zeros(len(at))
    if s.size < 3:
        return np.full(len(at), float(np.median(v)))
    robust = base.copy()
    for _ in range(iterations):
        fit_at_data = _local_linear(s, v, robust, s, sigma, degree)
        r = v - fit_at_data
        scale = max(1.4826 * float(np.median(np.abs(r))), 0.05)
        robust = base * np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(r), 1e-12))
    return _local_linear(s, v, robust, np.asarray(at, dtype=np.float64), sigma, degree)


def _local_linear(s, v, w0, at, sigma, degree: int = 1):
    """Kernel-weighted local polynomial fit evaluated at ``at`` (degree 1 or 2)."""
    d = (at[:, None] - s[None, :]) / sigma
    w = np.exp(-0.5 * d * d) * w0[None, :]
    x = s[None, :] - at[:, None]  # centered at each evaluation station
    moments = [(w * x**k).sum(axis=1) for k in range(2 * degree + 1)]
    rhs = [(w * x**k * v[None, :]).sum(axis=1) for k in range(degree + 1)]
    out = np.empty(len(at))
    for i in range(len(at)):
        a = np.array([[moments[r + c][i] for c in range(degree + 1)] for r in range(degree + 1)])
        b = np.array([rhs[r][i] for r in range(degree + 1)])
        if np.linalg.cond(a) > 1e10:
            a = a[:1, :1]
            b = b[:1]
        out[i] = np.linalg.solve(a, b)[0] if a[0, 0] > 1e-300 else np.nan
    return out


def register_line(line: Polyline, prof: Profiles) -> tuple[Polyline, np.ndarray, np.ndarray, np.ndarray]:
    """FR-VC-3: the registered centerline, from the measured edge midpoints.

    Measured centers are placed in plan, then smoothed along the road; far from
    any measured center the input line is moved by the smoothed offset of its
    neighbours, blended in, so the correction stays smooth through gaps. Returns
    the registered line, the profile stations, the measured center offsets (NaN
    where none) and their weights.
    """
    mid, weight = prof.centers()
    has = weight > 0
    if has.sum() < 3:
        return line, prof.s, mid, weight
    # Held at its end value beyond the measured stations, never extrapolated.
    held = np.clip(prof.s, prof.s[has][0], prof.s[has][-1])
    offsets = smooth_series(prof.s[has], mid[has], held, sigma=3.0 * SMOOTH_SIGMA_M)
    measured = line.offset_point(prof.s[has], mid[has])
    sx = smooth_series(prof.s[has], measured[:, 0], prof.s, sigma=XY_SIGMA_M, weights=weight[has], degree=2)
    sy = smooth_series(prof.s[has], measured[:, 1], prof.s, sigma=XY_SIGMA_M, weights=weight[has], degree=2)
    pts = np.column_stack([sx, sy])
    sv = prof.s[has]
    dist = np.min(np.abs(prof.s[:, None] - sv[None, :]), axis=1)
    blend = np.clip(1.0 - (dist - 3.0 * XY_SIGMA_M) / XY_SIGMA_M, 0.0, 1.0)[:, None]
    fallback = line.offset_point(prof.s, offsets)
    pts = blend * pts + (1.0 - blend) * fallback
    # Away from measured centers the plan fit is an extrapolation: there each point
    # keeps its profile station and only moves sideways, so the line cannot
    # stretch along itself where edges are sparse (decks and ramps in a stack).
    s_p, u_p, ok = line.project(pts)
    # On a tight curve a point can project onto another part of the line.
    same = ok & np.isfinite(u_p) & (np.abs(s_p - np.clip(prof.s, 0.0, line.length)) <= XY_SIGMA_M)
    sideways = line.offset_point(prof.s, np.where(same, u_p, offsets))
    far = np.clip((dist - XY_SIGMA_M) / XY_SIGMA_M, 0.0, 1.0)[:, None]
    pts = (1.0 - far) * pts + far * sideways
    return Polyline(pts), prof.s, mid, weight
