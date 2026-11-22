"""L2 vertical profile fitting (FR-AF-2, FR-AF-4, FR-AF-5, FR-OC-1).

The profile is a chain of constant grades joined at PVIs by symmetric parabolic
vertical curves. It is parameterized by the start elevation, the tangent grades,
the PVI stations and the curve lengths, so elevation and grade are continuous by
construction. Fitting is robust least squares (soft L1) on 1 m bin medians of
cross-slope-corrected returns. PVIs start at the vertices of a Douglas-Peucker
simplification of the binned profile; PVIs that do not pay for themselves are
removed, and a curve lying mostly in an unobserved gap is removed (FR-OC-1:
never invent a sag or bump in a gap). Grades and curve K values enter as soft
priors from the rule library and are flagged, never clamped, when the data put
them outside the rule range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from roader.fit.priors import bound_penalty
from roader.model.geometry import Profile, VerticalElement
from roader.model.provenance import Provenance, Source
from roader.model.serialize import FLOAT_DIGITS
from roader.rules import RuleBook

BIN_M = 1.0
DP_TOL_M = 0.08
F_SCALE_M = 0.05
MIN_CURVE_M = 10.0
MIN_DGRADE = 0.0015  # PVIs with a smaller grade change are tried for removal
PRUNE_COST_RATIO = 1.15
PRUNE_TRIES = 4
PRUNE_RMS_RISE_M = 0.005  # a simplification costing less than this RMS is always taken
MAX_PVI_PER_M = 1.0 / 40.0
MIN_CURVE_OBSERVED = 0.3  # FR-OC-1
RULE_WEIGHT = 0.3
IRLS_STEPS = 5


@dataclass
class ProfileFit:
    profile: Profile
    rms_m: float
    p95_m: float
    n_points: int
    flags: list[str] = field(default_factory=list)


@dataclass
class _Vert:
    z0: float
    grades: np.ndarray  # n_pvi + 1
    pvi: np.ndarray  # n_pvi stations
    lengths: np.ndarray  # n_pvi curve lengths

    def z(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        knots = np.concatenate([[0.0], self.pvi])
        zk = self.z0 + np.concatenate([[0.0], np.cumsum(self.grades[:-1] * np.diff(knots))])
        seg = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(knots) - 1)
        out = zk[seg] + self.grades[seg] * (s - knots[seg])
        for k, (v, length) in enumerate(zip(self.pvi, self.lengths, strict=True)):
            dg = self.grades[k + 1] - self.grades[k]
            t = s - (v - 0.5 * length)
            inside = (t > 0.0) & (t < length)
            m = np.minimum(t, length - t)
            out = out + np.where(inside, dg * m * m / (2.0 * length), 0.0)
        return out

    def grade(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        knots = np.concatenate([[0.0], self.pvi])
        seg = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(knots) - 1)
        out = self.grades[seg].astype(np.float64)
        for k, (v, length) in enumerate(zip(self.pvi, self.lengths, strict=True)):
            g0, g1 = self.grades[k], self.grades[k + 1]
            t = s - (v - 0.5 * length)
            inside = (t > 0.0) & (t < length)
            out = np.where(inside, g0 + (g1 - g0) * t / length, out)
        return out


def _bins(s: np.ndarray, z: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.floor(s / BIN_M).astype(np.int64)
    ok = (k >= 0) & (s <= length)
    k, zz = k[ok], z[ok]
    order = np.lexsort((zz, k))
    k, zz = k[order], zz[order]
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    med = 0.5 * (zz[start + (cnt - 1) // 2] + zz[start + cnt // 2])
    keep = cnt >= 2
    return (uk[keep] + 0.5) * BIN_M, med[keep], cnt[keep].astype(np.float64)


def _douglas_peucker(x: np.ndarray, y: np.ndarray, tol: float) -> list[int]:
    keep = {0, len(x) - 1}
    stack = [(0, len(x) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (x[a + 1 : b] - x[a]) / (x[b] - x[a])
        d = np.abs(y[a + 1 : b] - (y[a] + t * (y[b] - y[a])))
        i = int(np.argmax(d))
        if d[i] > tol:
            j = a + 1 + i
            keep.add(j)
            stack.extend([(a, j), (j, b)])
    return sorted(keep)


def _design(sb: np.ndarray, pvi: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Columns [z0, g_0 .. g_n]: the profile is linear in them once PVIs and curve lengths are fixed."""
    n = len(pvi)
    knots = np.concatenate([[0.0], pvi, [np.inf]])
    a = np.zeros((len(sb), n + 2))
    a[:, 0] = 1.0
    for j in range(n + 1):
        a[:, 1 + j] = np.clip(sb - knots[j], 0.0, knots[j + 1] - knots[j])
    for k in range(n):
        t = sb - (pvi[k] - 0.5 * lengths[k])
        inside = (t > 0.0) & (t < lengths[k])
        m = np.minimum(t, lengths[k] - t)
        q = np.where(inside, m * m / (2.0 * lengths[k]), 0.0)
        a[:, 1 + k + 1] += q
        a[:, 1 + k] -= q
    return a


def _inner(sb, zb, wb, pvi, lengths, rules, road_class, scale):
    """Robust (soft L1 by IRLS) linear solve for z0 and grades, with grade rules as soft bounds."""
    a = _design(sb, pvi, lengths)
    n_g = len(pvi) + 1
    w = wb.copy()
    high = rules.get("grade", road_class).high if rules is not None else np.inf
    x = np.zeros(n_g + 1)
    for _ in range(IRLS_STEPS):
        aw = a * w[:, None]
        lhs = a.T @ aw
        rhs = aw.T @ zb
        # FR-AF-4: a grade beyond the rule range is pulled toward it, softly.
        for j in range(n_g):
            g = x[1 + j]
            if abs(g) > high:
                lhs[1 + j, 1 + j] += scale
                rhs[1 + j] += scale * math.copysign(high, g)
        x = np.linalg.solve(lhs + 1e-9 * np.eye(n_g + 1), rhs)
        r = a @ x - zb
        w = wb / np.sqrt(1.0 + (r / F_SCALE_M) ** 2)
    r = a @ x - zb
    return x, r


def _robust(r: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """Pseudo-residuals whose squares sum to the soft L1 cost."""
    rho = 2.0 * (np.sqrt(1.0 + (r / F_SCALE_M) ** 2) - 1.0)
    return wb * F_SCALE_M * np.sign(r) * np.sqrt(rho)


def _outer_residuals(p, n, sb, zb, wb, length, rules, road_class, scale):
    pvi = p[:n]
    lengths = np.exp(np.clip(p[n:], 0.0, 9.0))
    x, r = _inner(sb, zb, wb, pvi, lengths, rules, road_class, scale)
    extra = []
    order_pen = 5.0 * F_SCALE_M * math.sqrt(len(sb))
    for k in range(n):
        if k > 0:
            gap = (pvi[k] - 0.5 * lengths[k]) - (pvi[k - 1] + 0.5 * lengths[k - 1])
            extra.append(max(0.0, -gap) * order_pen)
        extra.append(max(0.0, -pvi[k]) * order_pen + max(0.0, pvi[k] - length) * order_pen)
        if rules is not None:
            dg = float(x[2 + k] - x[1 + k])
            if abs(dg) > 1e-6:
                kval = lengths[k] / (100.0 * abs(dg))
                rid = "vcurve.k_crest" if dg < 0 else "vcurve.k_sag"
                extra.append(RULE_WEIGHT * scale * math.sqrt(bound_penalty(rules, rid, kval, road_class)))
            else:
                extra.append(0.0)
    return np.concatenate([_robust(r, wb), np.array(extra)])


def _fit(v: _Vert, sb, zb, wb, length, rules, road_class) -> tuple[_Vert, float]:
    n = len(v.pvi)
    scale = RULE_WEIGHT * math.sqrt(max(len(sb), 1)) * F_SCALE_M
    p = np.concatenate([v.pvi, np.log(np.maximum(v.lengths, 1.0))])
    if n:
        sol = least_squares(
            _outer_residuals,
            p,
            args=(n, sb, zb, wb, length, rules, road_class, scale),
            diff_step=1e-4,
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=60 * (n + 1),
        )
        p = sol.x
    pvi = p[:n].copy()
    lengths = np.exp(np.clip(p[n:], 0.0, 9.0))
    x, r = _inner(sb, zb, wb, pvi, lengths, rules, road_class, scale)
    cost = float(0.5 * np.sum(_robust(r, wb) ** 2))
    return _Vert(float(x[0]), x[1:].copy(), pvi, lengths), cost


def _init(sb: np.ndarray, zb: np.ndarray, pvi_idx: list[int]) -> _Vert:
    xs = sb[pvi_idx]
    ys = zb[pvi_idx]
    grades = np.diff(ys) / np.diff(xs)
    pvi = xs[1:-1].copy()
    gaps = np.diff(xs)
    lengths = np.array([max(MIN_CURVE_M, 0.5 * min(gaps[k], gaps[k + 1])) for k in range(len(pvi))])
    z0 = ys[0] - grades[0] * xs[0]
    return _Vert(float(z0), grades, pvi, lengths)


def _observed_fraction(a: float, b: float, sb: np.ndarray) -> float:
    if b <= a:
        return 0.0
    inside = sb[(sb >= a) & (sb <= b)]
    return min(1.0, len(inside) * BIN_M / (b - a))


def fit_profile(
    s: np.ndarray,
    z: np.ndarray,
    length: float,
    *,
    rules: RuleBook | None = None,
    road_class: str = "*",
) -> tuple[_Vert, ProfileFit, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit the vertical chain to point stations ``s`` and corrected heights ``z``."""
    sb, zb, cnt = _bins(np.asarray(s), np.asarray(z), length)
    if len(sb) < 4:
        raise ValueError("too few profile samples")
    wb = np.sqrt(np.minimum(cnt, 25.0)) / 5.0
    # Robust pre-smoothing for the PVI search only.
    zs = _running_median(sb, zb, 9.0)
    tol = DP_TOL_M
    idx = _douglas_peucker(sb, zs, tol)
    while len(idx) - 2 > max(1.0, length * MAX_PVI_PER_M):
        tol *= 1.5
        idx = _douglas_peucker(sb, zs, tol)
    v = _init(sb, zs, idx)
    v, cost = _fit(v, sb, zb, wb, length, rules, road_class)
    # Backward simplification (fewest elements): merge two neighbouring curves into
    # one, or drop a PVI, whenever the cost barely rises. Curves lying mostly in an
    # unobserved gap are dropped regardless (FR-OC-1).
    wsum = float(np.sum(wb**2))
    changed = True
    while changed and len(v.pvi):
        changed = False
        n = len(v.pvi)
        dg = np.abs(np.diff(v.grades))
        obs = [_observed_fraction(p - 0.5 * L, p + 0.5 * L, sb) for p, L in zip(v.pvi, v.lengths, strict=True)]
        invented = [k for k in range(n) if obs[k] < MIN_CURVE_OBSERVED]
        if invented:
            v, cost = _fit(_remove(v, invented[0]), sb, zb, wb, length, rules, road_class)
            changed = True
            continue
        moves = [("merge", k) for k in sorted(range(n - 1), key=lambda i: (dg[i] + dg[i + 1], i))]
        moves += [("remove", k) for k in sorted(range(n), key=lambda i: (dg[i], i))]
        for op, k in moves[:PRUNE_TRIES]:
            trial = _merge(v, k) if op == "merge" else _remove(v, k)
            trial, tcost = _fit(trial, sb, zb, wb, length, rules, road_class)
            rms_rise = math.sqrt(2.0 * tcost / wsum) - math.sqrt(2.0 * cost / wsum)
            if tcost <= cost * PRUNE_COST_RATIO + 1e-9 or rms_rise <= PRUNE_RMS_RISE_M:
                v, cost, changed = trial, tcost, True
                break
    v = _sanitize(v, length)
    return v, _report(v, sb, zb, s, z, length), (sb, zb, cnt)


def _remove(v: _Vert, k: int) -> _Vert:
    grades = np.delete(v.grades, k + 1)
    # The merged tangent takes the mean grade of the two it replaces.
    grades[k] = 0.5 * (v.grades[k] + v.grades[k + 1])
    pvi = np.delete(v.pvi, k)
    lengths = np.delete(v.lengths, k)
    z0 = v.z0
    return _Vert(z0, grades, pvi, lengths)


def _merge(v: _Vert, k: int) -> _Vert:
    """Curves k and k+1 replaced by one curve at the intersection of their outer tangents."""
    knots = np.concatenate([[0.0], v.pvi])
    zk = v.z0 + np.concatenate([[0.0], np.cumsum(v.grades[:-1] * np.diff(knots))])
    z1, z2 = zk[k + 1], zk[k + 2]
    p1, p2 = v.pvi[k], v.pvi[k + 1]
    g_in, g_out = v.grades[k], v.grades[k + 2]
    start = p1 - 0.5 * v.lengths[k]
    end = p2 + 0.5 * v.lengths[k + 1]
    if abs(g_in - g_out) > 1e-9:
        x = (z2 - z1 + g_in * p1 - g_out * p2) / (g_in - g_out)
    else:
        x = 0.5 * (p1 + p2)
    x = min(max(x, start + 0.5 * MIN_CURVE_M), end - 0.5 * MIN_CURVE_M)
    length = max(MIN_CURVE_M, 2.0 * min(x - start, end - x))
    grades = np.delete(v.grades, k + 1)
    pvi = np.concatenate([v.pvi[:k], [x], v.pvi[k + 2 :]])
    lengths = np.concatenate([v.lengths[:k], [length], v.lengths[k + 2 :]])
    return _Vert(v.z0, grades, pvi, lengths)


def _sanitize(v: _Vert, length: float) -> _Vert:
    """Order PVIs and shrink overlapping curves so the chain is well formed."""
    order = np.argsort(v.pvi, kind="stable")
    pvi, lengths = v.pvi[order], v.lengths[order].copy()
    grades = np.concatenate([[v.grades[0]], v.grades[1:][order]])
    for k in range(len(pvi)):
        lengths[k] = max(MIN_CURVE_M, lengths[k])
        if k > 0:
            room = pvi[k] - pvi[k - 1]
            total = lengths[k] + lengths[k - 1]
            if total > 2 * room:
                f = 2 * room / total
                lengths[k] *= f
                lengths[k - 1] *= f
    return _Vert(v.z0, grades, pvi, lengths)


def _running_median(x: np.ndarray, y: np.ndarray, half: float) -> np.ndarray:
    out = np.empty_like(y)
    lo = np.searchsorted(x, x - half, side="left")
    hi = np.searchsorted(x, x + half, side="right")
    for i in range(len(x)):
        out[i] = np.median(y[lo[i] : hi[i]])
    return out


def _report(v: _Vert, sb, zb, s, z, length) -> ProfileFit:
    res = np.asarray(z) - v.z(np.asarray(s))
    ok = np.abs(res) < 1.0
    r = res[ok]
    return ProfileFit(
        Profile(),
        float(np.sqrt(np.mean(r**2))) if r.size else math.nan,
        float(np.percentile(np.abs(r), 95.0)) if r.size else math.nan,
        int(r.size),
    )


def element_ranges(v: _Vert, length: float) -> list[tuple[float, float, str]]:
    """(s0, s1, kind) of the L2 elements, kind "tangent" or "curve"."""
    out = []
    cursor = 0.0
    for p, L in zip(v.pvi, v.lengths, strict=True):
        a, b = p - 0.5 * L, p + 0.5 * L
        a_c, b_c = max(a, 0.0), min(b, length)
        if a_c - cursor > 1e-6:
            out.append((cursor, a_c, "tangent"))
        if b_c - max(a_c, cursor) > 1e-6:
            out.append((max(a_c, cursor), b_c, "curve"))
        cursor = max(cursor, b_c)
    if length - cursor > 1e-6:
        out.append((cursor, length, "tangent"))
    return out


def to_profile(v: _Vert, length: float, provs: list[Provenance] | None = None) -> Profile:
    """L2 elements; each starts exactly at the end elevation and grade of the previous one.

    Stations are rounded as the canonical JSON rounds them (D-011) and each length
    is taken between rounded stations, so a model read back has no station gap at
    any joint (AC-H4).
    """
    elements: list[VerticalElement] = []
    for n, (a_raw, b_raw, kind) in enumerate(element_ranges(v, length)):
        a, b = round(a_raw, FLOAT_DIGITS), round(b_raw, FLOAT_DIGITS)
        if elements:
            a = round(elements[-1].s0 + elements[-1].length, FLOAT_DIGITS)
        if b - a <= 0.0:
            continue
        prov = provs[n] if provs else Provenance(Source.MEASURED, 0.5)
        if kind == "curve":
            g0, g1 = _curve_grade(v, a), _curve_grade(v, b)
        else:
            g0 = g1 = float(v.grade(np.array([0.5 * (a + b)]))[0])
        z0 = float(v.z(np.array([a]))[0])
        if elements:
            prev = elements[-1]
            z0 = prev.z(prev.length)
            g_end = prev.grade(prev.length)
            if kind == "curve":
                g0 = g_end
            else:
                g0 = g1 = g_end
        elements.append(VerticalElement(a, round(b - a, FLOAT_DIGITS), z0, g0, g1, prov))
    return Profile(elements)


def _curve_grade(v: _Vert, s: float) -> float:
    for k, (p, L) in enumerate(zip(v.pvi, v.lengths, strict=True)):
        a, b = p - 0.5 * L, p + 0.5 * L
        if a - 1e-9 <= s <= b + 1e-9:
            t = min(max(s - a, 0.0), L)
            return float(v.grades[k] + (v.grades[k + 1] - v.grades[k]) * t / L)
    return float(v.grade(np.array([s]))[0])
