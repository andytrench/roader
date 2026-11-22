"""L1 horizontal alignment fitting (FR-AF-1, FR-AF-4, FR-AF-5).

The alignment is a chain whose curvature is continuous and piecewise linear in
station, so every piece is a line (zero curvature), an arc (constant) or a
spiral (linear). Position and heading are continuous by construction and so is
curvature, which the spec requires on freeway mainlines. The fewest pieces that
keep the deviation from the registered centerline under ``MAX_DEV_M`` are
chosen; near-zero and near-constant curvatures are then snapped to exact lines
and arcs when the snapped chain still meets the deviation limit.

Minimum radius is a soft prior (``curve.radius``, FR-AF-4); a fitted radius
outside the rule's range is kept and flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from roader.conflate.polyline import Polyline
from roader.fit.priors import curvature_penalty
from roader.model.geometry import AlignElement, Alignment
from roader.model.provenance import Provenance, Source
from roader.model.serialize import FLOAT_DIGITS
from roader.rules import RuleBook

MAX_DEV_M = 0.25  # FR-AF-1
GRID_M = 1.0
ALONG_WEIGHT = 0.1
LINE_K = 2e-5  # |curvature| below this (radius over 50 km) is a line
ARC_DK = 2e-5  # curvature change across a piece below this is an arc
MIN_PIECE_M = 15.0
RULE_WEIGHT = 0.5
PROJECTION_PASSES = 2
END_WEIGHT = 10.0
ANCHOR_WEIGHT = 100.0
PIN_HEADING_M = 20.0  # a heading error at a pinned end weighs as its lateral offset this far along
KAPPA_WINDOW_M = 10.0
SUPPORT_M = 10.0  # FR-AF-1 deviation is checked where measured centerline samples are this close


@dataclass
class AlignmentFit:
    alignment: Alignment
    max_dev_m: float
    rms_m: float
    n_pieces: int
    flags: list[str] = field(default_factory=list)


@dataclass
class _Chain:
    x0: float
    y0: float
    h0: float
    knots: np.ndarray  # stations, 0 .. L
    kappa: np.ndarray  # curvature at knots

    def sample(self, step: float = GRID_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = float(self.knots[-1])
        n = max(2, math.ceil(length / step) + 1)
        s = np.union1d(np.linspace(0.0, length, n), self.knots)
        k = np.interp(s, self.knots, self.kappa)
        ds = np.diff(s)
        h = self.h0 + np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
        # Simpson on each interval with the exact quadratic heading midpoint.
        hm = 0.5 * (h[1:] + h[:-1]) - (k[1:] - k[:-1]) * ds / 8.0
        dx = ds / 6.0 * (np.cos(h[:-1]) + 4.0 * np.cos(hm) + np.cos(h[1:]))
        dy = ds / 6.0 * (np.sin(h[:-1]) + 4.0 * np.sin(hm) + np.sin(h[1:]))
        x = self.x0 + np.concatenate([[0.0], np.cumsum(dx)])
        y = self.y0 + np.concatenate([[0.0], np.cumsum(dy)])
        return s, np.column_stack([x, y]), h


def _pieces_from(params: np.ndarray, m: int, length: float, ties: dict) -> _Chain:
    x0, y0, h0 = params[:3]
    # Total length is free within +-5%: with both ends anchored, a fixed length
    # would force lateral bulges to absorb millimetres of length error.
    total = length * math.exp(float(np.clip(params[3], -0.05, 0.05)))
    a = np.clip(params[4 : 4 + m - 1], -8.0, 8.0)
    w = np.exp(np.concatenate([[0.0], a]))
    knots = np.concatenate([[0.0], np.cumsum(w / w.sum() * total)])
    knots[-1] = total
    free = params[4 + m - 1 :]
    kappa = np.empty(m + 1)
    j = 0
    for i in range(m + 1):
        tie = ties.get(i)
        if tie == "zero":
            kappa[i] = 0.0
        elif isinstance(tie, int):
            kappa[i] = np.nan  # filled below from the knot it is tied to
        else:
            kappa[i] = free[j]
            j += 1
    for i in range(m + 1):
        tie = ties.get(i)
        if isinstance(tie, int) and not isinstance(tie, bool):
            kappa[i] = kappa[tie]
    return _Chain(float(x0), float(y0), float(h0), knots, kappa)


def _n_free(m: int, ties: dict) -> int:
    return sum(1 for i in range(m + 1) if i not in ties)


def _residuals(params, m, length, ties, s_data, xy_data, w_data, rules, road_class, ends):
    chain = _pieces_from(params, m, length, ties)
    s, pts, h = chain.sample()
    sd = np.clip(s_data, 0.0, chain.knots[-1])
    px = np.interp(sd, s, pts[:, 0])
    py = np.interp(sd, s, pts[:, 1])
    hh = np.interp(sd, s, h)
    dx, dy = xy_data[:, 0] - px, xy_data[:, 1] - py
    lat = -dx * np.sin(hh) + dy * np.cos(hh)
    along = dx * np.cos(hh) + dy * np.sin(hh)
    res = [w_data * lat, ALONG_WEIGHT * w_data * along]
    # Stations are anchored at both ends of the registered line, so the chain cannot slide along itself.
    # An end pinned to a junction node (FR-AF-3) is held in both directions and in heading.
    for (ex, ey), (px_, py_), hh_, pin in ((ends[0], pts[0], h[0], ends[2][0]), (ends[1], pts[-1], h[-1], ends[2][1])):
        res.append(np.array([END_WEIGHT * ((ex - px_) * math.cos(hh_) + (ey - py_) * math.sin(hh_))]))
        if pin is not None:
            res.append(np.array([ANCHOR_WEIGHT * (-(ex - px_) * math.sin(hh_) + (ey - py_) * math.cos(hh_))]))
            if pin[2] is not None:
                res.append(np.array([ANCHOR_WEIGHT * PIN_HEADING_M * math.remainder(hh_ - pin[2], 2.0 * math.pi)]))
    if rules is not None:
        pen = [math.sqrt(curvature_penalty(rules, float(k), road_class)) for k in chain.kappa]
        res.append(RULE_WEIGHT * math.sqrt(max(len(s_data), 1)) * np.array(pen))
    # Keep pieces from collapsing.
    lengths = np.diff(chain.knots)
    res.append(np.maximum(0.0, MIN_PIECE_M - lengths) * 10.0)
    return np.concatenate(res)


def _fit(m, length, ties, init, s_data, xy_data, w_data, rules, road_class, ends):
    sol = least_squares(
        _residuals,
        init,
        args=(m, length, ties, s_data, xy_data, w_data, rules, road_class, ends),
        loss="soft_l1",
        f_scale=0.1,
        x_scale="jac",
        max_nfev=400,
    )
    return sol.x


def _curvature_signal(check_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Stations, curvature and initial heading of the registered centerline (local linear fit of heading)."""
    line = Polyline(check_xy, step=0.5)
    n = max(3, int(line.length / 2.0) + 1)
    s = np.linspace(0.0, line.length, n)
    h = np.unwrap(line.heading(s))
    k = np.zeros(n)
    for i in range(n):
        win = np.abs(s - s[i]) <= KAPPA_WINDOW_M
        if win.sum() >= 3:
            k[i] = np.polyfit(s[win], h[win], 1)[0]
    return s, k, float(h[0])


def _segment(s: np.ndarray, k: np.ndarray, m: int) -> np.ndarray:
    """m-piece knots by top-down splitting of the curvature signal at the worst interpolation error."""
    knots = [0, len(s) - 1]
    while len(knots) < m + 1:
        idx = sorted(knots)
        interp = np.interp(s, s[idx], k[idx])
        err = np.abs(k - interp)
        for j in idx:
            lo, hi = max(0, j - int(MIN_PIECE_M / 2.0)), j + int(MIN_PIECE_M / 2.0) + 1
            err[lo:hi] = -1.0
        best = int(np.argmax(err))
        if err[best] <= 0:
            break
        knots.append(best)
    return np.array(sorted(knots))


def _initial(m: int, length: float, check_xy: np.ndarray, ties: dict) -> np.ndarray:
    s, k, h0 = _curvature_signal(check_xy)
    idx = _segment(s, k, m)
    m_eff = len(idx) - 1
    knots = s[idx] * (length / s[-1])
    lengths = np.diff(knots)
    a = np.log(np.maximum(lengths, 1e-6) / max(lengths[0], 1e-6))[1:]
    a = np.concatenate([a, np.zeros(m - m_eff)])
    kap = list(k[idx]) + [k[idx][-1]] * (m - m_eff)
    free = [kap[i] for i in range(m + 1) if i not in ties]
    return np.concatenate([[check_xy[0, 0], check_xy[0, 1], h0, 0.0], a, free])


def _params_with_ties(chain: _Chain, m: int, ties: dict, length: float) -> np.ndarray:
    lengths = np.diff(chain.knots)
    a = np.log(np.maximum(lengths, 1e-6) / max(lengths[0], 1e-6))[1:]
    free = [chain.kappa[i] for i in range(m + 1) if i not in ties]
    return np.concatenate([[chain.x0, chain.y0, chain.h0, math.log(chain.knots[-1] / length)], a, free])


def _deviation(chain: _Chain, check_xy: np.ndarray) -> tuple[float, float]:
    """Max and RMS lateral distance of the supported registered centerline from the chain."""
    _s, pts, _h = chain.sample(0.5)
    line = Polyline(pts, step=0.5)
    _s, u, _ok = line.project(check_xy)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return math.inf, math.inf
    return float(np.max(np.abs(u))), float(np.sqrt(np.mean(u**2)))


def _reproject(chain: _Chain, xy: np.ndarray) -> np.ndarray:
    _s, pts, _h = chain.sample(0.5)
    line = Polyline(pts, step=0.5)
    sp, _u, _ok = line.project(xy)
    # Polyline stations are chord sums of the sampled chain; at 0.5 m they equal arclength to 1e-6.
    return np.nan_to_num(sp, nan=0.0)


def fit_alignment(
    xy: np.ndarray,
    weights: np.ndarray,
    check_xy: np.ndarray,
    *,
    rules: RuleBook | None = None,
    road_class: str = "*",
    max_pieces: int | None = None,
    max_dev: float = MAX_DEV_M,
    pins: tuple = (None, None),
) -> tuple[_Chain, AlignmentFit]:
    """Fit the chain to centerline samples ``xy`` (ordered along travel) with ``weights``.

    ``check_xy`` is the registered centerline the FR-AF-1 deviation limit applies to.
    ``pins``: per end, None or (x, y, heading or None) of a junction node on another
    road's reference line (FR-AF-3); the chain's end is held on the first / last
    ``check_xy`` point, which must be (x, y), and on that heading when given.
    Returns the chain and a fit report; ``to_alignment`` turns the chain into L1 elements.
    """
    xy = np.asarray(xy, dtype=np.float64)
    seg = np.hypot(*np.diff(check_xy, axis=0).T)
    length = float(seg.sum())
    ref = Polyline(check_xy, step=0.5)
    s_data, _u, _ok = ref.project(xy)
    s_data = np.nan_to_num(s_data, nan=0.0)
    s_check = np.concatenate([[0.0], np.cumsum(seg)])
    supported = np.zeros(len(check_xy), dtype=bool)
    if len(s_data):
        sd_sorted = np.sort(s_data)
        pos = np.clip(np.searchsorted(sd_sorted, s_check), 1, len(sd_sorted) - 1) if len(sd_sorted) > 1 else np.zeros(len(s_check), int)
        near = np.minimum(np.abs(s_check - sd_sorted[pos - 1 if len(sd_sorted) > 1 else 0]), np.abs(s_check - sd_sorted[pos]))
        supported = near <= SUPPORT_M
    check_sup = check_xy[supported] if supported.sum() >= 2 else check_xy
    w = np.asarray(weights, dtype=np.float64)
    max_pieces = max_pieces or max(2, min(12, math.ceil(length / 60.0)))
    ends = (check_xy[0], check_xy[-1], tuple(pins))
    best = None
    for m in range(1, max_pieces + 1):
        ties: dict = {}
        params = _initial(m, length, check_xy, ties)
        s_fit = s_data.copy()
        for _ in range(PROJECTION_PASSES):
            params = _fit(m, length, ties, params, s_fit, xy, w, rules, road_class, ends)
            chain = _pieces_from(params, m, length, ties)
            s_fit = _reproject(chain, xy)
        dev, rms = _deviation(chain, check_sup)
        snapped = _snap(chain, m, length, s_fit, xy, w, rules, road_class, check_sup, max_dev, ends)
        if snapped is not None:
            chain, dev, rms = snapped
        if best is None or dev < best[1]:
            best = (chain, dev, rms, m)
        if dev <= max_dev:
            break
    chain, dev, rms, m = best
    flags = [] if dev <= max_dev else [f"FR-AF-1:deviation {dev:.2f} m"]
    return chain, AlignmentFit(Alignment(), dev, rms, m, flags)


def _snap(chain, m, length, s_fit, xy, w, rules, road_class, check_xy, max_dev, ends):
    k = chain.kappa
    ties: dict = {}
    for i in range(m + 1):
        if abs(k[i]) < LINE_K:
            ties[i] = "zero"
    for i in range(m):
        if i in ties or (i + 1) in ties:
            continue
        if abs(k[i + 1] - k[i]) < ARC_DK and k[i] * k[i + 1] > 0:
            ties[i + 1] = i
    if not ties:
        return None
    params = _params_with_ties(chain, m, ties, length)
    params = _fit(m, length, ties, params, s_fit, xy, w, rules, road_class, ends)
    snapped = _pieces_from(params, m, length, ties)
    dev, rms = _deviation(snapped, check_xy)
    if dev <= max_dev:
        return snapped, dev, rms
    return None


def to_alignment(chain: _Chain, provs: list[Provenance] | None = None) -> Alignment:
    """L1 elements from the chain; consecutive lines, and arcs of equal curvature, are merged."""
    pieces = []
    for i in range(len(chain.knots) - 1):
        k0, k1 = float(chain.kappa[i]), float(chain.kappa[i + 1])
        s0, s1 = float(chain.knots[i]), float(chain.knots[i + 1])
        if k0 == 0.0 and k1 == 0.0:
            kind = "line"
        elif k0 == k1:
            kind = "arc"
        else:
            kind = "spiral"
        if pieces and pieces[-1][0] == kind and kind in ("line", "arc") and pieces[-1][3] == k0:
            pieces[-1] = (kind, pieces[-1][1], s1, k0, k1)
        else:
            pieces.append((kind, s0, s1, k0, k1))
    # Every value is rounded as the canonical JSON will round it (D-011, FLOAT_DIGITS), and
    # each element starts at the end of its rounded predecessor, so reading the model
    # back moves no joint (AC-H4, AC-H5).
    elements = []
    x, y, h = chain.x0, chain.y0, chain.h0
    for n, (kind, s0, s1, k0, k1) in enumerate(pieces):
        prov = provs[n] if provs else Provenance(Source.MEASURED, 0.5)
        s0r, s1r = _r(s0), _r(s1)
        el = AlignElement(kind, s0r, _r(s1r - s0r), _r(x), _r(y), _r(h), _r(k0), _r(k1), prov)
        elements.append(el)
        x, y, h, _k = el.end_state()
    return Alignment(elements)


def _r(v: float) -> float:
    out = round(float(v), FLOAT_DIGITS)
    return 0.0 if out == 0.0 else out


def element_ranges(chain: _Chain) -> list[tuple[float, float]]:
    """Station ranges of the elements ``to_alignment`` will produce."""
    al = to_alignment(chain)
    return [(e.s0, e.s0 + e.length) for e in al.elements]
