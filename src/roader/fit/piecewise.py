"""Robust piecewise-linear fits along a road (FR-XS-1, FR-XS-2, AC-P2).

Edge lines and cross slopes are fitted along the road, not station by station:
values at knots joined linearly, solved by iteratively reweighted least squares
(Huber), then thinned to the fewest knots that stay within a tolerance. Knots
are never placed inside an unobserved gap, so a gap is bridged by a straight
taper between observed values (FR-OC-1).
"""

from __future__ import annotations

import numpy as np

HUBER_K = 1.5
IRLS_STEPS = 6


def hat_basis(s: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """(N, K) linear interpolation weights; constant beyond the end knots."""
    s = np.clip(np.asarray(s, dtype=np.float64), knots[0], knots[-1])
    basis = np.zeros((len(s), len(knots)))
    if len(knots) == 1:
        basis[:, 0] = 1.0
        return basis
    j = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(knots) - 2)
    t = (s - knots[j]) / (knots[j + 1] - knots[j])
    rows = np.arange(len(s))
    basis[rows, j] = 1.0 - t
    basis[rows, j + 1] = t
    return basis


def fit_piecewise(
    s: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    *,
    scale: np.ndarray | None = None,
    prior: float | None = None,
    prior_weight: float = 0.0,
    min_sigma: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Values at knots of y ~ scale * f(s), f piecewise linear. Returns (values, residuals)."""
    basis = hat_basis(s, knots)
    if scale is not None:
        basis = basis * np.asarray(scale)[:, None]
    w = np.ones(len(s))
    reg = np.eye(len(knots)) * prior_weight
    rhs_prior = np.full(len(knots), (prior or 0.0) * prior_weight)
    # A light smoothness term ties neighbouring knots where the data are thin.
    diff = np.diff(np.eye(len(knots)), axis=0) * 1e-3
    values = np.zeros(len(knots))
    for _ in range(IRLS_STEPS):
        aw = basis * w[:, None]
        lhs = basis.T @ aw + reg + diff.T @ diff
        rhs = aw.T @ y + rhs_prior
        values = np.linalg.solve(lhs + 1e-12 * np.eye(len(knots)), rhs)
        r = y - basis @ values
        sigma = max(1.4826 * float(np.median(np.abs(r))), min_sigma)
        a = np.abs(r) / (HUBER_K * sigma)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))
    return values, y - basis @ values


def knot_stations(length: float, spacing: float, gaps: list[tuple[float, float]] = ()) -> np.ndarray:
    """Evenly spaced knots over [0, length], none strictly inside a gap."""
    n = max(1, round(length / spacing))
    knots = np.linspace(0.0, length, n + 1)
    keep = np.ones(len(knots), dtype=bool)
    for a, b in gaps:
        keep &= ~((knots > a) & (knots < b))
    # No knot at a gap edge either: data there are thin and a knot would swing.
    return knots[keep]


def simplify(knots: np.ndarray, values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop knots while linear interpolation of the kept ones stays within tol of every original value."""
    keep = list(range(len(knots)))
    changed = True
    while changed and len(keep) > 2:
        changed = False
        best, best_err = None, np.inf
        for pos in range(1, len(keep) - 1):
            trial = keep[:pos] + keep[pos + 1 :]
            interp = np.interp(knots, knots[trial], values[trial])
            err = float(np.max(np.abs(interp - values)))
            if err <= tol and err < best_err:
                best, best_err = pos, err
        if best is not None:
            keep.pop(best)
            changed = True
    if len(keep) == 2 and abs(values[keep[0]] - values[keep[1]]) <= tol and np.max(np.abs(values - values.mean())) <= tol:
        return knots[:1], np.array([float(np.mean(values))])
    return knots[keep], values[keep]
