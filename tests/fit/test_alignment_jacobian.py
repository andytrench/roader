"""FR-AF-1 (D-102): the batched alignment Jacobian is scipy's 2-point Jacobian, bit for bit.

Batching must not change the fitted model: every batched row is the single-chain
residual exactly, and the Jacobian equals scipy's finite differences exactly, so
least_squares follows the same path to the same optimum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from roader.fit import alignment as fa


def _problem(ties):
    m, length = 5, 800.0
    rng = np.random.default_rng(3)
    p = np.concatenate([[10.0, -5.0, 0.3, 0.01], rng.normal(0, 0.3, m - 1), rng.normal(0, 0.002, m + 1 - len(ties))])
    s = np.linspace(0, length, 300)
    xy = np.column_stack([s * 0.95, 40 * np.sin(s / 300)])
    ends = ((0.0, 0.0), (760.0, 30.0), (None, (760.0, 30.0, 0.2, 0.0)), 0.2, 0.004)
    return p, (m, length, ties, s, xy, np.ones(len(s)), None, "motorway", ends)


def test_fr_af_1_batched_rows_equal_single_residuals():
    p, args = _problem({})
    rng = np.random.default_rng(5)
    batch = p[None, :] + rng.normal(0, 1e-3, (6, len(p)))
    m, length, ties, s, xy, w, rules, _rc, ends = args
    rows = fa._residuals_batch(batch, m, length, ties, s, xy, w, rules, ends)
    for b in range(len(batch)):
        assert np.array_equal(rows[b], fa._residuals(batch[b], *args))


def test_fr_af_1_jacobian_is_scipy_two_point_exactly():
    for ties in ({}, {0: "zero", 2: 1}):
        p, args = _problem(ties)
        f0 = fa._residuals(p, *args)
        expected = approx_derivative(fa._residuals, p, method="2-point", f0=f0, args=args)
        assert np.array_equal(fa._jacobian(p, *args), expected)
