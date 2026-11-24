"""FR-XS-1, D-063: edge lines across thinly measured stretches (Woburn I-93 northbound).

Along a merge or gore one side's edge is measured at a few isolated stations; a
width knot resting on one of them swings far beyond any measured value. Such a
stretch is bridged by a taper between well-supported knots instead.
"""

from __future__ import annotations

import numpy as np

from roader.conflate.register import Profiles
from roader.fit.section import fit_edges


def _profiles(left: np.ndarray, right: np.ndarray, s: np.ndarray) -> Profiles:
    n = len(s)
    nan = np.full(n, np.nan)
    return Profiles(s, left, -right, np.zeros(n), nan, nan, np.full(n, 50), 30.0)


def test_fr_xs_1_isolated_edges_do_not_set_a_width_knot():
    """Left edge at 14 m up to s = 560, one edge at 8.5 m at s = 625, then 10.5 m from s = 770."""
    s = np.arange(0.0, 1000.0, 5.0)
    rng = np.random.default_rng(3)
    left = np.where(s <= 560.0, 14.0, np.where(s >= 770.0, 10.5, np.nan)) + rng.normal(0.0, 0.03, len(s))
    left[s == 625.0] = 8.5
    right = np.full(len(s), 5.5) + rng.normal(0.0, 0.03, len(s))
    fit = fit_edges(_profiles(left, right, s), 1000.0, [])
    assert fit is not None
    width = np.interp(s, fit.knots_left, fit.left)
    # Never far beyond the measured values on either side of the thin stretch.
    assert np.all(width > 8.0) and np.all(width < 14.5), (fit.knots_left, fit.left)
    assert np.max(np.abs(np.interp(s, fit.knots_right, fit.right) - 5.5)) < 0.2


def test_d155_a_step_between_two_knots_keeps_each_knot_within_the_measured_widths():
    """Bellevue I-405 SB: left edge 11.5 m, a gap, 1.83 m over s = 250-280, then 10 m from s = 300.

    The knot at 250 took -0.2 m to fit the narrow stretch against the wide one beyond the next
    knot; the left edge then crossed the centre line and the road could not be swept.
    """
    s = np.arange(0.0, 500.0, 5.0)
    left = np.where(s <= 165.0, 11.5, np.where((s >= 250.0) & (s <= 280.0), 1.83, np.where(s >= 300.0, 10.0, np.nan)))
    right = np.full(len(s), 9.9)
    fit = fit_edges(_profiles(left, right, s), 500.0, [(195.0, 235.0)])
    assert fit is not None
    measured = left[np.isfinite(left)]
    assert fit.left.min() >= measured.min() - 1e-9 and fit.left.max() <= measured.max() + 1e-9, (fit.knots_left, fit.left)


def test_fr_xs_1_dense_width_change_is_followed():
    """A measured taper from 7.0 to 10.0 m over 100 m stays in the edge line."""
    s = np.arange(0.0, 600.0, 5.0)
    left = np.interp(s, [0.0, 250.0, 350.0, 600.0], [7.0, 7.0, 10.0, 10.0])
    right = np.full(len(s), 5.5)
    fit = fit_edges(_profiles(left, right, s), 600.0, [])
    width = np.interp(s, fit.knots_left, fit.left)
    assert np.max(np.abs(width - left)) < 0.5
