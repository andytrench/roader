"""FR-VC-3: registration with measured centers on a short stretch of a long line (Dallas stack).

Far from every measured center the kernel smoothers have no weight left; the
registered line there must fall back to the shifted input line, never NaN.
"""

from __future__ import annotations

import numpy as np

from roader.conflate.polyline import Polyline
from roader.conflate.register import Profiles, register_line


def _profiles(s, left, right):
    n = len(s)
    nan = np.full(n, np.nan)
    return Profiles(s, left, right, np.zeros(n), nan, nan, np.full(n, 50), 30.0)


def test_fr_vc_3_far_from_measured_centers_the_line_stays_finite():
    s = np.arange(0.0, 3000.0, 5.0)
    line = Polyline(np.column_stack([s, np.zeros_like(s)]))
    left = np.where((s > 100.0) & (s < 160.0), 6.0, np.nan)
    right = np.where((s > 100.0) & (s < 160.0), -4.0, np.nan)
    reg, _st, _mid, _w = register_line(line, _profiles(s, left, right))
    assert np.all(np.isfinite(reg.xy))
    _s, u, ok = line.project(reg.xy)
    assert np.allclose(u[ok], 1.0, atol=0.05)


def test_fr_vc_3_two_far_apart_measured_stretches_stay_finite():
    s = np.arange(0.0, 4000.0, 5.0)
    line = Polyline(np.column_stack([s, np.zeros_like(s)]))
    inside = ((s > 100.0) & (s < 160.0)) | ((s > 3800.0) & (s < 3860.0))
    left = np.where(inside, 6.0, np.nan)
    right = np.where(inside, -4.0, np.nan)
    reg, *_ = register_line(line, _profiles(s, left, right))
    assert np.all(np.isfinite(reg.xy))
