"""FR-AF-1, FR-AF-3, D-051: a ramp leaving its parent at an angle from a pinned node.

The child's start is pinned to the parent's line and heading (D-051) while its
registered line leaves at 11 degrees and, over its first 150 m, is only a weak
vector hint (D-064). The chain must turn onto that line after the pin, not run
on along the parent (Woburn ramp 9522279 ran 4 to 6 m off its pavement).
"""

from __future__ import annotations

import math

import numpy as np

from roader.conflate.polyline import Polyline
from roader.fit import alignment as fa
from roader.rules import RuleBook

RULES = RuleBook.load()
ANGLE = math.radians(11.0)


def _scene():
    s = np.arange(0.0, 400.0, 0.5)
    check = np.column_stack([s * math.cos(ANGLE), s * math.sin(ANGLE)])
    hint_s = np.arange(0.0, 150.0, 10.0)
    meas_s = np.arange(150.0, 400.0, 5.0)
    xy = np.vstack([check[(hint_s / 0.5).astype(int)], check[(meas_s / 0.5).astype(int)]])
    w = np.concatenate([np.full(len(hint_s), 0.3), np.ones(len(meas_s))])
    pins = ((0.0, 0.0, 0.0, 0.0), None)
    return xy, w, check, pins


def test_fr_af_1_chain_leaves_a_pinned_end_onto_the_line():
    xy, w, check, pins = _scene()
    chain, rep = fa.fit_alignment(xy, w, check, rules=RULES, road_class="motorway_link", pins=pins)
    al = fa.to_alignment(chain)
    x0, y0, h0, _k = al.evaluate(0.0)
    assert math.hypot(x0, y0) < 0.05 and abs(h0) < 1e-3
    line = Polyline.from_alignment(al)
    s, u, ok = line.project(check)
    beyond = ok & (s > fa.PIN_ZONE_M)
    assert np.max(np.abs(u[beyond])) < 1.0, rep


def test_fr_af_1_pinned_zone_does_not_force_extra_pieces():
    """A straight line pinned on its own heading needs one piece."""
    s = np.arange(0.0, 300.0, 0.5)
    check = np.column_stack([s, np.zeros_like(s)])
    xy = check[::10]
    _chain, rep = fa.fit_alignment(xy, np.ones(len(xy)), check, rules=RULES, road_class="motorway_link", pins=((0.0, 0.0, 0.0, 0.0), None))
    assert rep.n_pieces == 1 and rep.max_dev_m < 0.05
