"""AC-A3, D-103: on the facing side of D-100 twins the AC-A3 reference edge search stops at the fit's cap."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.fit.median import cap_at, reference_caps, split_twin_medians
from tests.fit import scenes


def _fit(road):
    return SimpleNamespace(road=road, line=Polyline.from_alignment(road.alignment), flags=[])


def test_ac_a3_reference_on_twin_facing_sides_stops_at_the_fits_cap():
    roads = scenes.twin_carriageways(median=3.0, dz=0.0)
    for r in roads:
        r.cross_section.width_left = scenes.piecewise([(0.0, 9.0)])
    fits = [_fit(r) for r in roads]
    split_twin_medians(fits, np.zeros((0, 3)))
    caps = reference_caps(roads, np.zeros((0, 3)))
    for r in roads:
        cap = cap_at(caps[r.id].get(1, []), np.array([300.0]))[0]
        assert cap == pytest.approx(r.cross_section.width_left(300.0), abs=0.02)


def test_ac_a3_reference_twins_short_of_the_median_have_no_cap():
    roads = scenes.twin_carriageways(median=3.0, dz=0.0)
    caps = reference_caps(roads, np.zeros((0, 3)))
    assert all(np.isinf(cap_at(caps.get(r.id, {}).get(1, []), np.array([300.0]))[0]) for r in roads)


def test_ac_a3_reference_edges_clamp_at_the_cap_before_the_neighbour_limits():
    from roader.pipeline import _clamp_edges

    st = np.array([0.0, 10.0, 20.0])
    left = np.array([5.0, 7.0, np.nan])
    right = np.array([-4.0, -9.0, -3.0])
    caps = {1: [(st, np.array([6.0, 6.0, 6.0]))], -1: [(st, np.array([8.0, 8.0, 8.0]))]}
    lo, ro = _clamp_edges(st, left, right, caps, lambda s: (np.full(len(s), 6.5), np.full(len(s), 8.5)))
    assert np.allclose(lo[:2], [5.0, 6.0]) and np.isnan(lo[2])
    assert np.allclose(ro, [-4.0, -8.0, -3.0])
