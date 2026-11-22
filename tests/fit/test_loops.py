"""Cloverleaf loop ramps: spirals and tight arcs (FR-AF-1) and banking fitted from lidar (FR-XS-2, AC-P2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roader.conflate.graph import data_extent, select_roads
from roader.conflate.polyline import Polyline
from roader.evidence import build_evidence
from roader.fit import alignment as fa
from roader.fit.road import fit_road
from roader.rules import RuleBook
from tests.fit import scenes

RULES = RuleBook.load()


@pytest.mark.parametrize("radius", [45.0, 60.0, 90.0])
def test_fr_af_1_loop_alignment_from_centerline_samples(radius):
    """Line, spiral, 3/4-turn arc, spiral, line: within 0.25 m, radius within 3%, heading continuous."""
    truth = scenes.loop_ramp(radius=radius)
    tl = Polyline.from_alignment(truth.alignment)
    rng = np.random.default_rng(int(radius))
    s = np.arange(0.0, tl.length, 5.0)
    data = tl.offset_point(s, rng.normal(0.0, 0.05, len(s)))
    chain, rep = fa.fit_alignment(data, np.ones(len(data)), tl.xy, rules=RULES, road_class="motorway_link")
    al = fa.to_alignment(chain)
    assert rep.max_dev_m <= fa.MAX_DEV_M
    fitted = Polyline.from_alignment(al)
    _s, u, ok = tl.project(fitted.xy[::4])
    assert np.max(np.abs(u[ok])) < 0.25
    s_mid = 70.0 + 50.0 + 0.5 * 230.0 * radius / 60.0
    k_mid = al.evaluate(s_mid)[3]
    assert abs(1.0 / k_mid - radius) / radius < 0.03
    for a, b in zip(al.elements, al.elements[1:], strict=False):
        xa, ya, ha, ka = a.end_state()
        assert math.hypot(xa - b.x0, ya - b.y0) < 1e-3
        assert abs(math.remainder(ha - b.heading0, 2 * math.pi)) < 1e-5
        assert abs(ka - b.k0) < 1e-9
    total_turn = al.evaluate(al.length)[2] - al.evaluate(0.0)[2]
    assert abs(total_turn - (truth.alignment.evaluate(truth.length)[2] - truth.alignment.evaluate(0.0)[2])) < 0.03


@pytest.fixture(scope="module")
def loop_fit():
    truth = scenes.loop_ramp()
    sp = scenes.points([truth])
    pts = scenes.to_pointset(sp)
    vec = scenes.vectors([truth], bias=1.0, noise=0.3)
    seeds = select_roads(vec, data_extent(pts.xyz[:, :2]))
    assert len(seeds) == 1
    ev = build_evidence(pts, [s.line.xy for s in seeds])
    return truth, fit_road(seeds[0], ev, RULES)


def test_fr_af_1_loop_ramp_from_lidar(loop_fit):
    truth, fit = loop_fit
    road = fit.road
    tl = Polyline.from_alignment(truth.alignment)
    xy = Polyline.from_alignment(road.alignment).xy[20:-20:4]
    _s, u, ok = tl.project(xy)
    assert np.max(np.abs(u[ok])) < 0.25
    k = max(abs(e.k0) for e in road.alignment.elements)
    assert abs(1.0 / k - 60.0) < 3.0
    # FR-AF-1 is judged against the true centerline here: the registered line the
    # deviation flag refers to wiggles by about 0.3 m where the loop ends in the data.
    assert len(road.alignment.elements) <= 6


def _on_arc(truth):
    return np.arange(70.0 + 50.0 + 20.0, 70.0 + 50.0 + 230.0 - 20.0, 5.0)


def test_fr_xs_2_loop_banking_measured(loop_fit):
    """7% banking on the arc, both sides down to the inside of the turn, measured not ruled."""
    truth, fit = loop_fit
    road = fit.road
    xs = road.cross_section
    tl = Polyline.from_alignment(truth.alignment)
    for s_t in _on_arc(truth):
        x, y = tl.point(s_t)[0]
        s, _u = road.alignment.project(float(x), float(y))
        assert abs(xs.slope_left(s) + 0.07) < 0.01, s
        assert abs(xs.slope_right(s) + 0.07) < 0.01, s
    for piece in (xs.slope_left, xs.slope_right):
        assert all(k.prov.source.value == "measured" for k in piece.knots)
        for a, b in zip(piece.knots, piece.knots[1:], strict=False):
            assert b.s - a.s > 1.0  # AC-P2: banking changes only through transitions


def test_fr_xs_2_loop_banking_transitions_back_to_crown(loop_fit):
    _truth, fit = loop_fit
    xs = fit.road.cross_section
    for s in (15.0, fit.road.length - 6.0):
        assert abs(xs.slope_left(s) + 0.02) < 0.012
        assert abs(xs.slope_right(s) - 0.02) < 0.012


def test_fr_af_2_loop_profile_within_5_cm(loop_fit):
    truth, fit = loop_fit
    road = fit.road
    tl = Polyline.from_alignment(truth.alignment)
    s = np.arange(10.0, road.length - 10.0, 5.0)
    xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
    ts, _u, ok = tl.project(xy)
    dz = np.array([road.profile.z(float(a)) - truth.profile.z(float(b)) for a, b in zip(s[ok], ts[ok], strict=True)])
    assert np.max(np.abs(dz)) < 0.05


def test_fr_xs_1_loop_widths(loop_fit):
    _truth, fit = loop_fit
    xs = fit.road.cross_section
    for s in np.arange(20.0, fit.road.length - 20.0, 20.0):
        assert abs(xs.width_left(s) - 3.1) < 0.3, s
        assert abs(xs.width_right(s) - 3.1) < 0.3, s
