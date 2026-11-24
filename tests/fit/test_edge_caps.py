"""FR-XS-1, AC-A3, D-103, D-104: tapers and medians bound the edges, in the fit and in the AC-A3 reference.

D-104 (amended): where a linked child runs alongside its parent (a merge or diverge
taper) and their pavements overlap, parent and child share the boundary at the
midline between the parent's fitted edge and the child's fitted inner edge (second
pass, after the children are fitted), its width change limited to a 1:15 design
taper, and never inside either road's own least half-width.
D-103: on the facing side of D-100 twins the AC-A3 reference edge search stops at
the fit's cap; D-104: the reference takes the taper boundary too.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.fit.links import Junction, bound_tapers
from roader.fit.median import (
    _least_half_width,
    cap_at,
    reference_caps,
    split_twin_medians,
)
from roader.model.schema import Link
from tests.fit import scenes

Z = scenes.MAIN_Z


def _fit(road):
    return SimpleNamespace(road=road, line=Polyline.from_alignment(road.alignment), flags=[])


def _scene(child_y=8.0, child_w=2.0, child_dz=0.0):
    """A one-way mainline 10 m wide on its left, and a ramp diverging on that side and running alongside."""
    main = scenes.road("main", "mainline", "motorway", True, scenes.alignment(-300.0, 0.0, 0.0, [("line", 600.0, 0.0, 0.0)]), scenes.profile(Z, 0.0, [(600.0, 0.0)]), scenes.section(10.0, 5.5, 0.02, 0.02))
    ramp = scenes.road("ramp", "ramp", "motorway_link", True, scenes.alignment(-100.0, child_y, 0.0, [("line", 200.0, 0.0, 0.0)]), scenes.profile(Z + child_dz, 0.0, [(200.0, 0.0)]), scenes.section(child_w, child_w, 0.02, 0.02))
    if child_y > 10.0:
        # The diverge starts on the parent's pavement (a model link needs its seam there).
        main.cross_section.width_left = scenes.piecewise([(0.0, 10.0), (190.0, 10.0), (200.0, child_y + 0.4), (210.0, 10.0), (600.0, 10.0)])
    return [_fit(main), _fit(ramp)]


def _left(fit, x):
    return fit.road.cross_section.width_left(x + 300.0)


def _right(fit, s):
    return fit.road.cross_section.width_right(s)


def test_fr_xs_1_taper_boundary_is_the_midline_and_trims_both_roads():
    # Parent edge at +10, child (at +12, 3 m each side) inner edge at +9: they share the midline +9.5.
    fits = _scene(child_y=12.0, child_w=3.0)
    flags = bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    assert _left(fits[0], 0.0) == pytest.approx(9.45, abs=0.05)
    assert _right(fits[1], 100.0) == pytest.approx(12.0 - 9.55, abs=0.05)
    assert fits[1].road.cross_section.width_left(100.0) == pytest.approx(3.0)
    assert fits[0].road.cross_section.width_right(300.0) == pytest.approx(5.5)
    assert _left(fits[0], -250.0) == pytest.approx(10.0)
    assert {(f.code, f.element_id) for f in flags} >= {("FR-XS-1:taper_boundary", "rd_main"), ("FR-XS-1:taper_boundary", "rd_ramp")}


def test_fr_xs_1_taper_boundary_changes_width_no_faster_than_the_design_taper():
    fits = _scene(child_y=8.0, child_w=2.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    for f, w in ((fits[0], fits[0].road.cross_section.width_left), (fits[1], fits[1].road.cross_section.width_right)):
        s = np.arange(0.0, f.road.length, 1.0)
        assert np.max(np.abs(np.diff([w(x) for x in s]))) <= 1.0 / 15.0 + 0.01


def test_fr_xs_1_taper_boundary_keeps_each_roads_own_lanes():
    # Child inner edge at +6, inside the parent's 10 m: midline +8, and the child keeps its lane.
    fits = _scene(child_y=8.0, child_w=2.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    assert _left(fits[0], 0.0) == pytest.approx(7.95, abs=0.05)
    assert _right(fits[1], 100.0) >= _least_half_width(fits[1].road) - 1e-6
    fits = _scene(child_y=3.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    assert _left(fits[0], 0.0) >= _least_half_width(fits[0].road) - 1e-6


def test_fr_xs_1_continuations_and_other_levels_are_not_tapers():
    fits = _scene()
    assert bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", True)]) == []
    fits = _scene(child_dz=3.0)
    assert bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)]) == []
    assert _left(fits[0], 0.0) == pytest.approx(10.0)


def test_ac_a3_reference_takes_the_taper_boundary():
    fits = _scene(child_y=12.0, child_w=3.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    link = Link("lk", "rd_main", "start", "rd_ramp", "start", "diverge", from_s=200.0, to_s=0.0)
    caps = reference_caps([f.road for f in fits], [link], np.zeros((0, 3)))
    assert cap_at(caps["rd_main"].get(1, []), np.array([300.0]))[0] == pytest.approx(_left(fits[0], 0.0), abs=0.05)
    assert cap_at(caps["rd_ramp"].get(-1, []), np.array([100.0]))[0] == pytest.approx(_right(fits[1], 100.0), abs=0.05)
    assert cap_at(caps["rd_main"].get(1, []), np.array([50.0]))[0] >= 10.0


def test_ac_a3_reference_on_twin_facing_sides_stops_at_the_fits_cap():
    roads = scenes.twin_carriageways(median=3.0, dz=0.0)
    for r in roads:
        r.cross_section.width_left = scenes.piecewise([(0.0, 9.0)])
    fits = [_fit(r) for r in roads]
    split_twin_medians(fits, np.zeros((0, 3)))
    caps = reference_caps(roads, [], np.zeros((0, 3)))
    for r in roads:
        cap = cap_at(caps[r.id].get(1, []), np.array([300.0]))[0]
        assert cap == pytest.approx(r.cross_section.width_left(300.0), abs=0.02)


def test_ac_a3_reference_edges_clamp_at_the_cap_before_the_neighbour_limits():
    from roader.pipeline import _clamp_edges

    st = np.array([0.0, 10.0, 20.0])
    left = np.array([5.0, 7.0, np.nan])
    right = np.array([-4.0, -9.0, -3.0])
    caps = {1: [(st, np.array([6.0, 6.0, 6.0]))], -1: [(st, np.array([8.0, 8.0, 8.0]))]}
    lo, ro = _clamp_edges(st, left, right, caps, lambda s: (np.full(len(s), 6.5), np.full(len(s), 8.5)))
    assert np.allclose(lo[:2], [5.0, 6.0]) and np.isnan(lo[2])
    assert np.allclose(ro, [-4.0, -8.0, -3.0])


def test_fr_xs_1_capped_width_keeps_its_provenance():
    """AC-H6: a rule-sourced width capped at a taper boundary keeps its rule id."""
    from roader.model.geometry import Knot, Piecewise
    from roader.model.provenance import Provenance, Source

    fits = _scene(child_y=12.0, child_w=3.0)
    rule = Provenance(Source.RULE, 0.5, rule_id="lane.width")
    ramp = fits[1].road
    ramp.cross_section.width_right = Piecewise([Knot(0.0, 3.0, rule), Knot(200.0, 3.0, rule)])
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)])
    knots = ramp.cross_section.width_right.knots
    assert _right(fits[1], 100.0) < 3.0
    assert all(k.prov.source == Source.RULE and k.prov.rule_id == "lane.width" for k in knots)
    assert all("FR-XS-1:taper_boundary" in k.prov.flags for k in knots)


def test_fr_xs_1_taper_boundary_is_not_applied_on_a_deck():
    """D-104: deck widths unchanged, the 1:15 ramp finishing off the deck (no step at its ends)."""
    fits = _scene(child_y=8.0, child_w=2.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)], decks={"rd_main": [(280.0, 320.0)]})
    s = np.arange(0.0, 600.0, 1.0)
    w = np.array([fits[0].road.cross_section.width_left(x) for x in s])
    on = (s >= 280.0) & (s <= 320.0)
    assert np.allclose(w[on], 10.0)
    assert np.max(np.abs(np.diff(w))) <= 1.0 / 15.0 + 0.01
    assert _left(fits[0], -60.0) == pytest.approx(7.95, abs=0.05)


def test_fr_xs_1_taper_boundary_skips_the_childs_deck_too():
    fits = _scene(child_y=8.0, child_w=2.0)
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)], decks={"rd_ramp": [(80.0, 120.0)]})
    s = np.arange(0.0, 200.0, 1.0)
    w = np.array([_right(fits[1], x) for x in s])
    assert np.allclose(w[(s >= 80.0) & (s <= 120.0)], 2.0)
    assert np.max(np.abs(np.diff(w))) <= 1.0 / 15.0 + 0.01


def test_ac_a3_reference_taper_boundary_is_not_applied_on_a_deck():
    fits = _scene(child_y=12.0, child_w=3.0)
    decks = {"rd_main": [(280.0, 320.0)]}
    bound_tapers(fits, [Junction("rd_ramp", "start", "rd_main", False)], decks=decks)
    link = Link("lk", "rd_main", "start", "rd_ramp", "start", "diverge", from_s=200.0, to_s=0.0)
    caps = reference_caps([f.road for f in fits], [link], np.zeros((0, 3)), decks=decks)
    assert cap_at(caps["rd_main"].get(1, []), np.array([300.0]))[0] >= 10.0 - 1e-6
    assert cap_at(caps["rd_main"].get(1, []), np.array([350.0]))[0] == pytest.approx(_left(fits[0], 50.0), abs=0.05)
