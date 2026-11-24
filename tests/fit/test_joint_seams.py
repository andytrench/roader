"""FR-AF-3, AC-H4: seams that the joint solve must hold without bending the roads (Woburn cases).

- A ramp whose first 40 m are unobserved and whose surface beyond rises 0.5 m off
  its parent's tangent: the independent fit extrapolates the wrong elevation and
  grade to the seam, and the seam must be met inside the gap (D-050 extra curves).
- A street crossing a mainline exactly where an entry ramp merges onto it: the
  ramp lies on the mainline's pavement there, so its crossing is the mainline's
  and a slightly different recorded height difference must not be forced.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from roader.fit import profile as fp
from roader.fit.joint import JointRoad, joint_fit
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing, Link
from roader.rules import RuleBook
from roader.validate.seams import link_violations
from tests.fit import scenes
from tests.fit.test_joint import _independent, _links, _surface_returns

RULES = RuleBook.load()
GAP_M = 40.0
STEP_M = 0.3


def _gap_scene():
    """Mainline and an exit ramp leaving at 20 degrees, climbing 0.3 m above the parent's tangent between 16 and 40 m.

    The steep departure takes its crown off the parent's pavement within 16 m, so
    the climb is physically its own surface.
    """
    m = scenes.mainline()
    # Planar, so a ramp leaving at an angle can share its surface (D-051).
    m.cross_section.slope_left = scenes.piecewise([(0.0, -0.02)])
    m.cross_section.slope_right = scenes.piecewise([(0.0, -0.02)])
    ramp = scenes._ramp_from(m, scenes.EXIT_S, "exit", -1.0, 0.035)
    x, y, h, _k = m.alignment.evaluate(scenes.EXIT_S)
    ramp.alignment = scenes.alignment(x, y, h - math.radians(20.0), [("line", 80.0, 0.0, 0.0), ("arc", 200.0, -0.005, -0.005), ("line", ramp.length - 280.0, 0.0, 0.0)])
    z0, g0 = m.profile.z(scenes.EXIT_S), m.profile.grade(scenes.EXIT_S)
    a = STEP_M / 12.0
    ramp.profile = scenes.profile(z0, g0, [(16.0, g0), (12.0, g0 + a), (12.0, g0), (100.0, 0.035), (ramp.length - 140.0, 0.035)])
    return [m, ramp]


def _gap_items(truth):
    rng = np.random.default_rng(7)
    items = []
    for t in truth:
        rd = copy.deepcopy(t)
        s, u, z = _surface_returns(t, rng)
        if t.kind == "ramp":
            keep = s >= GAP_M
            s, u, z = s[keep], u[keep], z[keep]
        xs = rd.cross_section
        zc = z - np.array([(xs.slope_left if b >= 0 else xs.slope_right)(float(a)) * b for a, b in zip(s, u, strict=True)])
        vert, _rep, _bins = fp.fit_profile(s, zc, rd.length, rules=RULES, road_class=rd.road_class)
        rd.profile = fp.to_profile(vert, rd.length)
        gaps = [(0.0, GAP_M)] if t.kind == "ramp" else []
        items.append(JointRoad(rd, vert, s, u, z, gaps))
    return items


def test_fr_af_3_gap_fixture_breaks_the_seam():
    """Precondition: the ramp fitted on its own misses its parent at the seam by decimetres."""
    truth = _gap_scene()
    items = _gap_items(truth)
    m, ramp = (it.road for it in items)
    dz = ramp.profile.z(0.0) - m.profile.z(scenes.EXIT_S)
    assert abs(dz) > 0.2


def test_fr_af_3_seam_in_an_unobserved_gap_is_met_without_bending_the_ramp():
    truth = _gap_scene()
    items = _gap_items(truth)
    link = Link("lk_exit", truth[0].id, "start", truth[1].id, "start", "diverge", from_s=scenes.EXIT_S, to_s=0.0)
    flags = joint_fit(items, [link], [], RULES)
    assert not [f for f in flags if f.code in ("FR-AF-3:seam_dropped", "FR-AF-3:reverted", "FR-AF-3:infeasible")], [f.message for f in flags]
    roads = {it.road.id: it.road for it in items}
    v, n = link_violations(link, roads)
    assert n > 0 and v == [], [x.message for x in v]
    # Where the ramp is observed it still follows its own returns (FR-AF-2).
    ramp, t = items[1].road, truth[1]
    s = np.arange(GAP_M, t.length - 5.0, 5.0)
    dz = np.array([ramp.profile.z(float(a)) - t.profile.z(float(a)) for a in s])
    assert np.max(np.abs(dz)) < 0.03
    main, tm = items[0].road, truth[0]
    s = np.arange(5.0, tm.length - 5.0, 5.0)
    assert max(abs(main.profile.z(float(a)) - tm.profile.z(float(a))) for a in s) < 0.03


def test_fr_af_3_seam_transition_curves_lie_next_to_the_seam():
    """The extra curves of a seam end within the seam window (here the unobserved stretch); every
    curve of the ramp's own fit keeps its far end."""
    from roader.fit.joint import SEAM_WINDOW_M

    truth = _gap_scene()
    items = _gap_items(truth)
    v0 = items[1].vert
    before = v0.pvi + 0.5 * v0.lengths
    link = Link("lk_exit", truth[0].id, "start", truth[1].id, "start", "diverge", from_s=scenes.EXIT_S, to_s=0.0)
    joint_fit(items, [link], [], RULES)
    v1 = items[1].vert
    after = v1.pvi + 0.5 * v1.lengths
    assert all(np.any(np.abs(after - b) < 1e-6) for b in before)
    new = [b for b in after if not np.any(np.abs(before - b) < 1e-6)]
    assert len(new) == 2 and max(new) <= min(SEAM_WINDOW_M, GAP_M + fp.BIN_M) + 1e-6


def test_ac_h4_seam_holds_where_the_parents_cross_slope_is_in_transition():
    """A ramp leaving where the parent's right side banks from 2% to 6%: grade matches at every seam offset."""
    truth = scenes.diverge_merge_scene()[:2]
    m = truth[0]
    m.cross_section.slope_right = scenes.piecewise([(scenes.EXIT_S - 30.0, 0.02), (scenes.EXIT_S + 30.0, 0.06)])
    truth[1] = scenes._ramp_from(m, scenes.EXIT_S, "exit", -1.0, 0.035)
    items = _independent(truth, biases={truth[1].id: 0.03})
    link = Link("lk_exit", m.id, "start", truth[1].id, "start", "diverge", from_s=scenes.EXIT_S, to_s=0.0)
    joint_fit(items, [link], [], RULES)
    v, n = link_violations(link, {it.road.id: it.road for it in items})
    assert n > 0 and v == [], [q.message for q in v]


def test_fr_af_3_unobserved_child_on_its_parents_pavement_follows_the_parent():
    """Exit ramp with no returns over its first 60 m, where it still lies on the mainline; its returns beyond read 5 cm high."""
    truth = scenes.diverge_merge_scene()[:2]
    rng = np.random.default_rng(9)
    items = []
    for t in truth:
        rd = copy.deepcopy(t)
        s, u, z = _surface_returns(t, rng)
        if t.kind == "ramp":
            keep = s >= 60.0
            s, u, z = s[keep], u[keep], z[keep] + 0.05
        xs = rd.cross_section
        zc = z - np.array([(xs.slope_left if b >= 0 else xs.slope_right)(float(a)) * b for a, b in zip(s, u, strict=True)])
        vert, _rep, _bins = fp.fit_profile(s, zc, rd.length, rules=RULES, road_class=rd.road_class)
        rd.profile = fp.to_profile(vert, rd.length)
        items.append(JointRoad(rd, vert, s, u, z, [(0.0, 60.0)] if t.kind == "ramp" else []))
    link = Link("lk_exit", truth[0].id, "start", truth[1].id, "start", "diverge", from_s=scenes.EXIT_S, to_s=0.0)
    joint_fit(items, [link], [], RULES)
    main, ramp = items[0].road, items[1].road
    for a in np.arange(5.0, 45.0, 5.0):
        x, y = ramp.alignment.offset_point(float(a), 0.0)
        ps, pu = main.alignment.project(x, y)
        assert abs(ramp.surface_z(float(a), 0.0) - main.surface_z(ps, pu)) < 0.01, a


def _merge_street(truth):
    """A two-lane street crossing the mainline at the entry ramp's merge point, 6 m above."""
    m, entry = truth[0], truth[2]
    x, y, h, _k = entry.alignment.evaluate(entry.length - 5.0)
    hh = h + math.pi / 2.0
    al = scenes.alignment(x - 150.0 * math.cos(hh), y - 150.0 * math.sin(hh), hh, [("line", 300.0, 0.0, 0.0)])
    sm, _um = m.alignment.project(x, y)
    z = m.profile.z(sm) + 6.0
    pr = scenes.profile(z - 0.02 * 150.0, 0.02, [(100.0, 0.02), (100.0, -0.02), (100.0, -0.02)])
    return scenes.road("over", "crossing_street", "primary", False, al, pr, scenes.section(4.0, 4.0, -0.02, 0.02)), (x, y)


def test_fr_af_3_crossing_on_the_parents_pavement_is_the_parents_crossing():
    truth = scenes.diverge_merge_scene()
    over, (x, y) = _merge_street(truth)
    truth = [*truth, over]
    items = _independent(truth, biases={truth[2].id: 0.03})
    roads = {it.road.id: it.road for it in items}
    main, entry, street = roads[truth[0].id], roads[truth[2].id], roads[over.id]
    prov = Provenance(Source.MEASURED, 0.9)
    su, uu = street.alignment.project(x, y)
    crossings = []
    for low, extra in ((main, 0.0), (entry, 0.1)):
        sl, ul = low.alignment.project(x, y)
        dz = street.surface_z(su, uu) - low.surface_z(sl, ul) + extra
        crossings.append(Crossing(f"cx_{low.id}", street.id, su, low.id, sl, x, y, dz, 0.0, prov))
    links = _links(truth)
    flags = joint_fit(items, links, crossings, RULES)
    codes = {(f.code, f.element_id) for f in flags}
    assert ("FR-AF-3:crossing_on_parent", crossings[1].id) in codes
    assert not [f for f in flags if f.code in ("FR-AF-3:seam_dropped", "FR-AF-3:reverted", "FR-AF-3:infeasible")], [f.message for f in flags]
    for lk in links:
        v, _n = link_violations(lk, roads)
        assert v == [], [q.message for q in v]
    sl, ul = main.alignment.project(x, y)
    assert abs(street.surface_z(su, uu) - main.surface_z(sl, ul) - crossings[0].dz_m) < 1e-3
    for it, t in zip(items, truth, strict=True):
        s = np.arange(5.0, t.length - 5.0, 10.0)
        bias = 0.03 if t.id == truth[2].id else 0.0
        assert max(abs(it.road.profile.z(float(v)) - t.profile.z(float(v)) - bias) for v in s) < 0.1, t.id


def test_ac_h4_continuation_continues_the_parents_cross_slope_transition():
    """D-164: a road continuing another end to end, where the first's superelevation runs through
    its end (2% to -2% over 100 m), keeps the change of cross slope across the seam (AC-H4, FR-AF-3).

    Beyond the parent's end there is no parent surface to follow, so the child's hold
    continues the parent's change of cross slope instead of holding the seam value.
    """
    a = scenes.road("first", "ramp", "motorway_link", True, scenes.alignment(-300.0, 0.0, 0.0, [("arc", 300.0, 0.02, 0.02)]), scenes.profile(scenes.MAIN_Z, -0.02, [(300.0, -0.02)]), scenes.section(5.5, 5.5, 0.0, 0.0))
    x, y, h, k = a.alignment.evaluate(a.length)
    b = scenes.road("second", "ramp", "motorway_link", True, scenes.alignment(x, y, h, [("arc", 300.0, k, k)]), scenes.profile(a.profile.z(a.length), -0.02, [(300.0, -0.02)]), scenes.section(5.5, 5.5, 0.0, 0.0))
    a.cross_section.slope_left = scenes.piecewise([(0.0, 0.02), (250.0, 0.02), (300.0, 0.0)])
    a.cross_section.slope_right = scenes.piecewise([(0.0, -0.02), (250.0, -0.02), (300.0, 0.0)])
    b.cross_section.slope_left = scenes.piecewise([(0.0, 0.0), (50.0, -0.02), (300.0, -0.02)])
    b.cross_section.slope_right = scenes.piecewise([(0.0, 0.0), (50.0, 0.02), (300.0, 0.02)])
    items = _independent([a, b])
    link = Link("lk_cont", a.id, "end", b.id, "start", "continue", from_s=a.length, to_s=0.0)
    joint_fit(items, [link], [], RULES)
    v, n = link_violations(link, {it.road.id: it.road for it in items})
    assert n > 0 and v == [], [q.message for q in v]
