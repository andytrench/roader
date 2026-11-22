"""FR-AF-3: joint L2 fitting over connected components, checked with the AC-H4 seam test.

Core tests fit each road's profile on its own from noisy returns, then solve the
component jointly: every link seam must then meet D-020 (1 cm, 0.1%) and every
crossing keep its measured height difference. The pipeline test runs the whole
chain on a synthetic diverge and merge.
"""

from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.fit import profile as fp
from roader.fit.joint import JointRoad, joint_fit, seam_of
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing, Link
from roader.rules import RuleBook
from roader.validate.seams import link_violations
from tests.fit import scenes

RULES = RuleBook.load()


def _surface_returns(road, rng, spacing=0.5, bias=0.0):
    line = Polyline.from_alignment(road.alignment)
    s = np.arange(0.0, road.length, spacing)
    u = np.arange(-2.5, 2.5, 1.0)
    ss, uu = (a.ravel() for a in np.meshgrid(s, u))
    ss = np.clip(ss + rng.uniform(-0.2, 0.2, ss.size), 0.0, road.length)
    z = np.array([road.surface_z(float(a), float(b)) for a, b in zip(ss, uu, strict=True)])
    del line
    return ss, uu, z + rng.normal(0.0, 0.03, z.size) + bias


def _independent(truth, seed=5, biases=None):
    """Each road's L2 fitted on its own returns; cross slopes perturbed by 1% so seams start wrong."""
    rng = np.random.default_rng(seed)
    biases = biases or {}
    items = []
    for t in truth:
        rd = copy.deepcopy(t)
        s, u, z = _surface_returns(t, rng, bias=biases.get(t.id, 0.0))
        xs = rd.cross_section
        zc = z - np.array([(xs.slope_left if b >= 0 else xs.slope_right)(float(a)) * b for a, b in zip(s, u, strict=True)])
        vert, _rep, _bins = fp.fit_profile(s, zc, rd.length, rules=RULES, road_class=rd.road_class)
        rd.profile = fp.to_profile(vert, rd.length)
        if rd.kind == "ramp":
            for k in xs.slope_left.knots + xs.slope_right.knots:
                k.value += 0.01
        items.append(JointRoad(rd, vert, s, u, z))
    return items


def _links(truth):
    m, exit_ramp, entry = truth[:3]
    s_merge = m.alignment.project(*entry.alignment.evaluate(entry.length)[:2])[0]
    return [
        Link("lk_exit", m.id, "start", exit_ramp.id, "start", "diverge", from_s=scenes.EXIT_S, to_s=0.0),
        Link("lk_entry", entry.id, "end", m.id, "end", "merge", from_s=entry.length, to_s=s_merge),
    ]


def _violations(items, links):
    roads = {it.road.id: it.road for it in items}
    out = []
    for lk in links:
        v, n = link_violations(lk, roads)
        assert n > 0
        out.extend(v)
    return out


def test_fr_af_3_independent_fits_break_seams():
    """Precondition: fitted one by one, the scene's seams fail AC-H4 (else the next test proves nothing)."""
    truth = scenes.diverge_merge_scene()
    items = _independent(truth, biases={truth[1].id: 0.04, truth[2].id: -0.04})
    assert _violations(items, _links(truth))


def test_fr_af_3_joint_fit_matches_elevation_and_slope_at_seams():
    truth = scenes.diverge_merge_scene()
    items = _independent(truth, biases={truth[1].id: 0.04, truth[2].id: -0.04})
    links = _links(truth)
    flags = joint_fit(items, links, [], RULES)
    assert _violations(items, links) == []
    assert not [f for f in flags if f.code == "FR-AF-3:infeasible"]
    # Away from the seams each road still follows its own returns (FR-AF-2).
    for it, t in zip(items, truth, strict=True):
        s = np.arange(80.0, t.length - 80.0, 5.0)
        dz = np.array([it.road.profile.z(float(v)) - t.profile.z(float(v)) for v in s])
        bias = 0.04 if t.id == truth[1].id else -0.04 if t.id == truth[2].id else 0.0
        assert np.max(np.abs(dz - bias)) < 0.05, t.id
        assert all("FR-AF-3:joint" in e.prov.flags for e in it.road.profile.elements)


def test_fr_af_3_profile_continuity_kept_after_joint_fit():
    """AC-H4 within a road: elevation and grade continuous at every L2 joint after the joint solve."""
    truth = scenes.diverge_merge_scene()
    items = _independent(truth, biases={truth[1].id: 0.04})
    joint_fit(items, _links(truth), [], RULES)
    for it in items:
        els = it.road.profile.elements
        for a, b in itertools.pairwise(els):
            assert abs(a.z(a.length) - b.z0) < 1e-6
            assert abs(a.grade(a.length) - b.g0) < 1e-6
            assert abs(a.s0 + a.length - b.s0) < 1e-6


def _over_street(truth, ramp):
    """A two-lane street crossing the exit ramp at its s = 250 m, 6 m above it."""
    x, y, h, _k = ramp.alignment.evaluate(250.0)
    hh = h + math.pi / 2.0 + 0.2
    al = scenes.alignment(x - 150.0 * math.cos(hh), y - 150.0 * math.sin(hh), hh, [("line", 300.0, 0.0, 0.0)])
    z = ramp.profile.z(250.0) + 6.0
    pr = scenes.profile(z - 0.02 * 150.0, 0.02, [(100.0, 0.02), (100.0, -0.02), (100.0, -0.02)])
    return scenes.road("over", "crossing_street", "primary", False, al, pr, scenes.section(4.0, 4.0, -0.02, 0.02))


def test_fr_af_3_crossing_clearance_kept_as_a_constraint():
    truth = scenes.diverge_merge_scene()
    over = _over_street(truth, truth[1])
    truth = [*truth, over]
    items = _independent(truth, biases={truth[1].id: 0.06})
    roads = {it.road.id: it.road for it in items}
    ramp, street = roads[truth[1].id], roads[over.id]
    x, y, _h, _k = truth[1].alignment.evaluate(250.0)
    su, uu = street.alignment.project(x, y)
    sl, ul = ramp.alignment.project(x, y)
    dz0 = street.surface_z(su, uu) - ramp.surface_z(sl, ul)
    cx = Crossing("cx_over", street.id, su, ramp.id, sl, x, y, dz0, 0.2, Provenance(Source.MEASURED, 0.9))
    links = _links(truth)
    joint_fit(items, links, [cx], RULES)
    assert _violations(items, links) == []
    dz1 = street.surface_z(su, uu) - ramp.surface_z(sl, ul)
    assert abs(dz1 - dz0) < 1e-3


def test_fr_af_3_seam_of_link():
    truth = scenes.diverge_merge_scene()
    m, exit_ramp, entry = truth
    roads = {r.id: r for r in truth}
    a, b = _links(truth)
    sa = seam_of(a, roads)
    assert (sa.child, sa.parent) == (exit_ramp.id, m.id)
    assert sa.child_s == 0.0 and abs(sa.parent_s - scenes.EXIT_S) < 1e-3 and abs(sa.parent_u) < 1e-3
    sb = seam_of(b, roads)
    assert (sb.child, sb.parent) == (entry.id, m.id)
    assert sb.child_s == pytest.approx(entry.length)


# ---------------------------------------------------------------- pipeline


@pytest.fixture(scope="module")
def diverge_result():
    from roader.pipeline import run_pipeline

    truth = scenes.diverge_merge_scene()
    pts = scenes.to_pointset(scenes.points(truth))
    from roader.model.synthetic import overpass_model

    ref = overpass_model()
    return truth, run_pipeline(pts, scenes.vectors(truth), ref.frame, ref.unit)


def _by_feature(model, fid):
    return next(r for r in model.roads if any(ref.feature_id == fid for ref in r.vector_refs))


def test_fr_af_3_pipeline_finds_links(diverge_result):
    _truth, res = diverge_result
    model = res.model
    m, ex, en = (_by_feature(model, f) for f in ("main", "exit", "entry"))
    kinds = {(lk.from_road, lk.to_road): lk.kind for lk in model.links}
    assert kinds == {(m.id, ex.id): "diverge", (en.id, m.id): "merge"}
    for lk in model.links:
        assert lk.prov is not None and lk.prov.source == Source.MEASURED


def test_ac_h4_pipeline_seams_pass(diverge_result):
    from roader.validate.seams import seam_violations

    _truth, res = diverge_result
    violations, n = seam_violations(res.model)
    assert n > 0
    assert violations == [], [v.message for v in violations]


def test_scene_ramps_do_not_cross():
    import shapely

    _m, ex, en = scenes.diverge_merge_scene()
    lines = [shapely.LineString(Polyline.from_alignment(r.alignment).xy) for r in (ex, en)]
    assert lines[0].distance(lines[1]) > 50.0


def test_fr_af_2_pipeline_ramp_profiles_follow_truth(diverge_result):
    """Ramps are compared where they are their own surface (past the separation from the mainline)."""
    truth, res = diverge_result
    for t in truth:
        road = _by_feature(res.model, t.vector_refs[0].feature_id)
        tl = Polyline.from_alignment(t.alignment)
        fid = t.vector_refs[0].feature_id
        lo = scenes.SEPARATE_M if fid == "exit" else 30.0
        hi = road.length - (scenes.SEPARATE_M if fid == "entry" else 30.0)
        s = np.arange(lo, hi, 5.0)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        ts, _u, ok = tl.project(xy)
        dz = np.array([road.profile.z(float(a)) - t.profile.z(float(b)) for a, b in zip(s[ok], ts[ok], strict=True)])
        assert np.max(np.abs(dz)) < 0.06, t.id
