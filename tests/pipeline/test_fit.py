"""Stages 6 and 7: FR-AF-1..5, FR-XS-1..3, FR-RS-4 barriers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.fit import alignment as fa
from roader.fit import profile as fp
from roader.model.geometry import AlignElement, Alignment
from roader.model.provenance import Source
from roader.rules import RuleBook
from tests.pipeline.conftest import by_feature

RULES = RuleBook.load()


@pytest.mark.parametrize("fid", ["eb", "wb", "over"])
def test_fr_af_1_alignment_within_a_few_cm(model, truth, fid):
    road = by_feature(model, fid)
    t = Polyline.from_alignment(by_feature(truth, fid).alignment)
    xy = Polyline.from_alignment(road.alignment).xy[10:-10:5]
    _s, u, ok = t.project(xy)
    assert np.max(np.abs(u[ok])) < 0.10
    assert [e.kind for e in road.alignment.elements] == ["line"]


def test_fr_af_1_arc_and_spirals_recovered():
    """Line, spiral, arc (R = 400 m), spiral, line: radius and continuity come back."""
    prov = None
    from roader.model.provenance import measured

    prov = measured(1.0)
    k = 1.0 / 400.0
    specs = [("line", 150.0, 0.0, 0.0), ("spiral", 80.0, 0.0, k), ("arc", 120.0, k, k), ("spiral", 80.0, k, 0.0), ("line", 150.0, 0.0, 0.0)]
    els, x, y, h, s0 = [], 0.0, 0.0, 0.3, 0.0
    for kind, length, k0, k1 in specs:
        el = AlignElement(kind, s0, length, x, y, h, k0, k1, prov)
        els.append(el)
        x, y, h, _k = el.end_state()
        s0 += length
    truth = Polyline.from_alignment(Alignment(els))
    rng = np.random.default_rng(4)
    s = np.arange(0.0, truth.length, 5.0)
    data = truth.offset_point(s, rng.normal(0.0, 0.05, len(s)))
    chain, rep = fa.fit_alignment(data, np.ones(len(data)), truth.xy, rules=RULES, road_class="motorway_link")
    al = fa.to_alignment(chain)
    assert rep.max_dev_m <= fa.MAX_DEV_M
    radii = [1.0 / abs(e.k0) for e in al.elements if e.kind == "arc" or abs(e.k0) > 1e-4]
    assert any(abs(r - 400.0) / 400.0 < 0.1 for r in radii)
    for a, b in zip(al.elements, al.elements[1:], strict=False):
        xa, ya, ha, ka = a.end_state()
        assert math.hypot(xa - b.x0, ya - b.y0) < 1e-3
        assert abs(math.remainder(ha - b.heading0, 2 * math.pi)) < 1e-5
        assert abs(ka - b.k0) < 1e-9  # curvature continuous (mainline requirement)


@pytest.mark.parametrize("fid", ["eb", "wb", "over"])
def test_fr_af_2_profile_within_5_cm(model, truth, fid):
    road = by_feature(model, fid)
    t_road = by_feature(truth, fid)
    t = Polyline.from_alignment(t_road.alignment)
    s = np.arange(5.0, road.length - 5.0, 5.0)
    xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
    ts, _u, _ok = t.project(xy)
    dz = np.array([road.profile.z(float(a)) - t_road.profile.z(float(b)) for a, b in zip(s, ts, strict=True)])
    assert np.max(np.abs(dz)) < 0.05


def test_fr_af_2_vertical_curve_recovered(model):
    road = by_feature(model, "over")
    curves = [e for e in road.profile.elements if e.g0 != e.g1]
    assert len(curves) == 1
    c = curves[0]
    assert abs(c.length - 160.0) < 5.0
    assert abs(c.g0 - 0.03) < 0.002 and abs(c.g1 + 0.03) < 0.002


def test_fr_af_4_dense_data_override_rule_and_are_flagged():
    """An 8% grade on a motorway (rule max 6%) is kept, not clamped, and flagged."""
    rng = np.random.default_rng(5)
    s = rng.uniform(0.0, 300.0, 20000)
    z = 100.0 + 0.08 * s + rng.normal(0.0, 0.03, s.size)
    v, _rep, _ = fp.fit_profile(s, z, 300.0, rules=RULES, road_class="motorway")
    assert abs(v.grades[0] - 0.08) < 0.002
    ok, _rule = RULES.evaluate("grade", float(v.grades[0]), road_class="motorway")
    assert not ok


def test_fr_af_4_override_flag_in_model(points, truth):
    """The pipeline writes the override into the element provenance and the model flags."""
    from roader.io import from_synthetic
    from roader.model.synthetic import overpass_points
    from roader.pipeline import run_pipeline
    from tests.pipeline.conftest import synthetic_vectors

    steep = __import__("copy").deepcopy(truth)
    for road in steep.roads:
        if road.kind == "mainline":
            for el in road.profile.elements:
                el.g0 = el.g1 = 0.07 if road.alignment.elements[0].heading0 == 0.0 else -0.07
                el.z0 = 100.0 - 0.07 * 300.0 if el.g0 > 0 else 100.0 + 0.07 * 300.0
    pts = from_synthetic(overpass_points(steep))
    res = run_pipeline(pts, synthetic_vectors(steep), steep.frame, steep.unit)
    eb = by_feature(res.model, "eb")
    assert any("FR-AF-4:override:grade" in e.prov.flags for e in eb.profile.elements)
    assert any(f.code == "FR-AF-4:override" and f.element_id == eb.id for f in res.model.flags)
    # The scene's flat terrain puts this road in a 20 m cut at one end, so judge the dominant measured element.
    main = max(eb.profile.elements, key=lambda e: e.prov.n_points)
    assert abs(main.g0 - 0.07) < 0.003


def test_fr_af_5_element_provenance(model):
    for road in model.roads:
        for el in [*road.profile.elements, *road.alignment.elements]:
            p = el.prov
            assert p.source in (Source.MEASURED, Source.VECTOR)
            assert p.rms_m is not None and p.p95_m is not None
            assert p.n_points > 0
            assert p.observed_fraction is not None and 0.0 <= p.observed_fraction <= 1.0
        for el in road.profile.elements:
            assert el.prov.rms_m < 0.05 and el.prov.density_pts_m2 > 2.0


@pytest.mark.parametrize("fid,half", [("eb", 5.5), ("wb", 5.5), ("over", 5.0)])
def test_fr_xs_1_widths_fitted_as_edge_lines(model, fid, half):
    xs = by_feature(model, fid).cross_section
    for piece in (xs.width_left, xs.width_right):
        assert all(abs(k.value - half) < 0.3 for k in piece.knots)
        assert len(piece.knots) <= 3


@pytest.mark.parametrize("fid", ["eb", "wb", "over"])
def test_fr_xs_2_cross_slope_and_crown(model, fid):
    xs = by_feature(model, fid).cross_section
    assert all(abs(k.value + 0.02) < 0.005 for k in xs.slope_left.knots)
    assert all(abs(k.value - 0.02) < 0.005 for k in xs.slope_right.knots)


def test_fr_xs_3_lanes_inferred_and_tagged(model):
    eb = by_feature(model, "eb")
    assert sum(g.count for g in eb.cross_section.lanes) == 2
    assert all(g.prov.source == Source.RULE and g.prov.rule_id == "lane.width" for g in eb.cross_section.lanes)
    assert all("FR-XS-3:inferred" in g.prov.flags for g in eb.cross_section.lanes)


def test_fr_rs_4_parapets_become_barriers_on_the_deck(model):
    over = by_feature(model, "over")
    st = model.structures[0]
    parapets = [b for b in over.cross_section.barriers if b.kind == "parapet"]
    assert {b.side for b in parapets} == {"left", "right"}
    for b in parapets:
        assert b.prov.source == Source.MEASURED
        assert abs(b.height_m - 1.07) < 0.1
        assert abs(b.offset_m - 5.0) < 0.3
        assert b.s0 < st.s_start + 3.0 and b.s1 > st.s_end - 3.0
