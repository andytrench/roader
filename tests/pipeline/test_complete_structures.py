"""Stages 8 and 9: FR-OC-1..3, FR-ST-2..6, L4 side slopes, L7 terrain."""

from __future__ import annotations

import copy
import math

import numpy as np

from roader.complete import check_clearances
from roader.model.provenance import Source
from roader.rules import RuleBook
from tests.pipeline.conftest import by_feature

RULES = RuleBook.load()


def test_fr_oc_1_lower_road_carried_through_without_inventing_a_sag(model):
    for fid in ("eb", "wb"):
        road = by_feature(model, fid)
        iv = next(i for i in road.support if i.kind == "occluded")
        z = [road.profile.z(float(s)) for s in np.linspace(iv.s0 - 20.0, iv.s1 + 20.0, 41)]
        assert max(z) - min(z) < 0.03
        assert all(abs(v - 100.0) < 0.05 for v in z)
        assert all(e.g0 == e.g1 for e in road.profile.elements)  # no vertical curve placed in the gap
        assert any("FR-OC-1:carried" in e.prov.flags for e in road.profile.elements)
        knots = road.cross_section.width_left.knots
        assert not any(iv.s0 < k.s < iv.s1 for k in knots)


def test_fr_oc_2_clearance_checked_against_the_structure(model):
    for c in model.crossings:
        assert c.clearance_m is not None
        assert c.clearance_m >= 4.877  # motorway rural minimum, confirmed rule
    assert not [f for f in model.flags if f.req == "FR-OC-2"]


def test_fr_oc_2_failure_refits_depth_then_flags(model):
    m = copy.deepcopy(model)
    st = m.structures[0]
    st.deck.depth_m = 5.9  # an impossible rule fill: soffit near the freeway
    flags = check_clearances(m, RULES)
    assert "FR-OC-2:depth_refit_to_rule_low" in st.deck.prov.flags
    assert st.deck.depth_m < 5.9
    m2 = copy.deepcopy(model)
    for road in m2.roads:
        if road.kind == "mainline":
            for el in road.profile.elements:
                el.z0 += 3.0  # measured surfaces closer together: refit cannot help
    flags = check_clearances(m2, RULES)
    assert {f.code for f in flags} >= {"FR-OC-2:below_minimum"}


def test_fr_oc_3_side_slopes_and_no_slope_over_the_span(model):
    over = by_feature(model, "over")
    st = model.structures[0]
    for s in model.side_slopes:
        road = model.road(s.road_id)
        if road.kind == "mainline":
            assert s.kind == "fill" and abs(s.ratio - 4.0) < 0.4
        else:
            assert s.kind == "fill" and abs(s.ratio - 2.0) < 0.3
            assert s.s1 <= st.s_start + 1e-6 or s.s0 >= st.s_end - 1e-6
    assert {(s.side) for s in model.side_slopes if s.road_id == over.id} == {"left", "right"}


def test_l7_terrain_has_no_ground_on_the_deck(model):
    t = model.terrain
    assert t is not None
    over = by_feature(model, "over")
    st = model.structures[0]
    for s in np.linspace(st.s_start + 2.0, st.s_end - 2.0, 9):
        x, y = over.alignment.offset_point(float(s), 0.0)
        ix, iy = int((x - t.origin_x) // t.cell_m), int((y - t.origin_y) // t.cell_m)
        z = t.z[iy * t.nx + ix]
        assert z is None or z < over.profile.z(float(s)) - 2.0


def test_fr_st_3_and_4_supports(model, truth):
    st = model.structures[0]
    ts = truth.structures[0]
    kinds = [sp.kind for sp in sorted(st.supports, key=lambda p: p.station)]
    assert kinds == ["abutment", "pier", "abutment"]
    pier = next(sp for sp in st.supports if sp.kind == "pier")
    t_pier = next(sp for sp in ts.supports if sp.kind == "pier")
    assert abs(pier.station - t_pier.station) < 2.0  # in the median, never in a lane
    assert all(abs(sp.skew_rad - 0.17) < 0.01 for sp in st.supports)
    assert all(abs(sp.footing_z - (99.0 - 1.5)) < 0.5 for sp in st.supports)
    assert all(sp.prov.source == Source.RULE and sp.prov.rule_id for sp in st.supports)


def test_fr_st_2_and_5_family_and_rule_sized_deck(model):
    st = model.structures[0]
    spans = np.diff(sorted(sp.station for sp in st.supports))
    assert st.family == "multi_girder"  # spans of about 26 m exceed the slab rule
    ratio = RULES.get("superstructure.depth_ratio.multi_girder").typical
    assert math.isclose(st.deck.depth_m, ratio * spans.max(), rel_tol=1e-6)
    assert st.deck.prov.source == Source.RULE and st.deck.prov.rule_id
    assert st.deck.overhang_m >= 0.5


def test_fr_st_6_short_span_is_not_queued_for_review(model):
    assert model.structures[0].review is False
    assert not [f for f in model.flags if f.req == "FR-ST-6"]
