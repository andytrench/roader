"""A road with too few surface returns gets a rule-sourced profile instead of failing (FR-AF-2, FR-VC-2)."""

from __future__ import annotations

import dataclasses

import numpy as np

from roader.conflate.graph import data_extent, select_roads
from roader.conflate.polyline import Polyline
from roader.evidence import build_evidence
from roader.fit.road import fit_road
from roader.rules import RuleBook
from tests.fit import scenes

RULES = RuleBook.load()


def test_fr_af_2_too_few_profile_samples_fall_back_to_a_flagged_rule_profile():
    truth = scenes.loop_ramp()
    pts = scenes.to_pointset(scenes.points([truth]))
    seed = select_roads(scenes.vectors([truth]), data_extent(pts.xyz[:, :2]))[0]
    ev = build_evidence(pts, [seed.line.xy])
    # The same road drawn 300 m away, where the lidar has no returns at all.
    away = dataclasses.replace(seed, line=Polyline(seed.line.xy + np.array([0.0, 300.0])))
    fit = fit_road(away, ev, RULES)
    assert any(f.code == "FR-AF-2:profile_fallback" for f in fit.flags)
    for el in fit.road.profile.elements:
        assert el.prov.source.value == "rule"
        assert el.prov.rule_id == "grade"
        assert "FR-AF-2:profile_fallback" in el.prov.flags
    for piece in (fit.road.cross_section.slope_left, fit.road.cross_section.slope_right):
        assert all(k.prov.rule_id == "cross_slope.crown" for k in piece.knots)
    assert np.isfinite(fit.road.profile.z(0.5 * fit.road.length))
