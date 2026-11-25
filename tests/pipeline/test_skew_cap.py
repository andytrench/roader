"""Supports carry the skew the sweep can hold (FR-ST-4, FR-SG-2, D-171).

A 53-degree street bridge whose road is cut 48 m before its first abutment: the solids
square the skew off toward the road end (D-118), so supports placed at the full skew
were swept at about half of it and the first abutment landed on the carriageway below
(AC-H1). Supports are now capped to the skew the sweep holds at their stations and laid
out at it, so placement and sweep agree.
"""

from __future__ import annotations

import math
import os

import pytest

from tests.pipeline import scenes as S


@pytest.fixture(scope="module")
def outcome():
    return S.run_scene(S.simple_overpass(skew=0.92, length=170.0), val3dity_exe=os.environ.get("VAL3DITY"))


def test_d171_supports_near_a_cut_road_end_are_capped_and_flagged(outcome):
    assert outcome.error is None, outcome.error
    model = outcome.model
    capped = [f for f in model.flags if f.code == "FR-ST-4:skew_capped"]
    assert capped, [f.code for f in model.flags]
    st_ids = {st.id for st in model.structures}
    assert all(f.element_id in st_ids for f in capped)
    skews = [abs(math.tan(sp.skew_rad)) for st in model.structures for sp in st.supports]
    assert min(skews) < 1.0 < max(skews)  # capped near the road end, full skew elsewhere


def test_d171_supports_carry_the_skew_the_solids_sweep(outcome):
    from roader.solids import swept_lines
    from roader.solids.terrain import TerrainSampler

    model = outcome.model
    for st in model.structures:
        road = next(r for r in model.roads if r.id == st.carried_road)
        swept = swept_lines(model, road, TerrainSampler(model.terrain), 0.01)
        for sp in st.supports:
            line = min(swept, key=lambda ln, s=sp.station: abs(ln.s - s))
            assert abs(line.t - math.tan(sp.skew_rad)) <= 0.02, (sp.id, line.t, math.tan(sp.skew_rad))


def test_d171_relaid_deck_ends_clear_the_crossed_pavement_with_the_deck_overhang(result):
    """An abutment spans the deck, overhangs included: at a skew other than the crossing's,
    its corners reach farther along the road than the pavement's (Wright Co, I-94 WB squared
    over Broadway Street: 0.44 m into the pavement)."""
    from roader.pipeline import _crossed_footprint
    from roader.rules import RuleBook

    rules = RuleBook.load()
    by_id = {f.road.id: f for f in result.fits}
    c = next(c for c in result.model.crossings if abs(c.skew_rad) > 0.1)
    a0, b0 = _crossed_footprint(by_id, c, 0.0, rules)
    a1, b1 = _crossed_footprint(by_id, c, 0.0, rules, overhang=2.0)
    assert a1 < a0 - 0.1 and b1 > b0 + 0.1


def test_d171_no_support_stands_on_the_road_below(outcome):
    """AC-H1, AC-H3, FR-EX-1 on the capped bridge."""
    assert "FR-EX-1" not in outcome.gates or outcome.gates["FR-EX-1"].passed
    for gate in ("AC-H1", "AC-H3"):
        g = outcome.gates[gate]
        assert g.passed, (gate, g.skipped_reason, [(v.element_id, v.message[:100]) for v in g.violations][:5])
