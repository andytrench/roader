"""A deck spans every road it crosses: data ends inside a crossed pavement are moved clear (FR-VC-4, FR-ST-3, AC-H3, D-166)."""

from __future__ import annotations

import math

import pytest

from roader.conflate.structures import StructureExtent, cover_crossed_roads
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing

OBS = Provenance(Source.MEASURED, 0.9)


def _crossing(s: float, skew: float = 0.0, cid: str = "cx_a") -> Crossing:
    return Crossing(cid, "up", s, "low", 50.0, 0.0, 0.0, 6.0, skew, OBS)


def test_d166_an_end_inside_the_crossed_footprint_moves_clear_and_is_flagged():
    """Columbus I-71: class 17 and the side drop both missed the deck's first 19 m, so the
    abutment stood on East Dublin Granville Road (AC-H1, AC-H3)."""
    e = StructureExtent("up", 606.0, 642.0, "class17", (605.9, 637.9), (606.0, 642.2), one_sided=True)
    seen = []

    def footprint(c, t):
        seen.append((c.id, t))
        return c.upper_s - 9.0, c.upper_s + 9.0

    out, flags = cover_crossed_roads([e], [_crossing(607.0), _crossing(624.5, cid="cx_b")], footprint)
    (got,) = out
    assert (got.s_start, got.s_end) == (598.0, 642.0)
    assert got.method == "class17" and got.class17 == (606.0, 642.2)  # the evidence is kept for AC-A5 and review
    assert [f.code for f in flags] == ["FR-VC-4:span_extended"]
    assert "start 606.0 -> 598.0" in flags[0].message
    assert sorted(seen) == [("cx_a", 0.0), ("cx_b", 0.0)]


def test_d166_ends_already_clear_rule_spans_and_other_roads_are_left_alone():
    e = StructureExtent("up", 590.0, 650.0, "class17", None, (590.0, 650.0))
    rule = StructureExtent("up", 700.0, 705.0, "rule", None, None)
    other = StructureExtent("other", 600.0, 610.0, "sheet_break", (600.0, 610.0), None)
    out, flags = cover_crossed_roads([e, rule, other], [_crossing(607.0), _crossing(702.0, cid="cx_r")], lambda c, t: (c.upper_s - 9.0, c.upper_s + 9.0))
    assert out == [e, rule, other] and flags == []


def test_d166_the_footprint_uses_the_supports_skew():
    """Supports follow the mean skew of the crossings under the deck (FR-ST-4), so the footprint is read along those lines."""
    e = StructureExtent("up", 600.0, 640.0, "class17", None, (600.0, 640.0))
    seen = []
    cover_crossed_roads([e], [_crossing(610.0, 0.4), _crossing(630.0, 0.6, "cx_b")], lambda c, t: (seen.append(t), (c.upper_s, c.upper_s))[1])
    assert seen == [pytest.approx(math.tan(0.5))] * 2


def test_d166_pipeline_moves_a_short_data_end_off_the_lower_road(points, vectors, truth, monkeypatch):
    """FR-VC-4, AC-H3: a data end on the lower road (evidence missed one end) becomes a span clear of it."""
    import roader.pipeline as P

    real = P.find_extents
    cut = {}

    def short(fit, ev, skews=None):
        out = real(fit, ev, skews)
        for e in out:
            for s, _k in skews or []:
                if e.s_start < s < e.s_end:
                    cut[fit.road.id] = s
                    e.s_start = s - 1.0  # the deck "starts" on the lower road's centre line
        return out

    monkeypatch.setattr(P, "find_extents", short)
    model = P.run_pipeline(points, vectors, truth.frame, truth.unit).model
    assert cut
    (st,) = [st for st in model.structures if st.carried_road in cut]
    assert any(f.code == "FR-VC-4:span_extended" for f in model.flags)
    lower = next(c for c in model.crossings if c.upper_road == st.carried_road)
    road = next(r for r in model.roads if r.id == lower.lower_road)
    half = max(road.cross_section.width_left(lower.lower_s), road.cross_section.width_right(lower.lower_s))
    assert st.s_start < lower.upper_s - half  # the abutment stands off the lower pavement
