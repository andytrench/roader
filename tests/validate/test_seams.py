"""AC-H4 extension: link and junction-area seams, mainline curvature continuity (D-020).

Links with seam stations and junction areas with a surface are milestone 2
contracts. Most checks run on test-local stand-ins with the same fields; the
contract tests at the end run on the schema and the synthetic diverge (D-046).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import pytest

from roader.model import Link, stable_id
from roader.model.geometry import (
    AlignElement,
    Alignment,
    Profile,
    VerticalElement,
    constant,
)
from roader.model.provenance import Provenance, Source
from roader.model.schema import CrossSection, JunctionArea, JunctionSeam, Road
from roader.model.synthetic import (
    MAINLINE_Z,
    MEDIAN_HALF,
    diverge_model,
    overpass_model,
)
from roader.validate.gates import check_continuity
from roader.validate.seams import junction_violations, link_violations

TRUTH = Provenance(Source.MEASURED, 1.0)


@pytest.fixture
def model():
    return overpass_model()


@dataclass
class SeamLink(Link):
    """Link with the milestone 2 seam stations."""

    from_s: float | None = None
    to_s: float | None = None


@dataclass
class Seam:
    road_id: str
    s: float


@dataclass
class PlaneJunction:
    """Junction area whose surface is a crowned plane continuing the eastbound mainline."""

    id: str
    road_ids: list[str]
    outline: list[tuple[float, float]]
    prov: Provenance
    seams: list[Seam] = field(default_factory=list)
    dz: float = 0.0
    grade: float = 0.0
    crown: float = 0.02
    x_end: float = 300.0

    def surface_z(self, x: float, y: float) -> float:
        return MAINLINE_Z + self.dz + self.grade * (x - self.x_end) - self.crown * abs(y + MEDIAN_HALF)


def _section(slope_left: float = -0.02, slope_right: float = 0.02, half: float = 5.5) -> CrossSection:
    return CrossSection(
        width_left=constant(half, TRUTH),
        width_right=constant(half, TRUTH),
        slope_left=constant(slope_left, TRUTH),
        slope_right=constant(slope_right, TRUTH),
        pavement_thickness=constant(0.3, TRUTH),
    )


def _road(name: str, x: float, y: float, h: float, z: float, grade: float = 0.0, xs: CrossSection | None = None, kind: str = "mainline") -> Road:
    return Road(
        id=stable_id("rd", "test", name),
        name=name,
        road_class="motorway",
        kind=kind,
        oneway=True,
        alignment=Alignment([AlignElement("line", 0.0, 200.0, x, y, h, 0.0, 0.0, TRUTH)]),
        profile=Profile([VerticalElement(0.0, 200.0, z, grade, grade, TRUTH)]),
        cross_section=xs or _section(),
    )


def _eb_end(model):
    eb = model.roads[0]
    x, y, h, _k = eb.alignment.evaluate(eb.length)
    return eb, x, y, h


# ---------------------------------------------------------------- links: cross slope (D-020)


def test_link_matching_cross_slope_passes(model):
    eb, x, y, h = _eb_end(model)
    nxt = _road("next", x, y, h, MAINLINE_Z)
    model.roads.append(nxt)
    model.links.append(Link("lk", eb.id, "end", nxt.id, "start", "continue"))
    result = check_continuity(model)
    assert result.passed, result.violations
    assert result.metrics["n_seam_samples"] >= 10


def test_link_cross_slope_mismatch_fails(model):
    eb, x, y, h = _eb_end(model)
    nxt = _road("next", x, y, h, MAINLINE_Z, xs=_section(slope_left=-0.025))
    model.roads.append(nxt)
    model.links.append(Link("lk", eb.id, "end", nxt.id, "start", "continue"))
    result = check_continuity(model)
    msgs = [v.message for v in result.violations if v.element_id == "lk"]
    assert any("cross-slope" in m for m in msgs), msgs


def test_link_reversed_road_with_mirrored_section_passes(model):
    """A continuation drawn the other way: its left is eb's right, so slopes mirror."""
    eb, x, y, h = _eb_end(model)
    back = _road("back", x + 200.0, y, h + math.pi, MAINLINE_Z, xs=_section(slope_left=-0.02, slope_right=0.02))
    model.roads.append(back)
    model.links.append(Link("lk", eb.id, "end", back.id, "end", "continue"))
    result = check_continuity(model)
    assert result.passed, result.violations


def test_link_reversed_grade_sign_is_compared_in_plan(model):
    eb, x, y, h = _eb_end(model)
    # Drawn backwards, a road falling away from eb has a positive grade along its own s.
    back = _road("back", x + 200.0, y, h + math.pi, MAINLINE_Z - 200.0 * 0.004, grade=0.004)
    model.roads.append(back)
    model.links.append(Link("lk", eb.id, "end", back.id, "end", "continue"))
    msgs = [v.message for v in check_continuity(model).violations if v.element_id == "lk"]
    assert any("grade" in m for m in msgs), msgs


def test_link_offset_elevation_catches_bank_step(model):
    """Centreline matches but the right half is banked differently: off-centre samples fail."""
    eb, x, y, h = _eb_end(model)
    nxt = _road("next", x, y, h, MAINLINE_Z, xs=_section(slope_right=0.05))
    model.roads.append(nxt)
    model.links.append(Link("lk", eb.id, "end", nxt.id, "start", "continue"))
    msgs = [v.message for v in check_continuity(model).violations if v.element_id == "lk"]
    assert any("elevation" in m for m in msgs) and any("cross-slope" in m for m in msgs), msgs


# ---------------------------------------------------------------- links: seam stations (contract)


def _diverge(model, dz: float = 0.0):
    """A ramp leaving eb at s=200, its centreline 3 m right of eb's, inside eb's right half."""
    eb = model.roads[0]
    s0, u0 = 200.0, -3.0
    x, y = eb.alignment.offset_point(s0, u0)
    z = eb.surface_z(s0, u0) + dz
    ramp = _road("ramp", x, y, 0.0, z, xs=_section(slope_left=0.02, slope_right=0.02, half=1.5), kind="ramp")
    model.roads.append(ramp)
    return SeamLink("dv", eb.id, "end", ramp.id, "start", "diverge", from_s=s0, to_s=0.0)


def test_diverge_with_seam_stations_passes(model):
    link = _diverge(model)
    roads = {r.id: r for r in model.roads}
    violations, n = link_violations(link, roads)
    assert not violations, violations
    assert n >= 3


def test_diverge_with_seam_stations_elevation_fails(model):
    link = _diverge(model, dz=0.015)
    violations, _n = link_violations(link, {r.id: r for r in model.roads})
    assert any("elevation" in v.message for v in violations), violations


def test_link_with_no_shared_pavement_fails(model):
    eb, x, y, h = _eb_end(model)
    far = _road("far", x, y + 40.0, h, MAINLINE_Z)
    model.roads.append(far)
    violations, _n = link_violations(Link("lk", eb.id, "end", far.id, "start", "continue"), {r.id: r for r in model.roads})
    assert any("do not overlap" in v.message for v in violations)


# ---------------------------------------------------------------- junction areas (contract)


def _junction(model, **kw) -> PlaneJunction:
    eb, x, y, _h = _eb_end(model)
    nxt = _road("after", x + 20.0, y, 0.0, MAINLINE_Z + kw.get("grade", 0.0) * 20.0 + kw.get("dz", 0.0), grade=kw.get("grade", 0.0))
    model.roads.append(nxt)
    outline = [(x, y - 6.0), (x + 20.0, y - 6.0), (x + 20.0, y + 6.0), (x, y + 6.0)]
    return PlaneJunction(
        "jn", [eb.id, nxt.id], outline, TRUTH, seams=[Seam(eb.id, eb.length), Seam(nxt.id, 0.0)], x_end=x, **kw
    )


def test_junction_matching_surface_passes(model):
    junction = _junction(model)
    model.junctions.append(junction)
    result = check_continuity(model)
    assert result.passed, result.violations
    assert result.metrics["n_junctions"] == 1.0


@pytest.mark.parametrize(
    ("kw", "word"),
    [({"dz": 0.02}, "elevation"), ({"grade": 0.003}, "grade"), ({"crown": 0.025}, "cross-slope")],
)
def test_junction_mismatch_fails(model, kw, word):
    junction = _junction(model)
    for k, v in kw.items():
        setattr(junction, k, v)
    violations, _n = junction_violations(junction, {r.id: r for r in model.roads})
    eb = model.roads[0]
    assert any(word in v.message and eb.id in v.message for v in violations), violations


def test_ac_h4_steep_terminal_link_inside_an_area_is_checked_on_its_seams_d128(model):
    """A ramp ending square on a street at grade meets the L6 area, not the street's own surface (D-128).

    Inside the area neither road is rendered, and a child meeting its parent more than
    25 degrees off its direction keeps its own crown (D-051), so the link's own seam
    is not a surface seam; the area's seams are checked. A shallow link still is.
    """
    eb = model.roads[0]
    x, y = eb.alignment.offset_point(150.0, 0.0)
    square = _road("terminal", x, y - 30.0, math.pi / 2.0, MAINLINE_Z, kind="ramp")
    square.alignment = Alignment([AlignElement("line", 0.0, 30.0, x, y - 30.0, math.pi / 2.0, 0.0, 0.0, TRUTH)])
    shallow = _road("shallow", x - 30.0, y, 0.0, MAINLINE_Z, xs=_section(slope_left=-0.025), kind="ramp")
    shallow.alignment = Alignment([AlignElement("line", 0.0, 30.0, x - 30.0, y, 0.0, 0.0, 0.0, TRUTH)])
    model.roads += [square, shallow]
    model.links += [
        SeamLink("lk_t", square.id, "end", eb.id, "start", "merge", from_s=30.0, to_s=150.0),
        SeamLink("lk_s", shallow.id, "end", eb.id, "start", "merge", from_s=30.0, to_s=150.0),
    ]
    flagged = {v.element_id for v in check_continuity(model).violations}
    assert {"lk_t", "lk_s"} <= flagged
    area = PlaneJunction("jn_t", [eb.id, square.id, shallow.id], [(x - 8.0, y - 8.0), (x + 8.0, y - 8.0), (x + 8.0, y + 8.0), (x - 8.0, y + 8.0)], TRUTH)
    model.junctions.append(area)
    flagged = {v.element_id for v in check_continuity(model).violations}
    assert "lk_t" not in flagged and "lk_s" in flagged


def test_junction_side_from_outline_at_road_start(model):
    """A seam at a road's start, with the junction area behind it, matches on a common grade."""
    junction = _junction(model, grade=0.002)
    after = model.roads[-1]
    after.profile = Profile([VerticalElement(0.0, 200.0, MAINLINE_Z + 0.04, 0.002, 0.002, TRUTH)])
    roads = {r.id: r for r in model.roads}
    junction.seams = [Seam(after.id, 0.0)]
    violations, n = junction_violations(junction, roads)
    assert n > 0 and not violations, violations


def test_junction_without_lifted_surface_fails(model):
    """A JunctionArea from before the D-046 patches has seams but no surface: its seams fail."""
    eb, x, y, _h = _eb_end(model)
    outline = [(x - 20.0, y - 6.0), (x, y - 6.0), (x, y + 6.0), (x - 20.0, y + 6.0)]
    seam = JunctionSeam(eb.id, eb.length - 20.0, -5.5, 5.5, side=1)
    model.junctions.append(JunctionArea("jn", [eb.id], outline, TRUTH, seams=[seam]))
    result = check_continuity(model)
    assert not result.passed
    assert any("no surface" in v.message for v in result.violations)


def test_junction_surface_missing_at_seam_fails(model):
    junction = _junction(model)
    junction.surface_z = lambda x, y: math.nan  # type: ignore[method-assign]
    violations, _n = junction_violations(junction, {r.id: r for r in model.roads})
    assert any("no surface" in v.message for v in violations)


# ---------------------------------------------------------------- mainline curvature


def _curved(kinds: list[tuple[str, float, float, float]], kind: str = "mainline") -> Road:
    els = []
    x = y = h = s = 0.0
    for name, length, k0, k1 in kinds:
        el = AlignElement(name, s, length, x, y, h, k0, k1, TRUTH)
        els.append(el)
        x, y, h, _k = el.end_state()
        s += length
    road = _road("curve", 0.0, 0.0, 0.0, MAINLINE_Z, kind=kind)
    road.alignment = Alignment(els)
    road.profile = Profile([VerticalElement(0.0, s, MAINLINE_Z, 0.0, 0.0, TRUTH)])
    return road


def test_mainline_line_to_arc_without_spiral_fails(model):
    model.roads.append(_curved([("line", 100.0, 0.0, 0.0), ("arc", 100.0, 1 / 500, 1 / 500)]))
    result = check_continuity(model)
    assert any("curvature break" in v.message for v in result.violations)


def test_mainline_with_spiral_passes(model):
    model.roads.append(_curved([("line", 100.0, 0.0, 0.0), ("spiral", 80.0, 0.0, 1 / 500), ("arc", 100.0, 1 / 500, 1 / 500)]))
    assert check_continuity(model).passed


def test_ramp_curvature_break_is_not_a_mainline_check(model):
    model.roads.append(_curved([("line", 100.0, 0.0, 0.0), ("arc", 100.0, 1 / 100, 1 / 100)], kind="ramp"))
    assert check_continuity(model).passed


# ---------------------------------------------------------------- schema contracts (skip until built)


def test_link_schema_has_seam_stations():
    names = {f.name for f in dataclasses.fields(Link)}
    if not {"from_s", "to_s"} <= names:
        pytest.skip("Link.from_s / Link.to_s (seam stations) not in the schema yet (lane model, milestone 2)")
    link = Link("lk", "a", "end", "b", "start", "diverge")
    assert link.from_s is None and link.to_s is None


def test_junction_area_schema_has_seams_and_surface():
    assert hasattr(JunctionArea, "surface_z") and "seams" in {f.name for f in dataclasses.fields(JunctionArea)}
    model = diverge_model()
    assert model.junctions
    result = check_continuity(model)
    assert result.passed, result.violations
    assert result.metrics["n_junctions"] == 1.0
