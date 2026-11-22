"""AC-H1 to AC-H6 on the synthetic overpass and hand-built meshes."""

from __future__ import annotations

import pytest

from roader.model import Link, Mesh, stable_id
from roader.model.geometry import AlignElement, Alignment, Profile, VerticalElement
from roader.model.provenance import Provenance, Source
from roader.model.schema import Road
from roader.model.synthetic import overpass_model
from roader.validate.gates import (
    check_continuity,
    check_crossing_order,
    check_provenance,
    check_solids,
    check_voids,
    compare_runs,
    hash_exports,
    run_gates,
)

from .meshes import box, box_arrays, mesh

TRUTH = Provenance(Source.MEASURED, 1.0)


@pytest.fixture
def model():
    return overpass_model()


# ---------------------------------------------------------------- AC-H1


def test_ac_h1_good_and_bad_solids(tmp_path):
    good = check_solids([box("a"), box("b", (1, 0, 0), (2, 1, 1))], val3dity_exe=str(tmp_path / "none"))
    assert not good.passed  # val3dity missing: a skipped oracle fails the gate
    assert good.skipped_reason
    assert good.violations == []

    v, f = box_arrays()
    bad = check_solids([mesh("inv", v, f[:, ::-1]), box("x", (0.5, 0.5, 0.5), (1.5, 1.5, 1.5))], val3dity_exe=str(tmp_path / "none"))
    reqs = {x.req for x in bad.violations}
    assert {"FR-SV-3", "FR-SV-7"} <= reqs


def test_ac_h1_no_meshes_does_not_pass():
    result = check_solids([])
    assert not result.passed and result.skipped_reason


# ---------------------------------------------------------------- AC-H2


def test_ac_h2_reference_by_name_passes(model):
    ref = [
        {"upper_name": "Route 1", "lower_name": "Freeway EB"},
        {"upper_name": "Route 1", "lower_name": "Freeway WB"},
    ]
    result = check_crossing_order(model, ref)
    assert result.passed and result.hard
    assert result.metrics["n_correct"] == 2


def test_ac_h2_reference_by_id_passes(model):
    ref = [{"upper_road": c.upper_road, "lower_road": c.lower_road} for c in model.crossings]
    assert check_crossing_order(model, ref).passed


def test_ac_h2_reversed_order_fails(model):
    ref = [
        {"upper_name": "Freeway EB", "lower_name": "Route 1"},
        {"upper_name": "Route 1", "lower_name": "Freeway WB"},
    ]
    result = check_crossing_order(model, ref)
    assert not result.passed
    assert any("reversed" in v.message for v in result.violations)


def test_ac_h2_unlabelled_crossing_fails(model):
    result = check_crossing_order(model, [{"upper_name": "Route 1", "lower_name": "Freeway EB"}])
    assert not result.passed
    assert any("not in the reference" in v.message for v in result.violations)


def test_ac_h2_without_reference_is_failed_skip(model):
    result = check_crossing_order(model, None)
    assert not result.passed and result.skipped_reason


# ---------------------------------------------------------------- AC-H3


def _curtain(model, lower_index: int) -> Mesh:
    c = model.crossings[lower_index]
    return box("curtain", (c.x - 2.0, c.y - 2.0, 100.5), (c.x + 2.0, c.y + 2.0, 103.0), element_id="curtain")


def test_ac_h3_no_meshes_under_deck_passes(model):
    result = check_voids(model, [])
    assert result.passed, result.violations
    assert result.metrics["n_rays"] > 100
    assert result.metrics["min_clearance_m"] > 4.0


def test_ac_h3_curtain_under_deck_fails(model):
    result = check_voids(model, [_curtain(model, 0)])
    assert not result.passed
    assert any("curtain" in v.message for v in result.violations)


def test_ac_h3_lower_road_body_and_low_terrain_are_ignored(model):
    eb = model.roads[0]
    c = model.crossings[0]
    own = box("eb:pavement", (c.x - 3, c.y - 3, 99.7), (c.x + 3, c.y + 3, 100.0), element_id=eb.id)
    ground = box("terrain", (c.x - 3, c.y - 3, 98.0), (c.x + 3, c.y + 3, 99.0), element_id="terrain")
    assert check_voids(model, [own, ground]).passed


def test_ac_h3_mesh_beside_the_deck_passes(model):
    c = model.crossings[0]
    beside = box("far", (c.x + 80.0, c.y - 2.0, 100.5), (c.x + 84.0, c.y + 2.0, 103.0))
    assert check_voids(model, [beside]).passed


# ---------------------------------------------------------------- AC-H4


def test_ac_h4_synthetic_model_is_continuous(model):
    result = check_continuity(model)
    assert result.passed, result.violations


def test_ac_h4_profile_z0_perturbation_fails(model):
    model.roads[2].profile.elements[1].z0 += 0.05
    result = check_continuity(model)
    assert not result.passed
    assert any("elevation" in v.message for v in result.violations)


def test_ac_h4_grade_break_fails(model):
    el = model.roads[2].profile.elements[2]
    el.g0 = el.g1 = -0.035
    result = check_continuity(model)
    assert any("grade" in v.message for v in result.violations)


def test_ac_h4_alignment_heading_break_fails(model):
    road = model.roads[0]
    e = road.alignment.elements[0]
    road.alignment = Alignment(
        [
            AlignElement("line", 0.0, 300.0, e.x0, e.y0, 0.0, 0.0, 0.0, TRUTH),
            AlignElement("line", 300.0, 300.0, e.x0 + 300.0, e.y0, 0.01, 0.0, 0.0, TRUTH),
        ]
    )
    result = check_continuity(model)
    assert any("heading" in v.message for v in result.violations)


def _continuation(model, dz: float = 0.0, grade: float = 0.0) -> Road:
    eb = model.roads[0]
    x, y, h, _k = eb.alignment.evaluate(eb.length)
    z = eb.profile.z(eb.length) + dz
    return Road(
        id=stable_id("rd", "test", "next"),
        name="Freeway EB next",
        road_class="motorway",
        kind="mainline",
        oneway=True,
        alignment=Alignment([AlignElement("line", 0.0, 200.0, x, y, h, 0.0, 0.0, TRUTH)]),
        profile=Profile([VerticalElement(0.0, 200.0, z, grade, grade, TRUTH)]),
        cross_section=eb.cross_section,
    )


def test_ac_h4_link_matching_ends_passes(model):
    nxt = _continuation(model)
    model.roads.append(nxt)
    model.links.append(Link("lk1", model.roads[0].id, "end", nxt.id, "start", "continue"))
    assert check_continuity(model).passed


def test_ac_h4_link_elevation_step_fails(model):
    nxt = _continuation(model, dz=0.02)
    model.roads.append(nxt)
    model.links.append(Link("lk1", model.roads[0].id, "end", nxt.id, "start", "continue"))
    result = check_continuity(model)
    assert any(v.element_id == "lk1" and "elevation" in v.message for v in result.violations)


def test_ac_h4_link_grade_step_fails(model):
    nxt = _continuation(model, grade=0.002)
    model.roads.append(nxt)
    model.links.append(Link("lk1", model.roads[0].id, "end", nxt.id, "start", "continue"))
    result = check_continuity(model)
    assert any(v.element_id == "lk1" and "grade" in v.message for v in result.violations)


# ---------------------------------------------------------------- AC-H5


def test_ac_h5_identical_runs_pass_and_differences_fail(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "sub").mkdir(parents=True)
        (root / "model.json").write_bytes(b'{"x":1}')
        (root / "sub" / "deck.glb").write_bytes(b"\x00\x01")
    assert hash_exports(a) == hash_exports(b)
    assert list(hash_exports(a)) == ["model.json", "sub/deck.glb"]
    assert compare_runs(a, b).passed

    (b / "sub" / "deck.glb").write_bytes(b"\x00\x02")
    (b / "extra.txt").write_text("x")
    result = compare_runs(a, b)
    assert not result.passed
    messages = " ".join(v.message for v in result.violations)
    assert "differs" in messages and "only in" in messages


# ---------------------------------------------------------------- AC-H6


def _mesh_for(element_id: str) -> Mesh:
    v, f = box_arrays()
    return mesh("m", v, f, element_id=element_id)


def test_ac_h6_synthetic_model_has_complete_provenance(model):
    result = check_provenance(model, [_mesh_for(model.structures[0].id)])
    assert result.passed, result.violations
    assert result.metrics["n_prov"] > 20


def test_ac_h6_missing_prov_fails(model):
    model.structures[0].supports[1].prov = None
    result = check_provenance(model, [])
    assert not result.passed
    assert any("supports[1]" in v.message for v in result.violations)


def test_ac_h6_bad_mesh_tags_fail(model):
    orphan = _mesh_for("no-such-element")
    blank = _mesh_for("")
    result = check_provenance(model, [orphan, blank])
    assert len(result.violations) == 2


def test_ac_h6_knot_prov_checked(model):
    model.roads[0].cross_section.width_left.knots[0].prov = None
    assert not check_provenance(model, []).passed


# ---------------------------------------------------------------- report


def test_run_gates_collects_every_hard_gate(model, tmp_path):
    report = run_gates(model, [], crossings_ref=None, val3dity_exe=str(tmp_path / "none"))
    gates = {r.gate for r in report.results}
    assert {"AC-H1", "AC-H2", "AC-H3", "AC-H4", "AC-H5", "AC-H6"} <= gates
    assert not report.publishable  # no reference list, no second run, no val3dity
    by_gate = {r.gate: r for r in report.results}
    assert by_gate["AC-H3"].passed and by_gate["AC-H4"].passed and by_gate["AC-H6"].passed
