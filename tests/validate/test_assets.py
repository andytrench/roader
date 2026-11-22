"""AC-P2 transitions, AC-P3 / FR-AS-2 triangle budgets, AC-P4 / FR-AS-4 UV stretch.

``Mesh.uv0`` is a milestone 2 contract (lane ``model`` / ``solids``); until the
dataclass has it, tests set it as an attribute. The generated-mesh UV test skips
until ``roader.solids.generate`` produces uv0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import roader.solids
from roader.model.geometry import Knot, Piecewise, VerticalElement
from roader.model.provenance import Provenance, Source
from roader.model.schema import LaneGroup
from roader.model.synthetic import overpass_model
from roader.validate.assets import (
    FREEWAY_BUDGET_PER_KM,
    OVERPASS_BUDGET,
    check_budgets,
    check_transitions,
    check_uv_stretch,
)

from .meshes import mesh

TRUTH = Provenance(Source.MEASURED, 1.0)
generate = getattr(roader.solids, "generate", None)


@pytest.fixture
def model():
    return overpass_model()


def _msgs(result) -> list[str]:
    return [v.message for v in result.violations]


# ---------------------------------------------------------------- AC-P2


def test_ac_p2_synthetic_model_passes(model):
    result = check_transitions(model)
    assert result.passed and not result.hard, _msgs(result)


def test_ac_p2_width_taper_passes_and_short_change_fails(model):
    xs = model.roads[0].cross_section
    xs.width_left = Piecewise([Knot(100.0, 5.5, TRUTH), Knot(160.0, 9.2, TRUTH)])
    assert check_transitions(model).passed
    xs.width_left = Piecewise([Knot(100.0, 5.5, TRUTH), Knot(100.5, 9.2, TRUTH)])
    assert any("width_left" in m and "step" in m for m in _msgs(check_transitions(model)))


def test_ac_p2_coincident_knots_are_a_step(model):
    pw = model.roads[0].cross_section.slope_right
    pw.knots.append(Knot(pw.knots[-1].s, 0.06, TRUTH))  # bypasses the constructor check
    assert any("slope_right" in m and "do not increase" in m for m in _msgs(check_transitions(model)))


def test_ac_p2_lane_width_step_between_groups_fails(model):
    xs = model.roads[0].cross_section
    xs.lanes = [LaneGroup("right", 0.0, 300.0, 2, 3.66, 1.2, 3.0, TRUTH), LaneGroup("right", 300.0, 600.0, 2, 3.3, 1.2, 3.0, TRUTH)]
    assert any("lane width steps" in m for m in _msgs(check_transitions(model)))


def test_ac_p2_grade_step_and_short_vertical_curve_fail(model):
    over = model.roads[2]
    over.profile.elements[1] = VerticalElement(over.profile.elements[1].s0, 0.5, over.profile.elements[1].z0, 0.03, -0.03, TRUTH)
    msgs = _msgs(check_transitions(model))
    assert any("vertical curve 1" in m for m in msgs), msgs
    el = over.profile.elements[2]
    over.profile.elements[2] = VerticalElement(el.s0, el.length, el.z0, -0.02, -0.02, TRUTH)
    assert any("grade steps" in m for m in _msgs(check_transitions(model)))


# ---------------------------------------------------------------- AC-P3


def _fan(mesh_id: str, element_id: str, n: int, at: tuple[float, float]) -> object:
    """n tiny open triangles at plan point ``at`` (only counts and centroids matter)."""
    k = np.arange(n + 1)
    v = np.column_stack([at[0] + 1e-3 * np.cos(k / n), at[1] + 1e-3 * np.sin(k / n), np.full(n + 1, 100.0)])
    v = np.vstack([[at[0], at[1], 100.0], v])
    f = np.column_stack([np.zeros(n, np.int64), k[:-1] + 1, k[1:] + 1])
    m = mesh(mesh_id, v, f, closed=False, element_id=element_id)
    return m


def test_ac_p3_counts_against_four_lane_km(model):
    eb = model.roads[0]  # 600 m, 2 lanes: 0.3 four-lane km
    n = int(0.9 * FREEWAY_BUDGET_PER_KM * 0.3)
    result = check_budgets(model, [_fan("eb", eb.id, n, eb.alignment.offset_point(100.0, 0.0))])
    assert result.metrics[f"tri_per_4lane_km:{eb.id}"] == pytest.approx(n / 0.3)
    assert result.passed, _msgs(result)


def test_ac_p3_threefold_overrun_is_a_defect(model):
    eb = model.roads[0]
    n = int(3.2 * FREEWAY_BUDGET_PER_KM * 0.3)
    result = check_budgets(model, [_fan("eb", eb.id, n, eb.alignment.offset_point(100.0, 0.0))])
    assert any(v.element_id == eb.id and "defect" in v.message for v in result.violations)


def test_ac_p3_overpass_includes_substructure(model):
    st = model.structures[0]
    over = model.road(st.carried_road)
    at = over.alignment.offset_point(0.5 * (st.s_start + st.s_end), 0.0)
    parts = [_fan("deck", st.id, 20_000, at), _fan("pier", st.supports[1].id, 20_000, at), _fan("par", over.cross_section.barriers[0].id, 30_000, at)]
    result = check_budgets(model, parts)
    assert result.metrics[f"tri_per_overpass:{st.id}"] == 70_000
    assert result.metrics["max_ratio"] == pytest.approx(70_000 / OVERPASS_BUDGET)
    assert result.passed  # median of (0, 0, 1.17) is 0


def test_ac_p3_generated_overpass_meets_budgets(model):
    if generate is None:
        pytest.skip("roader.solids.generate not implemented yet")
    result = check_budgets(model, generate(model))
    print("AC-P3 actuals", result.metrics)
    assert result.passed, _msgs(result)
    assert result.metrics["n_assets"] == 3.0


# ---------------------------------------------------------------- AC-P4


def _strip(radius: float | None, half: float = 5.0, length: float = 40.0, n: int = 20):
    """Pavement-like strip: straight (radius None) or on an arc; uv0 = (centre-line station, offset)."""
    s = np.linspace(0.0, length, n + 1)
    u = np.array([-half, 0.0, half])
    ss, uu = np.meshgrid(s, u, indexing="ij")
    if radius is None:
        x, y = ss, uu
    else:
        th = ss / radius
        x, y = (radius - uu) * np.sin(th), radius - (radius - uu) * np.cos(th)
    v = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 100.0)])
    faces = []
    for i in range(n):
        for j in range(2):
            a, b, c, d = i * 3 + j, (i + 1) * 3 + j, (i + 1) * 3 + j + 1, i * 3 + j + 1
            faces += [(a, b, c), (a, c, d)]
    m = mesh("strip", v, np.array(faces), closed=False)
    m.uv0 = np.column_stack([ss.ravel(), uu.ravel()])
    return m


def test_ac_p4_straight_strip_has_no_stretch():
    result = check_uv_stretch([_strip(None)])
    assert result.passed, _msgs(result)
    assert result.metrics["max_stretch_along"] < 1e-9 and result.metrics["max_stretch_across"] < 1e-9


def test_ac_p4_gentle_curve_passes_tight_curve_fails():
    """Centre-line station at 5 m offset stretches by u/R: 2.5% at R=200 m, 10% at R=50 m."""
    gentle = check_uv_stretch([_strip(200.0)])
    assert gentle.passed and 0.02 < gentle.metrics["max_stretch_along"] < 0.03
    tight = check_uv_stretch([_strip(50.0)])
    assert not tight.passed and any("stretch along" in m for m in _msgs(tight))


def test_ac_p4_scaled_offset_fails_across():
    m = _strip(None)
    m.uv0 = m.uv0 * np.array([1.0, 1.1])
    result = check_uv_stretch([m])
    assert not result.passed and result.metrics["max_stretch_across"] == pytest.approx(1 - 1 / 1.1, rel=1e-6)


def test_ac_p4_degenerate_uv_fails():
    m = _strip(None)
    m.uv0 = m.uv0.copy()
    m.uv0[:, 1] = 0.0
    assert any("zero uv0 area" in msg for msg in _msgs(check_uv_stretch([m])))


def test_ac_p4_without_uv0_is_a_soft_skip():
    result = check_uv_stretch([mesh("m", np.eye(3), np.array([[0, 1, 2]]), closed=False)])
    assert not result.passed and not result.hard and result.skipped_reason


def test_ac_p4_generated_uvs_have_low_stretch(model):
    if generate is None:
        pytest.skip("roader.solids.generate not implemented yet")
    meshes = generate(model)
    if not any(getattr(m, "uv0", None) is not None for m in meshes):
        pytest.skip("roader.solids.generate does not produce Mesh.uv0 yet (FR-AS-4, milestone 2)")
    result = check_uv_stretch(meshes)
    assert result.passed, _msgs(result)[:10]
    assert math.isfinite(result.metrics["max_stretch_along"])
