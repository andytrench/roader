"""AC-H1 oracle: val3dity harness. Needs the binary via $VAL3DITY or PATH."""

from __future__ import annotations

import pytest

from roader.validate.val3dity import find_val3dity, run_val3dity

from .meshes import box, box_arrays, mesh

needs_val3dity = pytest.mark.skipif(find_val3dity() is None, reason="val3dity not installed (scripts/install_val3dity.sh)")


def test_ac_h1_missing_val3dity_is_a_failed_skip(tmp_path):
    result = run_val3dity([box()], exe=str(tmp_path / "no-such-val3dity"))
    assert result.hard
    assert not result.passed
    assert result.skipped_reason


@needs_val3dity
def test_ac_h1_val3dity_accepts_good_box():
    result = run_val3dity([box("a"), box("b", (5123.25, -8871.5, 101.0), (5135.0, -8860.0, 101.3))])
    assert result.passed, result.violations
    assert result.metrics["n_primitives"] == 2


@needs_val3dity
def test_ac_h1_val3dity_rejects_inverted_and_open_boxes():
    v, f = box_arrays()
    result = run_val3dity([box("good"), mesh("inverted", v, f[:, ::-1]), mesh("open", v, f[1:])])
    assert not result.passed
    by_mesh = {(x.element_id, x.req) for x in result.violations}
    assert ("inverted", "FR-SV-3") in by_mesh
    assert ("open", "FR-SV-1") in by_mesh
    assert not any(x.element_id == "good" for x in result.violations)


@needs_val3dity
def test_ac_h1_val3dity_open_surface_as_composite_surface():
    v, f = box_arrays()
    result = run_val3dity([mesh("terrain", v, f[:2], closed=False)])
    assert result.passed, result.violations
