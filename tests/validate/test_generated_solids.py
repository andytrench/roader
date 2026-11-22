"""Milestone 1 acceptance: generated solids of the synthetic overpass pass AC-H1 and AC-H3.

Written before ``roader.solids.generate`` exists (spec section 0: tests first);
skipped until the solids lane lands it.
"""

from __future__ import annotations

import pytest

import roader.solids
from roader.model.synthetic import overpass_model
from roader.validate.gates import check_provenance, check_solids, check_voids
from roader.validate.val3dity import find_val3dity

generate = getattr(roader.solids, "generate", None)
pytestmark = pytest.mark.skipif(generate is None, reason="roader.solids.generate not implemented yet")


@pytest.fixture(scope="module")
def scene():
    model = overpass_model()
    return model, generate(model)


def _report(result) -> str:
    lines = [f"{v.req} {v.element_id}: {v.message}" for v in result.violations[:20]]
    return "\n".join([result.skipped_reason, *lines])


def test_ac_h1_generated_solids_native_and_val3dity(scene):
    if find_val3dity() is None:
        pytest.skip("val3dity not installed: AC-H1 cannot pass without the oracle (scripts/install_val3dity.sh)")
    _model, meshes = scene
    assert meshes
    result = check_solids(meshes)
    assert result.passed, _report(result)


def test_ac_h3_generated_solids_leave_voids_open(scene):
    model, meshes = scene
    result = check_voids(model, meshes)
    assert result.passed, _report(result)


def test_ac_h6_generated_meshes_name_model_elements(scene):
    model, meshes = scene
    result = check_provenance(model, meshes)
    assert result.passed, _report(result)
