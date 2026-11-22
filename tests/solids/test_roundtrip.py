"""Generation from a model loaded from canonical JSON (D-011 rounding) is valid and stable (AC-H5, FR-SG-6)."""

from __future__ import annotations

import hashlib

import pytest

from roader.model import dumps, loads
from roader.model.synthetic import overpass_model
from roader.solids import generate
from roader.validate.gates import check_solids
from roader.validate.val3dity import find_val3dity

from . import meshcheck as mc
from .test_curved import TOL, curved_model
from .test_overpass import check_terrain

SCENES = {"overpass": (overpass_model, 0.01), "curved": (curved_model, TOL)}


def _digest(meshes) -> str:
    h = hashlib.sha256()
    for m in meshes:
        h.update(m.id.encode())
        h.update(m.vertices.tobytes())
        h.update(m.faces.tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module", params=sorted(SCENES))
def scene(request):
    build, tol = SCENES[request.param]
    model = loads(dumps(build()))
    return request.param, build, tol, model, generate(model, chord_tol_m=tol)


def test_round_tripped_model_generates_valid_solids(scene):
    _name, _build, tol, model, meshes = scene
    closed = [m for m in meshes if m.closed]
    for m in closed:
        assert mc.solid_problems(m) == [], m.id
    assert mc.crossing_pairs(closed) == []
    check_terrain(model, meshes, tol)


def test_round_trip_is_stable_ac_h5(scene):
    _name, build, tol, model, meshes = scene
    digest = _digest(meshes)
    assert _digest(generate(model, chord_tol_m=tol)) == digest
    assert _digest(generate(loads(dumps(build())), chord_tol_m=tol)) == digest
    assert _digest(generate(loads(dumps(model)), chord_tol_m=tol)) == digest


def test_round_trip_keeps_ids(scene):
    _name, build, tol, _model, meshes = scene
    assert [m.id for m in meshes] == [m.id for m in generate(build(), chord_tol_m=tol)]


def test_round_tripped_overpass_passes_ac_h1():
    if find_val3dity() is None:
        pytest.skip("val3dity not installed")
    result = check_solids(generate(loads(dumps(overpass_model()))))
    assert result.passed, [f"{v.req} {v.element_id}: {v.message}" for v in result.violations[:10]]
