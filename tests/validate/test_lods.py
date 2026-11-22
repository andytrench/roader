"""FR-AS-1 LOD chain checks: levels, validity per level, voids, deck elevation, triangle counts.

``roader.solids.generate_lods(model)`` is a milestone 2 contract (lane
``solids``); its end-to-end test skips until it exists. The other tests use the
synthetic overpass structure generated at each level's chord tolerance as a
stand-in chain (structure meshes only, to keep the per-level FR-SV checks fast).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import roader.solids
from roader.model.synthetic import overpass_model
from roader.validate.lod import LOD_CHORD_TOL, check_lods

from .meshes import box

generate = getattr(roader.solids, "generate", None)


@pytest.fixture(scope="module")
def scene():
    if generate is None:
        pytest.skip("roader.solids.generate not implemented yet")
    model = overpass_model()
    lods = {k: [m for m in generate(model, chord_tol_m=tol) if m.layer == "L5"] for k, tol in LOD_CHORD_TOL.items()}
    return model, lods


def _copy(lods):
    return {k: list(v) for k, v in lods.items()}


def _messages(result) -> list[str]:
    return [v.message for v in result.violations]


def test_tolerances_are_the_spec_values():
    assert LOD_CHORD_TOL == {0: 0.01, 1: 0.05, 2: 0.25, 3: 1.0}


def test_stand_in_chain_passes(scene):
    model, lods = scene
    result = check_lods(model, lods)
    assert result.passed, _messages(result)[:10]
    assert result.metrics["n_rays_lod0"] > 100
    assert result.metrics["deck_dz_max_lod0"] <= 0.01


def test_missing_level_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    del lods[3]
    assert any("differ from the required" in m for m in _messages(check_lods(model, lods)))


def test_triangle_count_growing_with_level_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    lods[3].append(box("extra", (500, 500, 0), (501, 501, 1)))
    assert any("LOD 3" in m and "more than" in m for m in _messages(check_lods(model, lods)))


def test_invalid_solid_at_a_level_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    bad = lods[2][0]
    lods[2][0] = dataclasses.replace(bad, faces=bad.faces[:-1])
    result = check_lods(model, lods)
    assert any(v.req == "FR-SV-1" and v.message.startswith("LOD 2") for v in result.violations)


def test_blocked_void_at_a_level_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    c = model.crossings[0]
    lods[1].append(box("curtain", (c.x - 2, c.y - 2, 100.5), (c.x + 2, c.y + 2, 103.0), element_id="curtain"))
    assert any(m.startswith("LOD 1") and "curtain" in m for m in _messages(check_lods(model, lods)))


def test_deck_moved_beyond_level_tolerance_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    st = model.structures[0]
    k = next(n for n, m in enumerate(lods[1]) if m.element_id == st.id and m.part == "deck")
    deck = lods[1][k]
    lods[1][k] = dataclasses.replace(deck, vertices=deck.vertices + np.array([0.0, 0.0, 0.08]))
    msgs = _messages(check_lods(model, lods))
    assert any(m.startswith("LOD 1") and "deck top off the model" in m for m in msgs), msgs


def _proxy(model):
    st = model.structures[0]
    road = model.road(st.carried_road)
    pts = np.array([road.alignment.offset_point(s, u) for s in (st.s_start - 3, st.s_end + 3) for u in (-8.0, 8.0)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    top = road.surface_z(0.5 * (st.s_start + st.s_end), 0.0)
    return box(f"{st.id}:proxy", (lo[0], lo[1], 98.0), (hi[0], hi[1], top), element_id=st.id)


def test_coarsest_proxy_closes_the_void_legitimately(scene):
    model, lods = scene
    lods = _copy(lods)
    proxy = _proxy(model)
    proxy.part = "proxy"
    lods[3] = [proxy]
    result = check_lods(model, lods)
    assert result.passed, _messages(result)[:10]


def test_proxy_before_coarsest_level_fails(scene):
    model, lods = scene
    lods = _copy(lods)
    proxy = _proxy(model)
    proxy.part = "proxy"
    lods[2] = [proxy]
    lods[3] = [proxy]
    msgs = _messages(check_lods(model, lods))
    assert any(m.startswith("LOD 2") and "proxy before the coarsest" in m for m in msgs), msgs


def test_non_proxy_block_at_coarsest_level_still_tests_voids(scene):
    model, lods = scene
    lods = _copy(lods)
    lods[3] = [_proxy(model)]  # part "deck": not declared a proxy, so AC-H3 still applies
    assert any(m.startswith("LOD 3") and "start inside" in m for m in _messages(check_lods(model, lods)))


# ---------------------------------------------------------------- generated (contract)


def test_generated_lod_chain_passes():
    generate_lods = getattr(roader.solids, "generate_lods", None)
    if generate_lods is None:
        pytest.skip("roader.solids.generate_lods(model) not implemented yet (lane solids, milestone 2)")
    model = overpass_model()
    result = check_lods(model, generate_lods(model))
    assert result.passed, _messages(result)[:20]
