"""FR-SV-7 in the real-time export profile: embed by the declared minimum, no coplanar overlap.

``roader.solids.generate(model, profile="realtime")`` is the milestone 2 contract
(lane ``assets``) exercised end to end at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

import roader.solids
from roader.model.synthetic import overpass_model
from roader.validate.gates import check_solids
from roader.validate.realtime import check_realtime, embed_depth

from .meshes import box


def _reqs(violations) -> list[str]:
    return [v.message for v in violations]


def test_overlap_deeper_than_minimum_passes():
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (0.7, -0.5, -0.5), (1.7, 1.5, 1.5))
    assert check_realtime([a, b]) == []


def test_overlap_shallower_than_minimum_fails():
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (0.9, -0.5, -0.5), (1.9, 1.5, 1.5))
    msgs = _reqs(check_realtime([a, b]))
    assert any("embed only 0.100" in m for m in msgs), msgs


def test_declared_minimum_is_a_parameter():
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (0.9, -0.5, -0.5), (1.9, 1.5, 1.5))
    assert check_realtime([a, b], min_embed_m=0.05) == []


def test_face_to_face_contact_is_coplanar_overlap():
    """Analytic-style conforming contact z-fights in the real-time profile."""
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (1, 0, 0), (2, 1, 1))
    msgs = _reqs(check_realtime([a, b]))
    assert any("coplanar" in m for m in msgs), msgs


def test_flush_faces_on_embedded_parts_fail():
    """Embedding is fine, but faces left flush with the neighbour's faces z-fight."""
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (0.7, 0, 0), (1.7, 1, 1))
    msgs = _reqs(check_realtime([a, b]))
    assert any("coplanar" in m for m in msgs), msgs


def test_barrier_sunk_into_slab_exactly_the_minimum_passes():
    slab = box("slab", (0, 0, 0), (10, 4, 0.3))
    barrier = box("barrier", (0.5, 1, 0.1), (9.5, 1.5, 1.0))
    assert check_realtime([slab, barrier]) == []
    assert embed_depth(barrier, slab, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])) == pytest.approx(0.2, abs=1e-9)


def test_barrier_sunk_less_than_minimum_fails():
    slab = box("slab", (0, 0, 0), (10, 4, 0.3))
    barrier = box("barrier", (0.5, 1, 0.15), (9.5, 1.5, 1.0))
    msgs = _reqs(check_realtime([slab, barrier]))
    assert any("embed only 0.150" in m for m in msgs), msgs


def test_thin_post_sunk_the_minimum_passes():
    """Embed depth is the pull that separates the parts, not the width of the overlap."""
    deck = box("deck", (0, 0, 0), (4, 4, 1))
    post = box("post", (1, 1, 0.8), (1.1, 1.1, 2.0))
    assert check_realtime([deck, post]) == []


def test_separate_parts_pass():
    assert check_realtime([box("a"), box("b", (3, 0, 0), (4, 1, 1))]) == []


def test_check_solids_selects_the_profile(tmp_path):
    a = box("a", (0, 0, 0), (1, 1, 1))
    b = box("b", (0.7, -0.5, -0.5), (1.7, 1.5, 1.5))
    exe = str(tmp_path / "none")
    analytic = check_solids([a, b], val3dity_exe=exe)
    realtime = check_solids([a, b], val3dity_exe=exe, profile="realtime")
    assert analytic.gate == "AC-H1" and any(v.req == "FR-SV-7" for v in analytic.violations)
    assert realtime.gate == "AC-H1:realtime" and realtime.violations == []
    with pytest.raises(ValueError):
        check_solids([a], profile="lod")


# ---------------------------------------------------------------- generated (contract)


def test_generated_realtime_profile_passes():
    model = overpass_model()
    meshes = roader.solids.generate(model, chord_tol_m=0.01, profile="realtime")
    result = check_solids(meshes, profile="realtime")
    native = [v for v in result.violations if not v.message.startswith("val3dity")]
    assert native == [], [f"{v.req} {v.element_id}: {v.message}" for v in native[:20]]
