"""FR-SV checks on realistic swept slabs: no false positives, real defects found."""

from __future__ import annotations

import numpy as np
import pytest

from roader.validate.profile import check_profile
from roader.validate.solid import check_solid
from roader.validate.val3dity import find_val3dity, run_val3dity

from .sweeps import slab


def test_fr_sv_swept_slab_is_valid():
    assert check_solid(slab(np.linspace(0.0, 400.0, 401))) == []


def test_fr_sv_7_slabs_split_on_a_shared_station_conform():
    a = slab(np.linspace(0.0, 200.0, 201), "a")
    b = slab(np.linspace(200.0, 400.0, 201), "b")
    assert check_solid(a) == [] and check_solid(b) == []
    assert check_profile([a, b]) == []


def test_fr_sv_7_overlapping_slab_segments_fail():
    a = slab(np.linspace(0.0, 200.0, 201), "a")
    b = slab(np.linspace(150.0, 400.0, 251), "b")
    assert check_profile([a, b])


def test_fr_sv_4_slab_folded_back_on_itself_fails():
    stations = np.concatenate([np.linspace(0.0, 100.0, 101), [60.0]])
    assert "FR-SV-4" in {v.req for v in check_solid(slab(stations))}


@pytest.mark.skipif(find_val3dity() is None, reason="val3dity not installed")
def test_ac_h1_val3dity_accepts_swept_slabs():
    a = slab(np.linspace(0.0, 200.0, 201), "a")
    b = slab(np.linspace(200.0, 400.0, 201), "b")
    result = run_val3dity([a, b])
    assert result.passed, result.violations
