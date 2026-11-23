"""An unobserved road end carries the observed grade; it never gets a grade of its own (FR-OC-1, FR-AF-2, D-146)."""

from __future__ import annotations

import numpy as np
import pytest

from roader.fit.profile import fit_profile, to_profile
from roader.rules import RuleBook

LENGTH = 464.0


def _samples(head_gap: float, flip: bool = False, seed: int = 1):
    """Returns from ``head_gap`` on: a 0.3 m rise over the first 10 m observed (a cross street's crown), then a 0.02 % grade."""
    rng = np.random.default_rng(seed)
    s = np.arange(head_gap, LENGTH, 0.5)
    rise = 0.5 * np.maximum(0.0, head_gap + 10.0 - s) ** 2 / 160.0
    z = 0.06 + rise + 0.0002 * np.maximum(s - head_gap - 13.0, 0.0) + rng.normal(0.0, 0.02, s.size)
    if flip:
        s = LENGTH - s
    return s, z


@pytest.mark.parametrize("flip", [False, True])
def test_fr_oc_1_unobserved_end_is_carried_not_bent(flip):
    """Governor Nicholls, New Orleans: the first 27 m under a deck and a holdout block took a -0.11 grade, 3.7 m at the end."""
    s, z = _samples(27.0, flip)
    v, _rep, _bins = fit_profile(s, z, LENGTH, rules=RuleBook.load(), road_class="residential")
    p = to_profile(v, LENGTH)
    end, first = (0.0, 27.0) if not flip else (LENGTH, LENGTH - 27.0)
    assert abs(p.z(end) - p.z(first)) < 0.35, (p.z(end), p.z(first))
    assert np.sqrt(np.mean((np.array([p.z(float(a)) for a in s]) - z) ** 2)) < 0.1


def test_fr_af_2_a_short_unobserved_end_keeps_its_fit(monkeypatch):
    """Under MIN_END_TANGENT_M unobserved at the end, the rule does not touch the fit."""
    from roader.fit import profile

    s, z = _samples(5.0)
    v, _rep, _bins = fit_profile(s, z, LENGTH, rules=RuleBook.load(), road_class="residential")
    monkeypatch.setattr(profile, "MIN_END_TANGENT_M", 1e9)
    w, _rep, _bins = fit_profile(s, z, LENGTH, rules=RuleBook.load(), road_class="residential")
    assert v.z0 == w.z0 and all(np.array_equal(getattr(v, f), getattr(w, f)) for f in ("grades", "pvi", "lengths"))
