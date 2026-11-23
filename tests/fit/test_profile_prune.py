"""FR-AF-2: PVI pruning must not compound (Woburn loop ramp 9522025 kept one curve over 604 m).

Each removal is judged against the first fit, not against the last accepted one,
so a chain of individually cheap removals cannot flatten a rolling profile.
"""

from __future__ import annotations

import numpy as np

from roader.fit import profile as fp
from roader.rules import RuleBook
from tests.fit import scenes

RULES = RuleBook.load()


def test_fr_af_2_rolling_profile_keeps_its_curves():
    pr = scenes.profile(30.0, 0.01, [(80.0, 0.01), (60.0, -0.012), (90.0, -0.012), (60.0, 0.006), (90.0, 0.006), (60.0, -0.01), (160.0, -0.01)])
    rng = np.random.default_rng(4)
    s = np.sort(rng.uniform(0.0, pr.length, 12000))
    z = np.array([pr.z(float(a)) for a in s]) + rng.normal(0.0, 0.03, s.size)
    vert, _rep, _bins = fp.fit_profile(s, z, pr.length, rules=RULES, road_class="motorway_link")
    st = np.arange(0.0, pr.length, 5.0)
    err = np.abs(vert.z(st) - np.array([pr.z(float(a)) for a in st]))
    assert len(vert.pvi) >= 3 and float(np.max(err)) < 0.05, (vert.pvi, float(np.max(err)))


def test_fr_af_2_a_poorer_refit_does_not_replace_the_profile():
    """fit_road's later passes keep the earlier profile when the refit follows the same returns worse (ramp 9522025)."""
    from roader.fit.road import _better_profile

    s = np.linspace(0.0, 400.0, 2000)
    good = fp._Vert(10.0, np.array([0.01, -0.01]), np.array([200.0]), np.array([80.0]))
    flat = fp._Vert(10.0, np.array([0.0]), np.zeros(0), np.zeros(0))
    z = good.z(s)
    assert _better_profile((flat, None, None), (good, None, None), s, z)[0] is good
    assert _better_profile((good, None, None), (flat, None, None), s, z)[0] is good
    assert _better_profile(None, (good, None, None), s, z)[0] is good
