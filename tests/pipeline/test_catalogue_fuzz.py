"""Seeded fuzz over the catalogue scene parameters (section 9, AC-H1..AC-H6, FR-PL-3).

Per type, scenes are drawn from ``scenes.PARAM_RANGES`` (skew up to 60 degrees where the
type allows, loop and flyover radii, street and ramp spacing, level count). The pipeline,
the solids and the gates must never raise (a crash is a failure, robust track), and every
hard gate that fails must name the elements it failed on.

Seeds per type: ``ROADER_FUZZ_SEEDS`` (default 1). Expected crashes are recorded for seeds
1 to 3; a crash on a higher seed is a new finding. Heavy types run only with
``ROADER_CATALOGUE=all`` (see ``test_catalogue_scenes``).
"""

from __future__ import annotations

import os

import pytest

from tests.pipeline import scenes as S
from tests.pipeline.test_catalogue_scenes import FAST, FULL, HEAVY_REASON

SEEDS = int(os.environ.get("ROADER_FUZZ_SEEDS", "1"))

# (scene type, seed) -> "track: what raises today".
SKEWED_LINE = "robust: solids raise 'skewed station line leaves the road'"
CRASHES: dict[tuple[str, int], str] = {
    ("parclo", 3): f"{SKEWED_LINE} at the start of a loop leaving the street",
    ("trumpet", 2): f"{SKEWED_LINE} at a ramp start on the stem end",
    ("directional", 2): f"{SKEWED_LINE} at a flyover deck end",
    ("directional", 3): f"{SKEWED_LINE} at a road start",
    ("stack", 1): f"{SKEWED_LINE} on a stacked level ramp (levels=4)",
    ("stack", 2): f"{SKEWED_LINE} on a stacked level ramp (levels=4)",
    ("stack", 3): f"{SKEWED_LINE} on a stacked level ramp (levels=4)",
}


def _cases():
    for name in S.SCENES:
        for seed in range(1, SEEDS + 1):
            marks = []
            if name not in FAST and not FULL:
                marks.append(pytest.mark.skip(reason=HEAVY_REASON))
            if (name, seed) in CRASHES:
                marks.append(pytest.mark.xfail(strict=True, reason=CRASHES[(name, seed)]))
            yield pytest.param(name, seed, id=f"{name}-seed{seed}", marks=marks)


@pytest.mark.parametrize(("name", "seed"), list(_cases()))
def test_fuzzed_scene_never_raises_and_failing_gates_name_elements(name, seed):
    params = S.fuzz_params(name, seed)
    out = S.run_scene(S.SCENES[name](**params))
    if out.error:
        pytest.fail(f"{name} {params}: {out.stage} raised: {out.error}")
    for gate, result in sorted(out.gates.items()):
        if not result.hard or result.passed:
            continue
        if result.skipped_reason and not result.violations:
            continue  # did not run (val3dity off in the fuzz), nothing to name
        assert result.violations, f"{name} {params}: {gate} failed without naming anything"
        assert all(v.element_id for v in result.violations), f"{name} {params}: {gate} has a violation with no element"
