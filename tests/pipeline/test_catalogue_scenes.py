"""Synthetic interchange catalogue through the pipeline, solids and gates (section 9).

One parametric scene per in-scope type (``tests.pipeline.scenes``), each run once per
session: lidar and OSM-like vectors from the truth model, ``run_pipeline`` with the
D-018 holdout, solids in the analytic profile, then every hard gate AC-H1..AC-H6
(AC-H5 by two independent renderings, D-115) and AC-A1, AC-A4, AC-A5 against the scene
truth. A type that fails a gate today is ``xfail(strict=True)`` naming its track, so the
lane that fixes it flips the mark:

* parallel: overlapping parallel pavements, tapers, ramps riding on their parent;
* decks: junction areas on decks, stacked levels, supports with nowhere to stand, viaducts, the SPUI deck;
* terminals: ramps meeting the crossing street at grade;
* robust: a stage raises.

The full catalogue takes about ten minutes (the pipeline dominates), so by default only
the fast types in ``FAST`` run; ``ROADER_CATALOGUE=all`` runs every type.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest

from roader.validate.val3dity import find_val3dity
from tests.pipeline import scenes as S

GATES = ("FR-EX-1", "AC-H1", "AC-H2", "AC-H3", "AC-H4", "AC-H5", "AC-H6", "AC-A1", "AC-A4", "AC-A5")
FAST = frozenset({"simple_overpass", "twin_different_elevations", "stack"})
FULL = os.environ.get("ROADER_CATALOGUE", "").strip().lower() == "all"
HEAVY_REASON = "heavy catalogue scene (about a minute): set ROADER_CATALOGUE=all"

# (scene, gate) -> "track: what fails today". (scene, None) covers every gate (the scene does not get that far).
KNOWN: dict[tuple[str, str | None], str] = {
    ("diamond", "AC-H1"): "terminals: the exit and entry ways chained through the terminal node pass through the street pavement (FR-SV-7)",
    ("spui", "AC-H1"): "decks: parapets of the ramps and the wide street deck pass through each other where the ramps meet it",
    ("spui", "AC-H2"): "decks: ramps meeting the street at grade on its deck become crossings over it",
    ("spui", "AC-H3"): "decks: the false ramp-over-street crossings have no clearance; rays start inside parapets",
    ("spui", "AC-A1"): "decks: surface RMS 0.8 m where the ramps join the wide deck",
    ("parclo", "AC-H1"): "terminals: loop merging onto the street (no street parent, D-051) overlaps its pavement (FR-SV-7)",
    ("parclo", "AC-H4"): "terminals: seam of the loop joining the street off by 2 cm and 0.25 % grade",
    ("parclo", "AC-A1"): "terminals: surface RMS 0.18 m around the loop joining the street",
    ("cloverleaf", "AC-H1"): "decks: embankment from the I-88 deck end self-intersects beside the loop gore",
    ("cloverleaf", "AC-A5"): "decks: I-88 deck end 3 m off beside the diverging loop",
    ("directional", "AC-H1"): "decks: embankment from an I-88 deck end self-intersects beside the flyover",
    ("directional", "AC-A5"): "decks: stacked-level deck ends 20-26 m off (flyover over the I-88 deck)",
    ("spui", "FR-EX-1"): "decks: a pier column falls outside its cap on a ramp deck at the wide street deck; the ramp is swept with square structures",
    ("spui", "AC-A5"): "decks: the wide street deck and the entry-ramp decks are not found as their own structures; deck ends 50 m off",
    ("trumpet", "FR-EX-1"): "parallel: where the semi-directional and the outer ramp share their first metres, a junction seam is off the outline, station lines cross and the terrain footprint breaks",
    ("trumpet", "AC-H1"): "parallel: pavements of the semi-directional and the junction area at the shared stem end pass through each other",
    ("trumpet", "AC-H2"): "parallel: the semi-directional is reported crossing over the outer ramp it shares pavement with",
    ("trumpet", "AC-H3"): "parallel: the false semi-over-outer-ramp crossing has no clearance; rays start inside parapets",
    ("trumpet", "AC-A5"): "decks: loop deck end 44 m past the truth end where it lands beside the stem",
    ("directional", "AC-H3"): "decks: rays under a flyover crossing start inside the abutment of the I-88 deck below",
    ("directional", "AC-A1"): "decks: surface RMS 0.6 m where the flyover passes over the I-88 deck",
    ("directional", "AC-A4"): "decks: flyover-over-I-88 crossings measured against the wrong level (dz off by up to 14 m)",
    ("stack", "AC-H3"): "decks: rays under the level-1 crossings hit the level-2 deck and its parapet (stacked levels)",
    ("viaduct", "AC-H1"): "decks: viaduct deck faces self-intersect over the streets",
    ("viaduct", "AC-A1"): "decks: surface RMS 1.0 m on the viaduct that never rejoins the ground",
    ("viaduct", "AC-A4"): "decks: crossing dz 0.15-0.43 m low under the viaduct",
    ("viaduct", "AC-A5"): "decks: viaduct deck ends 10-26 m off",
}


@functools.cache
def outcome(name: str) -> S.Outcome:
    return S.run_scene(S.SCENES[name](), val3dity_exe=find_val3dity())


def _messages(result) -> list[str]:
    return [f"{v.element_id}: {v.message}" for v in result.violations][:5]


def _cases():
    for name in S.SCENES:
        for gate in GATES:
            reason = KNOWN.get((name, gate)) or KNOWN.get((name, None))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            if name not in FAST and not FULL:
                marks.append(pytest.mark.skip(reason=HEAVY_REASON))
            yield pytest.param(name, gate, id=f"{name}-{gate}", marks=marks)


@pytest.mark.parametrize(("name", "gate"), list(_cases()))
def test_section_9_gates_on_catalogue_scene(name, gate):
    """FR-EX-1 (no element replaced or dropped), AC-H1..AC-H6, AC-A1 (held-out returns), AC-A4 (crossing dz), AC-A5 (deck ends)."""
    out = outcome(name)
    if out.error:
        pytest.fail(f"{out.stage} raised: {out.error}")
    truth = out.scene.model
    if gate == "AC-A4" and not truth.crossings:
        pytest.skip("no grade separation in this scene")
    if gate == "AC-A5":
        if not truth.structures:
            pytest.skip("no structure in this scene")
        assert not out.missing_structures, f"truth structures with no model structure: {out.missing_structures}"
    result = out.gates.get(gate)
    if gate == "FR-EX-1":
        assert result is None, _messages(result)  # every element built (D-119)
        return
    assert result is not None, f"{gate} did not run: no model crossing or structure matched the truth"
    if gate == "AC-H1":
        assert result.metrics.get("n_native_violations") == 0.0, _messages(result)
        if find_val3dity() is None:
            pytest.skip("val3dity not available: AC-H1 oracle half not run (D-022)")
    assert result.passed, (result.skipped_reason, _messages(result))


# ---------------------------------------------------------------- the generator itself (fast)


@pytest.mark.parametrize("name", list(S.SCENES))
def test_fr_oc_1_scene_lidar_has_no_returns_under_a_higher_surface(name):
    """Under every truth grade separation the only returns are on the upper road (airborne occlusion, FR-OC-1)."""
    scene = S.SCENES[name]()
    sp = S.scene_points(scene, spacing=1.0, vehicles=0)
    xy, z = sp.xyz[:, :2], sp.xyz[:, 2]
    surf = np.isin(sp.truth, ["pavement", "deck", "terrain"])
    for c in scene.model.crossings:
        lower = scene.model.road(c.lower_road)
        z_low = lower.profile.z(c.lower_s)
        near = surf & (np.hypot(xy[:, 0] - c.x, xy[:, 1] - c.y) <= 2.0)
        assert near.any(), (name, c.id)
        assert np.all(z[near] > z_low + 3.0), (name, lower.name, float(z[near].min()), z_low)


@pytest.mark.parametrize("name", list(S.SCENES))
def test_ac_h2_scene_reference_and_vectors_follow_osm(name):
    """Every truth crossing is in the AC-H2 reference (D-044); bridge ways carry bridge/layer; ways meet at shared nodes (FR-VC-1)."""
    scene = S.SCENES[name]()
    ref = S.crossing_reference(scene)
    assert len(ref) == len(scene.model.crossings)
    edges = S.scene_vectors(scene)
    bridges = [e for e in edges if e.tags.get("bridge") == "yes"]
    assert len(bridges) == len(scene.model.structures)
    assert all(int(e.tags["layer"]) >= 1 for e in bridges)
    ends = np.array([p for e in edges for p in (e.coords[0], e.coords[-1])])
    for road in scene.model.roads:
        for s in (0.0, road.length):
            p = np.array(road.alignment.offset_point(s, 0.0))
            assert np.min(np.hypot(*(ends - p).T)) < 1e-6, (name, road.name, s)


@pytest.mark.parametrize("name", list(S.SCENES))
def test_fr_vc_1_scene_ways_chain_into_their_roads(name):
    """The vector graph makes one road per truth road, or joins ramps that meet end to end (FR-VC-1); never a stray piece."""
    from roader.conflate.graph import data_extent, select_roads
    from roader.io import from_synthetic

    scene = S.SCENES[name]()
    pts = from_synthetic(S.scene_points(scene, spacing=1.5, vehicles=0))
    seeds = select_roads(S.scene_vectors(scene), data_extent(pts.xyz[:, :2]))
    expected = {S.model_name(r) for r in scene.model.roads}
    assert {s.name for s in seeds} <= expected
    assert {ref.feature_id.split(":")[0] for s in seeds for ref in s.vector_refs} == {r.vector_refs[0].feature_id for r in scene.model.roads}
