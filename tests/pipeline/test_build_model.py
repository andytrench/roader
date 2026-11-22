"""build_model end to end on the synthetic overpass: AC-H2..H6, AC-A1, AC-A4, AC-A5, D-018."""

from __future__ import annotations

import numpy as np
import pytest

from roader.export.run import export_model
from roader.model import dumps, loads
from roader.pipeline import build_model, holdout_for, pavement_points, run_pipeline
from roader.solids import generate
from roader.validate import (
    crossing_dz_accuracy,
    structure_station_accuracy,
    surface_accuracy,
)
from roader.validate.accuracy import BLOCK_M
from roader.validate.val3dity import find_val3dity


def test_ac_h5_fr_pl_3_deterministic_bytes(points, vectors, truth, model):
    again = build_model(points, vectors, truth.frame, truth.unit)
    assert dumps(again) == dumps(model)
    assert dumps(loads(dumps(model))) == dumps(model)


REF = [{"upper_name": "Route 1", "lower_name": "I-99 eastbound"}, {"upper_name": "Route 1", "lower_name": "I-99 westbound"}]


def test_generate_renders_every_model_part(model):
    """Stage 10 consumes the completed model: terrain, embankments, deck, supports, parapets (FR-SG-6, FR-OC-3)."""
    meshes = generate(model)
    parts = {m.part for m in meshes}
    assert {"pavement", "embankment", "deck", "abutment", "pier_cap", "pier_column", "parapet", "terrain"} <= parts


def test_fr_oc_3_terrain_completed_with_cell_provenance(model):
    t = model.terrain
    assert t is not None and None not in t.z
    assert len(t.fill) == len(t.z)
    codes = set(t.fill)
    assert {0, 2, 3} <= codes  # measured, under the deck, under roads
    assert any(f.startswith("FR-OC-3:") for f in t.prov.flags)


def test_hard_gates_through_export(model, tmp_path):
    """AC-H1..AC-H6 through the stage 12 publish path (FR-EX-1)."""
    exe = find_val3dity()
    report, paths = export_model(model, tmp_path, crossings_ref=REF, val3dity_exe=exe)
    gates = {r.gate: r for r in report.results}
    for gate in ("AC-H2", "AC-H3", "AC-H4", "AC-H5", "AC-H6"):
        assert gates[gate].passed, (gate, gates[gate].skipped_reason, [v.message for v in gates[gate].violations][:5])
    native = gates["AC-H1"].metrics.get("n_native_violations")
    assert native == 0.0, [v.message for v in gates["AC-H1"].violations][:5]
    if exe is None:
        pytest.skip("val3dity not available: AC-H1 oracle half not run, unit is held by design (D-022)")
    assert gates["AC-H1"].passed, [v.message for v in gates["AC-H1"].violations][:5]
    assert all(r.passed for r in report.results if r.hard)
    assert "published" in str(paths["model"]) or any("published" in str(p) for p in paths.values())


def test_ac_a4_and_a5_against_truth(model, truth):
    t_dz = {by.vector_refs[0].feature_id: c.dz_m for c in truth.crossings for by in truth.roads if by.id == c.lower_road}
    measured = {}
    for c in model.crossings:
        fid = model.road(c.lower_road).vector_refs[0].feature_id
        measured[c.id] = t_dz[fid]
    assert crossing_dz_accuracy(model, measured, tol=0.1).passed
    ts = truth.structures[0]
    assert structure_station_accuracy(model, {model.structures[0].id: (ts.s_start, ts.s_end)}, tol=1.0).passed


@pytest.fixture(scope="module")
def held(points, vectors, truth):
    pav = pavement_points(points, vectors)
    hold = holdout_for(points, pav)
    res = run_pipeline(points, vectors, truth.frame, truth.unit, holdout=hold)
    return pav, hold, res


def test_d_018_holdout_points_never_reach_fitting(held, points):
    pav, hold, res = held
    assert hold.any()
    from roader.validate.accuracy import holdout_mask

    assert 0.1 < holdout_mask(pav[:, :2]).mean() < 0.4
    blocks = {tuple(b) for b in np.floor(points.xyz[hold, :2] / BLOCK_M).astype(int).tolist()}
    used = {tuple(b) for b in np.floor(res.evidence.xyz[:, :2] / BLOCK_M).astype(int).tolist()}
    assert not blocks & used


def test_ac_a1_surface_accuracy_on_held_out_returns(held):
    pav, _hold, res = held
    r = surface_accuracy(res.model, pav)
    assert r.passed, r.metrics
    assert r.metrics["rms_m"] <= 0.10 and r.metrics["p95_m"] <= 0.20


def test_ac_h6_every_rule_fill_names_its_rule(model):
    from dataclasses import fields, is_dataclass

    from roader.model.provenance import Provenance, Source

    def walk(obj):
        if isinstance(obj, Provenance):
            yield obj
        elif is_dataclass(obj):
            for f in fields(obj):
                yield from walk(getattr(obj, f.name))
        elif isinstance(obj, list):
            for item in obj:
                yield from walk(item)

    provs = list(walk(model))
    assert provs
    assert all(p.rule_id for p in provs if p.source == Source.RULE)
