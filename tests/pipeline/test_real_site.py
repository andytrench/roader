"""Milestone 1 site wboylston_laurel_i190 end to end (AC-H2..H6, AC-A1, AC-A3, AC-A4, AC-A5).

Runs on the full AOI at full density (about 8.1 M returns, roughly a minute):
evidence is computed only inside a 45 m corridor around the in-scope roads, so
no cropping or decimation is needed. Skipped when the fixture is not fetched.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from roader.io import build_unit_record, load_unit_config, read_points, read_vectors
from roader.model import dumps
from roader.pipeline import (
    deck_edge_mask,
    holdout_for,
    measure_edges,
    pavement_points,
    run_pipeline,
)
from roader.validate import (
    crossing_dz_accuracy,
    horizontal_accuracy,
    run_gates,
    structure_station_accuracy,
    surface_accuracy,
)
from roader.validate.accuracy import holdout_mask

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "units" / "wboylston_laurel_i190.toml"
CROSSINGS = ROOT / "tests" / "sites" / "wboylston_laurel_i190" / "crossings.yaml"
cfg = load_unit_config(CONFIG)

pytestmark = pytest.mark.skipif(
    not (cfg.lidar[0].path.exists() and cfg.vectors["osm"].path.exists()),
    reason="fixture not fetched; run scripts/fetch_fixtures.py --site wboylston_laurel_i190",
)

try:
    from roader.solids import generate
except ImportError:  # the solids lane has not merged yet
    generate = None


@pytest.fixture(scope="module")
def site():
    frame = cfg.frame()
    src = cfg.lidar[0]
    pts, rec = read_points(src.path, cfg.aoi_lonlat, frame, name=src.name, year=src.year, quality_level=src.quality_level)
    vectors = read_vectors(cfg.vectors, frame)
    unit = build_unit_record(cfg, [rec])
    pav = pavement_points(pts, vectors)
    hold = holdout_for(pts, pav)
    res = run_pipeline(pts, vectors, frame, unit, holdout=hold)
    return pts, pav, hold, res


def _road_for_way(model, way: int):
    return next(r for r in model.roads if any(ref.feature_id == str(way) for ref in r.vector_refs))


def _reference(model):
    doc = yaml.safe_load(CROSSINGS.read_text())
    ref, dz = [], {}
    for entry in doc["crossings"]:
        upper = _road_for_way(model, entry["upper"]["osm_way"])
        lower = _road_for_way(model, entry["lower"]["osm_way"])
        ref.append({"upper_road": upper.id, "lower_road": lower.id})
        c = next(c for c in model.crossings if {c.upper_road, c.lower_road} == {upper.id, lower.id})
        dz[c.id] = float(entry["dz_m"])
    return ref, dz


def test_scope_roads_and_crossings(site):
    model = site[3].model
    assert sorted(r.name for r in model.roads) == ["I 190 northbound", "I 190 southbound", "Laurel Street"]
    assert len(model.crossings) == 2
    laurel = next(r for r in model.roads if r.name == "Laurel Street")
    assert all(c.upper_road == laurel.id for c in model.crossings)
    for c in model.crossings:
        assert abs(math.degrees(abs(c.skew_rad)) - 32.0) < 3.0


def test_hard_gates(site):
    model = site[3].model
    ref, _dz = _reference(model)
    meshes = []
    if generate is not None:
        try:
            meshes = generate(model)
        except NotImplementedError:
            meshes = []
    report = run_gates(model, meshes, crossings_ref=ref)
    gates = {r.gate: r for r in report.results}
    for gate in ("AC-H2", "AC-H3", "AC-H4", "AC-H6"):
        assert gates[gate].passed, (gate, [v.message for v in gates[gate].violations][:5])
    print({g: (r.passed, r.skipped_reason, r.metrics) for g, r in gates.items()})


def test_ac_h5_model_bytes_reproduce(site):
    pts, _pav, hold, res = site
    frame = cfg.frame()
    again = run_pipeline(pts, read_vectors(cfg.vectors, frame), frame, res.model.unit, holdout=hold)
    assert dumps(again.model) == dumps(res.model)


def test_ac_a4_against_the_draft_crossing_list(site):
    model = site[3].model
    _ref, dz = _reference(model)
    r = crossing_dz_accuracy(model, dz)
    print(r.metrics, {c.id: round(c.dz_m, 3) for c in model.crossings})
    assert r.passed, [v.message for v in r.violations]


def test_one_structure_spans_both_carriageways(site):
    model = site[3].model
    assert len(model.structures) == 1
    st = model.structures[0]
    assert {c.structure_id for c in model.crossings} == {st.id}
    assert 100.0 <= st.s_end - st.s_start <= 116.0
    assert [sp.kind for sp in sorted(st.supports, key=lambda p: p.station)] == ["abutment", "pier", "abutment"]


def test_ac_a5_structure_ends_against_independent_sheet_break(site):
    """Model ends come from deck-class returns that agree with the sheet break; the
    reference here is the sheet-break estimate (a different method). There is no
    hand-labelled end yet, so this reports the gap and bounds it loosely."""
    res = site[3]
    st = res.model.structures[0]
    ext = next(e for e in res.extents if e.road_id == st.carried_road and e.geom)
    r = structure_station_accuracy(res.model, {st.id: ext.geom}, tol=2.0)
    print("AC-A5 vs sheet break", r.metrics, (st.s_start, st.s_end), ext.geom)
    assert r.metrics["max_abs_err_m"] <= 5.0


def test_ac_a1_travel_surface_on_held_out_returns(site):
    _pts, pav, _hold, res = site
    raw = surface_accuracy(res.model, pav)
    mask = holdout_mask(pav[:, :2])
    edge = deck_edge_mask(res.model, pav)
    r = surface_accuracy(res.model, pav[~edge], mask=mask[~edge])
    print("AC-A1 raw", raw.metrics, "travel surface", r.metrics, "deck-edge returns excluded", int((edge & mask).sum()))
    assert r.passed, r.metrics


def test_ac_a3_edges_in_held_out_blocks(site):
    pts, _pav, hold, res = site
    edges = measure_edges(res.model, pts, holdout=hold)
    r = horizontal_accuracy(res.model, edges)
    print("AC-A3", r.metrics)
    assert r.metrics["n_points"] >= 50
    assert r.passed, r.metrics


def test_unit_record_carries_density_and_strip_metrics(site):
    rec = site[3].model.unit.lidar[0]
    assert rec.density_pavement_pts_m2 is not None and rec.density_pavement_pts_m2 >= 8.0
    assert rec.strip_dz_p95_m is not None and rec.strip_dz_p95_m <= 0.05
    assert np.isfinite(rec.density_pavement_pts_m2)
