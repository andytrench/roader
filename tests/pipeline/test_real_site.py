"""Milestone 1 acceptance on wboylston_laurel_i190 (plan 1.12): AC-H1..H6, AC-A1, AC-A3, AC-A4, AC-A5.

Runs on the full AOI at full density (about 8.1 M returns): evidence is computed
only inside a 45 m corridor around the in-scope roads, so no cropping or
decimation is needed. The unit goes through ``roader.export.run.export_model``
(meshes, exports twice for AC-H5, every gate, publish or hold). About three
minutes; skipped when the fixture is not fetched.

References: AC-H2 and AC-A4 use the hand-labelled crossing list, AC-A5 the
structure ends in the same file (reviewed 2026-11-22; AC-A5 stays a consistency check, D-044).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from roader.build import load_site_reference
from roader.export.run import export_model
from roader.io import (
    FrameTransformer,
    build_unit_record,
    load_unit_config,
    read_points,
    read_vectors,
)
from roader.model import dumps
from roader.pipeline import holdout_for, measure_edges, pavement_points, run_pipeline
from roader.validate.val3dity import find_val3dity

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "units" / "wboylston_laurel_i190.toml"
CROSSINGS = ROOT / "tests" / "sites" / "wboylston_laurel_i190" / "crossings.yaml"
cfg = load_unit_config(CONFIG)

pytestmark = pytest.mark.skipif(
    not (cfg.lidar[0].path.exists() and cfg.vectors["osm"].path.exists()),
    reason="fixture not fetched; run scripts/fetch_fixtures.py --site wboylston_laurel_i190",
)

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
    assert sorted(r.name for r in model.roads) == ["I-190 northbound", "I-190 southbound", "Laurel Street"]
    assert len(model.crossings) == 2
    laurel = next(r for r in model.roads if r.name == "Laurel Street")
    assert all(c.upper_road == laurel.id for c in model.crossings)
    for c in model.crossings:
        assert abs(math.degrees(abs(c.skew_rad)) - 32.0) < 3.0


def test_ac_h5_model_bytes_reproduce(site):
    pts, _pav, hold, res = site
    frame = cfg.frame()
    again = run_pipeline(pts, read_vectors(cfg.vectors, frame), frame, res.model.unit, holdout=hold)
    assert dumps(again.model) == dumps(res.model)


def test_one_structure_spans_both_carriageways(site):
    model = site[3].model
    assert len(model.structures) == 1
    st = model.structures[0]
    assert {c.structure_id for c in model.crossings} == {st.id}
    assert 100.0 <= st.s_end - st.s_start <= 116.0
    assert [sp.kind for sp in sorted(st.supports, key=lambda p: p.station)] == ["abutment", "pier", "abutment"]


def _draft_ends(model) -> dict[str, tuple[float, float]]:
    """Draft structure ends (lon/lat on the centerline) as stations on the model's carried road."""
    doc = yaml.safe_load(CROSSINGS.read_text())
    to_unit = FrameTransformer(cfg.frame())
    out = {}
    ends = doc.get("structure_ends", {})
    assert ends["status"].startswith(("draft", "reviewed"))
    for entry in ends["structures"]:
        for end in entry["ends"]:
            road = _road_for_way(model, end["along"])
            st = next(s for s in model.structures if s.carried_road == road.id)
            stations = []
            for key in ("start_lonlat", "end_lonlat"):
                x, y = to_unit.lonlat_to_unit([end[key][0]], [end[key][1]])
                stations.append(road.alignment.project(float(x[0]), float(y[0]))[0])
            out[st.id] = (min(stations), max(stations))
    return out


@pytest.fixture(scope="module")
def acceptance(site, tmp_path_factory):
    pts, pav, hold, res = site
    model = res.model
    ref, dz = _reference(model)
    measured = {
        "edges": measure_edges(model, pts, holdout=hold),
        "crossing_dz": dz,
        "crossing_points": load_site_reference(CROSSINGS, model, cfg.frame()).crossing_points,  # D-096
        "structure_stations": _draft_ends(model),
    }
    out = tmp_path_factory.mktemp("m1")
    report, paths = export_model(model, out, crossings_ref=ref, points=pav, measured=measured, val3dity_exe=find_val3dity())
    return report, paths


def test_m1_acceptance_gate_table(acceptance):
    report, paths = acceptance
    gates = {r.gate: r for r in report.results}
    for name in sorted(gates):
        r = gates[name]
        print(f"{name:6s} {'PASS' if r.passed else 'FAIL'} {'hard' if r.hard else 'target'} {r.skipped_reason} {r.metrics}")
    for gate in ("AC-H2", "AC-H3", "AC-H4", "AC-H5", "AC-H6", "AC-A1", "AC-A3", "AC-A4", "AC-A5"):
        assert gates[gate].passed, (gate, gates[gate].skipped_reason, [v.message for v in gates[gate].violations][:5])
    assert gates["AC-H1"].metrics.get("n_native_violations") == 0.0, [v.message for v in gates["AC-H1"].violations][:5]
    if find_val3dity() is None:
        pytest.skip("val3dity not available: AC-H1 oracle half not run, unit is held by design (D-022)")
    assert gates["AC-H1"].passed, [v.message for v in gates["AC-H1"].violations][:5]
    assert all(r.passed for r in report.results if r.hard)
    assert "held" not in paths


def test_unit_record_carries_density_and_strip_metrics(site):
    rec = site[3].model.unit.lidar[0]
    assert rec.density_pavement_pts_m2 is not None and rec.density_pavement_pts_m2 >= 8.0
    assert rec.strip_dz_p95_m is not None and rec.strip_dz_p95_m <= 0.05
    assert np.isfinite(rec.density_pavement_pts_m2)
