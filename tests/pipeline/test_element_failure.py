"""No-crash contract (FR-EX-1, D-119): one element that cannot be built never aborts the unit.

The element is replaced by its simplest valid form or dropped, flagged with the
requirement and the element id (the exception type and message kept in the text), and
the unit is held with the flag in report.json.
"""

from __future__ import annotations

import json

import pytest

from roader import solids
from roader.export.run import export_model
from roader.model import dumps, loads
from roader.model.schema import ELEMENT_FAILED
from roader.model.synthetic import overpass_model
from roader.solids import generate

REAL_ROAD_BODIES = solids.road_bodies


def _ref(model):
    names = {r.id: r.name for r in model.roads}
    return [{"upper_name": names[c.upper_road], "lower_name": names[c.lower_road]} for c in model.crossings]


def _breaking(road_id: str, *, squared_too: bool):
    """road_bodies that raises for ``road_id`` (on every attempt, or only while its structures are skewed)."""

    def road_bodies(model, road, *args, **kwargs):
        if road.id == road_id:
            skewed = any(sp.skew_rad != 0.0 for st in model.structures if st.carried_road == road_id for sp in st.supports)
            if squared_too or skewed:
                raise RuntimeError("injected failure")
        return REAL_ROAD_BODIES(model, road, *args, **kwargs)

    return road_bodies


def test_generate_without_failures_list_still_raises(monkeypatch):
    model = overpass_model()
    monkeypatch.setattr(solids, "road_bodies", _breaking(model.roads[0].id, squared_too=True))
    with pytest.raises(RuntimeError, match="injected failure"):
        generate(model)


def test_fr_ex_1_failed_road_is_dropped_flagged_and_the_unit_held(monkeypatch, tmp_path):
    model = loads(dumps(overpass_model()))
    eb = model.roads[0]
    monkeypatch.setattr(solids, "road_bodies", _breaking(eb.id, squared_too=True))
    report, paths = export_model(model, tmp_path, crossings_ref=_ref(model), lods=False)
    assert not report.publishable
    gate = next(r for r in report.results if r.gate == "FR-EX-1")
    assert gate.hard and not gate.passed
    assert [(v.element_id, v.req) for v in gate.violations] == [(eb.id, "FR-SG-1")]
    assert "RuntimeError: injected failure" in gate.violations[0].message
    assert "held" in paths
    data = json.loads(paths["report_json"].read_text())
    flags = [f for f in data["flags"] if f["code"] == ELEMENT_FAILED]
    assert [(f["element_id"], f["req"]) for f in flags] == [(eb.id, "FR-SG-1")]
    assert "RuntimeError: injected failure" in flags[0]["message"]


def test_fr_ex_1_road_with_a_failing_skewed_deck_is_squared_not_dropped(monkeypatch):
    model = overpass_model()
    over = model.roads[2]
    monkeypatch.setattr(solids, "road_bodies", _breaking(over.id, squared_too=False))
    failures = []
    meshes = generate(model, failures=failures)
    assert [(f.code, f.element_id, f.req) for f in failures] == [(ELEMENT_FAILED, over.id, "FR-ST-4")]
    assert "RuntimeError: injected failure" in failures[0].message
    assert any(m.element_id == over.id for m in meshes)
    assert any(m.part == "deck" for m in meshes)


def test_fr_ex_1_failing_structure_inference_is_flagged_and_the_unit_held(monkeypatch, points, vectors, truth, tmp_path):
    from roader import pipeline
    from roader.pipeline import run_pipeline

    def infer_structure(*args, **kwargs):
        raise KeyError("injected inference failure")

    monkeypatch.setattr(pipeline, "infer_structure", infer_structure)
    model = run_pipeline(points, vectors, truth.frame, truth.unit).model
    failed = [f for f in model.flags if f.code == ELEMENT_FAILED]
    assert failed and all(f.req == "FR-ST-2" and "KeyError" in f.message for f in failed)
    assert not model.structures
    report, paths = export_model(model, tmp_path, lods=False)
    assert not report.publishable and "held" in paths
    gate = next(r for r in report.results if r.gate == "FR-EX-1")
    assert {v.element_id for v in gate.violations} == {f.element_id for f in failed}
