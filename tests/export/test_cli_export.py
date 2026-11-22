"""`roader export`: render, gate and publish from a native model (FR-EX-1, AC-H2, AC-H5)."""

import json

import yaml
from click.testing import CliRunner

from roader.cli import main
from roader.export.run import load_crossing_reference
from roader.model import dumps
from roader.model.synthetic import overpass_model


def _reference(model):
    names = {r.id: r.name for r in model.roads}
    return {
        "crossings": [
            {"upper": {"name": names[c.upper_road]}, "lower": {"name": names[c.lower_road]}}
            for c in model.crossings
        ]
    }


def test_load_crossing_reference_site_format(tmp_path):
    path = tmp_path / "crossings.yaml"
    path.write_text(yaml.safe_dump({"crossings": [{"upper": {"name": "A"}, "lower": {"name": "B"}}]}))
    assert load_crossing_reference(path) == [{"upper_name": "A", "lower_name": "B"}]


def test_cli_export_gates_and_routes_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("VAL3DITY", str(tmp_path / "missing"))
    model = overpass_model()
    model_path = tmp_path / "model.json"
    model_path.write_bytes(dumps(model))
    crossings = tmp_path / "crossings.yaml"
    crossings.write_text(yaml.safe_dump(_reference(model)))

    result = CliRunner().invoke(main, ["export", "--model", str(model_path), "--out", str(tmp_path / "out"), "--crossings", str(crossings)])
    assert result.exit_code == 0, result.output
    assert "PASS AC-H2 (hard)" in result.output
    assert "PASS AC-H5 (hard)" in result.output
    # Without val3dity, AC-H1 cannot pass, so the unit is held.
    assert "SKIP AC-H1 (hard)" in result.output or "FAIL AC-H1 (hard)" in result.output
    unit = tmp_path / "out" / "held" / model.unit.unit_id
    assert (unit / "model.json").read_bytes() == dumps(model)
    held = json.loads((unit / "HELD.json").read_text())
    assert "AC-H1" in [g["gate"] for g in held["failed_gates"]]
