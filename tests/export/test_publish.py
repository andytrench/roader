"""Publish gate, shared ids, report and determinism (FR-EX-1, FR-EX-3, FR-EX-4, AC-H5)."""

import json

from roader.export import FILES, write_unit
from roader.model import dumps

from .conftest import gate_report


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_ac_h5_two_runs_are_byte_identical(model, meshes, points, tmp_path):
    report = gate_report(model.unit.unit_id, passing=True)
    a = write_unit(model, meshes, report, tmp_path / "a", points=points)
    b = write_unit(model, meshes, report, tmp_path / "b", points=points)
    assert a.keys() == b.keys()
    tree_a, tree_b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    assert tree_a.keys() == tree_b.keys()
    for name in tree_a:
        assert tree_a[name] == tree_b[name], name
    for key in ("model", "glb", "glb_provenance", "usda", "cityjson", "xodr", "report_json", "report_md"):
        assert key in a
    assert any(key.startswith("figure:profile_") for key in a)


def test_fr_ex_1_failed_hard_gate_goes_to_held(model, meshes, tmp_path):
    paths = write_unit(model, meshes, gate_report(model.unit.unit_id, passing=False), tmp_path)
    unit = tmp_path / "held" / model.unit.unit_id
    assert paths["model"].parent == unit
    held = json.loads(paths["held"].read_text())
    assert [g["gate"] for g in held["failed_gates"]] == ["AC-H1"]
    assert held["failed_gates"][0]["violations"][0]["req"] == "FR-SV-1"
    assert not (tmp_path / "published").exists()


def test_fr_ex_1_passing_unit_is_published_and_replaces_held(model, meshes, tmp_path):
    write_unit(model, meshes, gate_report(model.unit.unit_id, passing=False), tmp_path)
    paths = write_unit(model, meshes, gate_report(model.unit.unit_id, passing=True), tmp_path)
    unit = tmp_path / "published" / model.unit.unit_id
    assert paths["model"].parent == unit
    assert "held" not in paths and not (unit / "HELD.json").exists()
    assert not (tmp_path / "held" / model.unit.unit_id).exists()
    write_unit(model, meshes, gate_report(model.unit.unit_id, passing=False), tmp_path)
    assert not unit.exists()  # a failing rerun withdraws the published copy


def test_fr_ex_3_all_exports_share_ids(model, meshes, tmp_path):
    paths = write_unit(model, meshes, gate_report(model.unit.unit_id, passing=True), tmp_path)
    assert paths["model"].read_bytes() == dumps(model)
    assert paths["model"].name == FILES["model"]
    usda = paths["usda"].read_text()
    city = json.loads(paths["cityjson"].read_text())
    xodr = paths["xodr"].read_text()
    report = json.loads(paths["report_json"].read_text())
    for mesh in meshes:
        assert mesh.id in usda
        assert mesh.id in paths["glb"].read_bytes().decode("latin-1")
        assert mesh.id in report["triangles"]["per_mesh"]
    for road in model.roads:
        assert road.id in city["CityObjects"] and f'id="{road.id}"' in xodr
    for st in model.structures:
        assert st.id in city["CityObjects"] and f'id="{st.id}"' in xodr


def test_fr_ex_4_report_contents(model, meshes, points, tmp_path):
    paths = write_unit(model, meshes, gate_report(model.unit.unit_id, passing=True), tmp_path, points=points)
    report = json.loads(paths["report_json"].read_text())
    assert report["inputs"]["unit_id"] == model.unit.unit_id
    assert report["gates"]["publishable"] is True
    assert [r["gate"] for r in report["gates"]["results"]] == ["AC-H1", "AC-H5", "AC-A1"]
    ramp = next(r for r in report["coverage"]["roads"] if r["name"] == "Ramp A")
    assert ramp["unsupported_m"] == 40.0 and ramp["observed_m"] == 260.0
    eb = model.roads[0]
    eb_cov = next(r for r in report["coverage"]["roads"] if r["road_id"] == eb.id)
    assert eb_cov["occluded_m"] == 28.0
    assert report["flags"][0]["code"] == "unsupported_gap"
    tri = report["triangles"]
    assert tri["total"] == sum(len(m.faces) for m in meshes)
    assert tri["per_structure"][0]["triangles"] > 0
    assert tri["unmatched_mesh_ids"] == []
    overlay = report["point_overlay"]["roads"][eb.id]
    assert overlay["n"] > 100 and overlay["rms_m"] < 0.1
    md = paths["report_md"].read_text()
    assert "## Coverage" in md and "AC-H1" in md
    for rel in report["figures"]:
        assert (paths["report_json"].parent / rel).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
