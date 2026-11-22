"""LODs, UVs and normals in the exports (FR-AS-1, FR-AS-3..5, FR-EX-3, AC-H5)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from roader.export import cityjson_bytes, glb_bytes, read_glb, usda_text, write_unit
from roader.export.run import export_model
from roader.model import dumps, loads
from roader.model.mesh import corner_values
from roader.model.synthetic import overpass_model
from roader.solids import generate_lods
from roader.validate.report import GateReport
from roader.validate.val3dity import find_val3dity

_WIDTH = {"VEC2": 2, "VEC3": 3, "SCALAR": 1}
_DTYPE = {5126: "<f4", 5125: "<u4"}


@pytest.fixture(scope="module")
def scene():
    model = overpass_model()
    return model, generate_lods(model)


def _read(doc, blob, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    width = _WIDTH[acc["type"]]
    data = np.frombuffer(blob, dtype=_DTYPE[acc["componentType"]], count=acc["count"] * width, offset=view["byteOffset"])
    return data.reshape(acc["count"], width)


def test_gltf_carries_normals_and_both_uv_sets_fr_as_3_4_5(scene):
    model, lods = scene
    meshes = lods[0]
    doc, blob = read_glb(glb_bytes(model, meshes))
    by_id = {m.id: m for m in meshes}
    assert "uv1_atlas" in doc["asset"]["extras"] and "texcoords" in doc["asset"]["extras"]
    for node in doc["nodes"]:
        mesh = by_id[node["name"]]
        prim = doc["meshes"][node["mesh"]]["primitives"][0]
        attrs = prim["attributes"]
        assert {"POSITION", "NORMAL", "TEXCOORD_1"} <= set(attrs)
        idx = _read(doc, blob, prim["indices"]).reshape(-1, 3)
        pos = _read(doc, blob, attrs["POSITION"])
        nrm = _read(doc, blob, attrs["NORMAL"])
        assert len(nrm) == len(pos)
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
        # Corners decode to the mesh's corners (Y-up) and to its per-corner attributes.
        v = mesh.vertices[mesh.faces]
        np.testing.assert_allclose(pos[idx], np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1), atol=1e-3)
        n = corner_values(mesh, mesh.normals, 3)
        np.testing.assert_allclose(nrm[idx], np.stack([n[..., 0], n[..., 2], -n[..., 1]], axis=-1), atol=1e-6)
        np.testing.assert_allclose(_read(doc, blob, attrs["TEXCOORD_1"])[idx], corner_values(mesh, mesh.uv1, 2), atol=1e-6)
        if mesh.closed:
            np.testing.assert_allclose(_read(doc, blob, attrs["TEXCOORD_0"])[idx], mesh.uv0, atol=1e-3)
            assert node["extras"]["uv1_texel_px_per_m"] > 0.0
        # Unwelded only where attributes differ: far fewer vertices than corners.
        assert len(pos) < 3 * len(mesh.faces)


def test_gltf_lod_chain_through_msft_lod_fr_as_1(scene):
    model, lods = scene
    data = glb_bytes(model, lods[0], lods=lods)
    assert data == glb_bytes(model, lods[0], lods=lods)
    doc, _ = read_glb(data)
    assert doc["extensionsUsed"] == ["MSFT_lod"]
    roots = doc["scenes"][0]["nodes"]
    assert len(roots) == len(lods[0])
    referenced = set()
    for r in roots:
        node = doc["nodes"][r]
        ids = node["extensions"]["MSFT_lod"]["ids"]
        assert [doc["nodes"][i]["name"] for i in ids] == [f"{node['name']}#lod{k}" for k in (1, 2, 3)]
        assert len(node["extras"]["MSFT_screencoverage"]) == len(ids) + 1
        referenced.update(ids)
    assert referenced.isdisjoint(roots) and referenced | set(roots) == set(range(len(doc["nodes"])))


def test_usd_primvars_and_lod_variants_fr_as_1_3_4_5(scene):
    model, lods = scene
    text = usda_text(model, lods[0], lods=lods)
    assert text == usda_text(model, lods[0], lods=lods)
    assert 'prepend variantSets = "lod"' in text and 'string lod = "lod0"' in text
    for k in range(4):
        assert f'"lod{k}" {{' in text
    assert len(re.findall(r'def Mesh "', text)) == sum(len(v) for v in lods.values())
    assert text.count("{") == text.count("}")
    closed = [m for m in lods[0] if m.closed]
    assert text.count("primvars:normals") == sum(len(v) for v in lods.values())
    assert text.count('interpolation = "faceVarying"') >= 3 * len(closed)
    first = closed[0]
    block = text.split(f'string mesh_id = "{first.id}"')[1]
    values = re.search(r"normal3f\[\] primvars:normals = \[(.*?)\] \(", block).group(1)
    assert values.count("(") == 3 * len(first.faces)


def test_cityjson_ignores_lods_and_stays_level_0(scene, tmp_path):
    model, lods = scene
    paths = write_unit(model, lods[0], GateReport(unit_id=model.unit.unit_id), tmp_path, lods=lods)
    assert paths["cityjson"].read_bytes() == cityjson_bytes(model, lods[0])


def test_export_model_realtime_profile_gates_and_lods(tmp_path):
    model = loads(dumps(overpass_model()))  # the saved form, as `roader export` reads it (D-043)
    names = {r.id: r.name for r in model.roads}
    ref = [{"upper_name": names[c.upper_road], "lower_name": names[c.lower_road]} for c in model.crossings]
    report, paths = export_model(model, tmp_path, crossings_ref=ref, val3dity_exe=find_val3dity(), profile="realtime")
    gates = {r.gate: r for r in report.results}
    assert "AC-H1:realtime" in gates and "AC-H1" not in gates
    assert gates["FR-AS-1"].passed and gates["AC-H5"].passed and gates["AC-H3"].passed
    native = [v for v in gates["AC-H1:realtime"].violations if not v.message.startswith("val3dity")]
    assert native == []
    assert gates["AC-P4"].passed
    doc, _ = read_glb(paths["glb"].read_bytes())
    assert doc["extensionsUsed"] == ["MSFT_lod"]
