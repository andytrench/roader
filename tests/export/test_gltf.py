"""glTF 2.0 binary export (FR-EX-3, FR-DM-3, AC-H6, FR-AS-6, FR-AS-10, FR-RH-1)."""

import numpy as np
import pytest

from roader.export import glb_bytes, points_glb_bytes, read_glb
from roader.export._common import PROVENANCE_COLORS
from roader.model import Mesh, Source

_DTYPE = {5126: "<f4", 5125: "<u4"}
_WIDTH = {"VEC3": 3, "SCALAR": 1}


def _accessor(doc, bin_chunk, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    assert view["byteOffset"] % 4 == 0
    assert view["byteOffset"] + view["byteLength"] <= len(bin_chunk)
    data = np.frombuffer(bin_chunk, dtype=_DTYPE[acc["componentType"]], count=acc["count"] * _WIDTH[acc["type"]],
                         offset=view["byteOffset"])
    assert data.nbytes == view["byteLength"]
    return acc, data.reshape(acc["count"], _WIDTH[acc["type"]])


def test_fr_ex_3_glb_parses_and_accessors_are_consistent(model, meshes):
    doc, bin_chunk = read_glb(glb_bytes(model, meshes))
    assert doc["asset"]["version"] == "2.0"
    assert doc["buffers"][0]["byteLength"] == len(bin_chunk)
    by_id = {m.id: m for m in meshes}
    assert len(doc["nodes"]) == len(meshes) == len(doc["meshes"])
    for node in doc["nodes"]:
        mesh = by_id[node["name"]]
        prim = doc["meshes"][node["mesh"]]["primitives"][0]
        pos_acc, pos = _accessor(doc, bin_chunk, prim["attributes"]["POSITION"])
        idx_acc, idx = _accessor(doc, bin_chunk, prim["indices"])
        assert pos_acc["count"] == len(mesh.vertices)
        assert idx_acc["count"] == 3 * len(mesh.faces)
        assert pos_acc["min"] == [float(v) for v in pos.min(axis=0)]
        assert pos_acc["max"] == [float(v) for v in pos.max(axis=0)]
        assert int(idx.max()) < pos_acc["count"]
        np.testing.assert_array_equal(idx.reshape(-1, 3), mesh.faces)


def test_fr_dm_3_ac_h6_every_node_has_mesh_and_extras(model, meshes):
    doc, _ = read_glb(glb_bytes(model, meshes))
    by_id = {m.id: m for m in meshes}
    scene_nodes = set(doc["scenes"][doc["scene"]]["nodes"])
    assert scene_nodes == set(range(len(doc["nodes"])))  # FR-AS-10: no orphan nodes
    used_meshes = set()
    for i, node in enumerate(doc["nodes"]):
        assert "mesh" in node
        used_meshes.add(node["mesh"])
        mesh = by_id[node["name"]]
        extras = node["extras"]
        assert extras["element_id"] == mesh.element_id
        assert extras["layer"] == mesh.layer
        assert extras["part"] == mesh.part
        assert extras["material"] == mesh.material
        assert extras["source"] == mesh.source.value
    assert used_meshes == set(range(len(doc["meshes"])))


def test_fr_as_10_y_up_transform_and_anchor(model, meshes):
    doc, bin_chunk = read_glb(glb_bytes(model, meshes))
    extras = doc["asset"]["extras"]
    assert extras["frame"]["proj"] == model.frame.proj
    assert extras["frame"]["anchor_lon"] == model.frame.anchor_lon
    assert extras["tile_origin_unit_frame"] == [0.0, 0.0, 0.0]
    node = doc["nodes"][0]
    mesh = {m.id: m for m in meshes}[node["name"]]
    _acc, pos = _accessor(doc, bin_chunk, doc["meshes"][node["mesh"]]["primitives"][0]["attributes"]["POSITION"])
    v = mesh.vertices
    np.testing.assert_allclose(pos, np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]), atol=1e-4)


def test_fr_as_6_one_material_per_slot(model, meshes):
    doc, _ = read_glb(glb_bytes(model, meshes))
    names = [m["name"] for m in doc["materials"]]
    assert names == sorted({m.material for m in meshes})
    for mat in doc["materials"]:
        assert set(mat["pbrMetallicRoughness"]) == {"baseColorFactor", "metallicFactor", "roughnessFactor"}


def test_fr_rh_1_provenance_variant_uses_four_source_colors(model, meshes):
    doc, _ = read_glb(glb_bytes(model, meshes, color_by="source"))
    colors = [tuple(m["pbrMetallicRoughness"]["baseColorFactor"]) for m in doc["materials"]]
    assert len(colors) == 4 and len(set(colors)) == 4
    used = set()
    for node in doc["nodes"]:
        mat = doc["materials"][doc["meshes"][node["mesh"]]["primitives"][0]["material"]]
        source = Source(node["extras"]["source"])
        assert tuple(mat["pbrMetallicRoughness"]["baseColorFactor"]) == PROVENANCE_COLORS[source]
        used.add(source)
    assert used == set(Source)
    plain, _ = read_glb(glb_bytes(model, meshes))
    assert plain["accessors"] == doc["accessors"]  # same geometry


def test_fr_as_10_empty_mesh_rejected(model):
    empty = Mesh("x:pavement", "x", "L3", "pavement", "asphalt", Source.RULE, np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        glb_bytes(model, [empty])


def test_fr_ex_4_points_overlay_is_deterministic_and_class_colored(model):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-50, 50, (1000, 3))
    cls = np.where((np.arange(1000) // 4) % 2, 17, 2)
    data = points_glb_bytes(model, xyz, cls, max_points=300)
    assert data == points_glb_bytes(model, xyz, cls, max_points=300)
    doc, blob = read_glb(data)
    prim = doc["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0
    n = doc["accessors"][0]["count"]
    assert n == len(xyz[::4]) and doc["asset"]["extras"]["decimation_step"] == 4
    view = doc["bufferViews"][1]
    colors = np.frombuffer(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]], dtype=np.uint8).reshape(-1, 4)
    assert {tuple(c[:3]) for c in colors} == {(150, 115, 70), (235, 200, 60)}
