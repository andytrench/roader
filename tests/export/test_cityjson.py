"""CityJSON 2.0 export (FR-DM-2, FR-EX-3, FR-RH-2)."""

import numpy as np

from roader.export import cityjson_document, validate_cityjson
from roader.export.cityjson import ALLOWED_GEOMETRY, SECOND_LEVEL


def test_fr_dm_2_structurally_valid(model, meshes):
    doc = cityjson_document(model, meshes)
    assert validate_cityjson(doc) == []
    assert doc["transform"]["scale"] == [0.001, 0.001, 0.001]


def test_fr_dm_2_boundary_indices_refer_to_vertices(model, meshes):
    doc = cityjson_document(model, meshes)
    n = len(doc["vertices"])

    def flat(node):
        if isinstance(node, int):
            yield node
        else:
            for item in node:
                yield from flat(item)

    count = 0
    for obj in doc["CityObjects"].values():
        for geom in obj.get("geometry", []):
            for i in flat(geom["boundaries"]):
                assert 0 <= i < n
                count += 1
    assert count == sum(3 * len(m.faces) for m in meshes)


def test_fr_dm_2_types_parents_children(model, meshes):
    doc = cityjson_document(model, meshes)
    objects = doc["CityObjects"]
    for oid, obj in objects.items():
        assert obj["type"] in ALLOWED_GEOMETRY
        for geom in obj.get("geometry", []):
            assert geom["type"] in ALLOWED_GEOMETRY[obj["type"]]
        if obj["type"] in SECOND_LEVEL:
            assert obj["parents"]
        for parent in obj.get("parents", []):
            assert oid in objects[parent]["children"]
        for child in obj.get("children", []):
            assert oid in objects[child]["parents"]
    for road in model.roads:
        assert objects[road.id]["type"] == "Road"
    st = model.structures[0]
    bridge = objects[st.id]
    assert bridge["type"] == "Bridge"
    children = set(bridge["children"])
    assert f"{st.id}:deck" in children
    assert {sp.id for sp in st.supports} <= children
    assert {b.id for b in model.road(st.carried_road).cross_section.barriers} <= children
    pier = next(sp for sp in st.supports if sp.kind == "pier")
    assert objects[pier.id]["geometry"][0]["type"] == "MultiSolid"
    assert objects["terrain"]["type"] == "TINRelief"
    for slope in model.side_slopes:
        assert objects[slope.id]["parents"] == [slope.road_id]


def test_fr_rh_2_provenance_attributes_and_shared_ids(model, meshes):
    doc = cityjson_document(model, meshes)
    objects = doc["CityObjects"]
    mesh_ids = set()
    for oid, obj in objects.items():
        attrs = obj["attributes"]
        if "geometry" in obj:
            assert attrs["source"] in {"measured", "inventory", "vector", "rule"}, oid
            mesh_ids.update(attrs["mesh_ids"])
    assert mesh_ids == {m.id for m in meshes}
    assert objects[model.structures[0].id]["attributes"]["source"] == "inventory"
    assert objects[model.structures[0].id]["attributes"]["confidence"] == 0.8


def test_fr_dm_2_vertices_decode_to_mesh_coordinates(model, meshes):
    doc = cityjson_document(model, meshes)
    scale = np.array(doc["transform"]["scale"])
    translate = np.array(doc["transform"]["translate"])
    verts = np.array(doc["vertices"]) * scale + translate
    deck = next(m for m in meshes if m.part == "deck")
    geom = doc["CityObjects"][deck.id]["geometry"][0]
    assert geom["type"] == "Solid"
    first = geom["boundaries"][0][0][0]
    np.testing.assert_allclose(verts[first], deck.vertices[deck.faces[0]], atol=5e-4)


def test_fr_dm_2_traffic_area_semantics_on_pavement(model, meshes):
    doc = cityjson_document(model, meshes)
    road = doc["CityObjects"][model.roads[0].id]
    geom = road["geometry"][0]
    assert geom["semantics"]["surfaces"] == [{"type": "TrafficArea"}]
    values = geom["semantics"]["values"]
    assert len(values) == len(geom["boundaries"])
    assert 0 < values.count(0) < len(values)
