"""Vector reading: FR-IN-5, FR-IN-7, spec section 1 scope."""

import json

import numpy as np
import pytest

from roader.io import (
    FrameTransformer,
    in_scope,
    is_limited_access,
    read_osm,
    read_vectors,
    unit_frame,
)
from roader.io.vectors import VectorEdge

LON, LAT = -71.8161, 42.38906
DLAT = 1.0 / 111045.0  # about 1 m
DLON = 1.0 / 82222.0


def _node(i, east_m, north_m):
    return {"type": "node", "id": i, "lon": LON + east_m * DLON, "lat": LAT + north_m * DLAT}


def _osm_doc():
    nodes = [
        _node(1, -300, 0), _node(2, 300, 0),  # motorway along x
        _node(3, 0, -200), _node(4, 0, 200),  # crossing road
        _node(5, -300, 100), _node(6, 300, 100),  # parallel road 100 m away
        _node(7, -300, 400), _node(8, 300, 400),  # parallel road 400 m away
        _node(9, 50, 50),
    ]
    ways = [
        {"type": "way", "id": 30, "nodes": [3, 4], "tags": {"highway": "secondary", "bridge": "yes", "layer": 1}},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "motorway", "ref": "I 190", "lanes": 2}},
        {"type": "way", "id": 50, "nodes": [5, 6], "tags": {"highway": "residential"}},
        {"type": "way", "id": 70, "nodes": [7, 8], "tags": {"highway": "residential"}},
        {"type": "way", "id": 90, "nodes": [9, 999], "tags": {"highway": "service"}},  # one node missing
        {"type": "way", "id": 95, "nodes": [1, 9], "tags": {"waterway": "stream"}},
    ]
    return {"version": 0.6, "elements": ways + nodes}


@pytest.fixture
def osm_file(tmp_path):
    path = tmp_path / "osm.json"
    path.write_text(json.dumps(_osm_doc()))
    return path


def test_fr_in_5_read_osm(osm_file):
    frame = unit_frame(LON, LAT)
    edges = read_osm(osm_file, frame)
    assert [e.feature_id for e in edges] == ["10", "30", "50", "70"]
    assert all(e.source == "osm" for e in edges)
    mw = edges[0]
    assert mw.tags == {"highway": "motorway", "ref": "I 190", "lanes": "2"}
    assert edges[1].tags["layer"] == "1"
    assert mw.coords.shape == (2, 2)
    np.testing.assert_allclose(mw.coords[:, 0], [-300.0, 300.0], atol=0.5)
    np.testing.assert_allclose(mw.coords[:, 1], [0.0, 0.0], atol=0.5)
    lon, _ = FrameTransformer(frame).unit_to_lonlat(mw.coords[:, 0], mw.coords[:, 1])
    np.testing.assert_allclose(lon, [LON - 300 * DLON, LON + 300 * DLON], atol=1e-9)


def test_fr_in_5_read_osm_geom_output(tmp_path):
    doc = {"elements": [{"type": "way", "id": 1, "tags": {"highway": "motorway"},
                         "geometry": [{"lon": LON, "lat": LAT}, {"lon": LON + 0.001, "lat": LAT}]}]}
    path = tmp_path / "geom.json"
    path.write_text(json.dumps(doc))
    edges = read_osm(path, unit_frame(LON, LAT))
    assert len(edges) == 1 and edges[0].coords.shape == (2, 2)


def test_fr_in_7_source_switch(osm_file):
    frame = unit_frame(LON, LAT)
    on = read_vectors({"osm": {"enabled": True, "path": osm_file}}, frame)
    assert list(on) == ["osm"] and len(on["osm"]) == 4
    off = read_vectors({"osm": {"enabled": False, "path": osm_file}}, frame)
    assert off == {}
    # A disabled source needs no reader; an enabled one does.
    assert read_vectors({"tiger": {"enabled": False, "path": "x"}}, frame) == {}
    with pytest.raises(ValueError, match="tiger"):
        read_vectors({"tiger": {"enabled": True, "path": "x"}}, frame)


def test_scope_in_scope_buffer(osm_file):
    edges = read_osm(osm_file, unit_frame(LON, LAT))
    kept = [e.feature_id for e in in_scope(edges)]
    assert kept == ["10", "30", "50"]
    assert [e.feature_id for e in in_scope(edges, buffer_m=50.0)] == ["10", "30"]


def test_scope_limited_access_classes():
    assert is_limited_access({"highway": "motorway"})
    assert is_limited_access({"highway": "motorway_link"})
    assert is_limited_access({"highway": "trunk_link"})
    assert is_limited_access({"highway": "trunk", "motorroad": "yes"})
    assert is_limited_access({"highway": "trunk", "expressway": "yes"})
    assert not is_limited_access({"highway": "trunk"})
    assert not is_limited_access({"highway": "primary", "motorroad": "yes"})
    plain = [VectorEdge("osm", "1", np.array([[0.0, 0.0], [1.0, 0.0]]), {"highway": "trunk"})]
    assert in_scope(plain) == []
