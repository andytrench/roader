"""ASAM OpenDRIVE 1.8 export (FR-DM-1)."""

import json
import xml.etree.ElementTree as ET

import numpy as np

from roader.export import lane_layout, read_xodr_geometry, xodr_bytes
from roader.model.serialize import to_plain


def test_fr_dm_1_parses_with_one_road_per_model_road(model):
    root = ET.fromstring(xodr_bytes(model))
    assert root.tag == "OpenDRIVE"
    header = root.find("header")
    assert (header.get("revMajor"), header.get("revMinor")) == ("1", "8")
    assert header.find("geoReference").text == model.frame.proj
    roads = {r.get("id"): r for r in root.findall("road")}
    assert set(roads) == {r.id for r in model.roads}


def test_fr_dm_1_plan_view_and_elevation_counts(model):
    root = ET.fromstring(xodr_bytes(model))
    roads = {r.get("id"): r for r in root.findall("road")}
    for road in model.roads:
        el = roads[road.id]
        geoms = el.find("planView").findall("geometry")
        assert len(geoms) == len(road.alignment.elements)
        for g, e in zip(geoms, road.alignment.elements, strict=True):
            assert g[0].tag == e.kind
        elev = el.find("elevationProfile").findall("elevation")
        assert len(elev) == len(road.profile.elements)
        for rec, v in zip(elev, road.profile.elements, strict=True):
            assert float(rec.get("s")) == v.s0
            assert float(rec.get("a")) == v.z0
            assert float(rec.get("b")) == v.g0
            assert abs(float(rec.get("c")) - (v.g1 - v.g0) / (2 * v.length)) < 1e-12
            assert float(rec.get("d")) == 0.0


def test_fr_dm_1_round_trip_alignment_and_profile_within_1e_6(model):
    back = read_xodr_geometry(xodr_bytes(model))
    kinds = set()
    for road in model.roads:
        alignment, profile = back[road.id]
        kinds.update(e.kind for e in road.alignment.elements)
        for s in np.linspace(0.0, road.length, 97):
            x0, y0, h0, k0 = road.alignment.evaluate(float(s))
            x1, y1, h1, k1 = alignment.evaluate(float(s))
            assert abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6
            assert abs(h0 - h1) < 1e-6 and abs(k0 - k1) < 1e-6
            assert abs(road.profile.z(float(s)) - profile.z(float(s))) < 1e-6
            assert abs(road.profile.grade(float(s)) - profile.grade(float(s))) < 1e-6
    assert kinds == {"line", "arc", "spiral"}


def test_fr_dm_1_bridge_object_per_structure_on_carried_road(model):
    root = ET.fromstring(xodr_bytes(model))
    bridges = {}
    for road_el in root.findall("road"):
        for b in road_el.findall("objects/bridge"):
            bridges[b.get("id")] = (road_el.get("id"), b)
    assert set(bridges) == {st.id for st in model.structures}
    for st in model.structures:
        road_id, b = bridges[st.id]
        assert road_id == st.carried_road
        assert abs(float(b.get("s")) - st.s_start) < 1e-9
        assert abs(float(b.get("length")) - (st.s_end - st.s_start)) < 1e-9


def test_fr_dm_1_lanes_sum_to_pavement_width(model):
    for road in model.roads:
        sections = lane_layout(road)
        xs = road.cross_section
        for i, sec in enumerate(sections):
            end = sections[i + 1]["s"] if i + 1 < len(sections) else road.length
            for s in np.linspace(sec["s"], end, 5):
                ds = s - sec["s"]
                left = sum(a + b * ds for _id, _t, a, b in sec["left"])
                right = sum(a + b * ds for _id, _t, a, b in sec["right"])
                assert abs(left - xs.width_left(float(s))) < 1e-9
                assert abs(right - xs.width_right(float(s))) < 1e-9


def test_fr_dm_1_links_junctions_and_lossless_user_data(model):
    root = ET.fromstring(xodr_bytes(model))
    roads = {r.get("id"): r for r in root.findall("road")}
    link = model.links[0]
    succ = roads[link.from_road].find("link/successor")
    assert succ.get("elementId") == link.to_road and succ.get("contactPoint") == link.to_end
    pred = roads[link.to_road].find("link/predecessor")
    assert pred.get("elementId") == link.from_road
    junctions = root.findall("junction")
    assert [j.get("id") for j in junctions] == [j.id for j in model.junctions]
    assert junctions[0].find("connection").get("incomingRoad") == link.from_road
    # userData carries each road's full L0 to L3 record at native precision (D-011): lossless.
    for road in model.roads:
        payload = json.loads(roads[road.id].find("userData").text)
        assert payload == json.loads(json.dumps(to_plain(road)))
