"""ASAM OpenDRIVE 1.8 (.xodr) writer for layers L0 to L3 and L6 (FR-DM-1, FR-EX-3).

Mapping (details and limits in docs/EXPORTS.md):

- header: unit id, extent, ``geoReference`` = the unit frame PROJ string (D-007).
- road: one per model road, id = model road id. Links from ``model.links``.
- planView: one ``geometry`` record per AlignElement (line, arc curvature, spiral
  curvStart/curvEnd), exact.
- elevationProfile: one ``elevation`` per VerticalElement: a = z0, b = g0,
  c = (g1 - g0) / (2 L), d = 0, exact.
- lateralProfile: the two cross slopes split into banking (mean slope, written as
  ``superelevation`` roll = atan(mean)) and crown (half the slope difference,
  written as two linear ``shape`` polynomials). The roll is linear between slope
  knots, the crown exact in the rolled plane; both are first-order approximations
  of the horizontal-offset model, which userData carries exactly.
- lanes: a laneSection at every width knot and lane-group change; per side inner
  shoulder, driving lanes and outer shoulder, scaled so their widths sum to the
  pavement width; one driving lane per side of the full width when lane counts are
  unknown.
- objects: barriers as continuous ``object``/``repeat``; each structure as a
  ``bridge`` (``tunnel`` for portals) on the carried road.
- junctions: one per JunctionArea, connections from model links between its roads.
- userData (code "roader"): canonical JSON of each road, structure, junction and of
  the unit's links/crossings, so L0 to L3 and L6 are lossless (D-011 rounding).
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from roader.export._common import num, unit_header
from roader.model.geometry import AlignElement, Alignment, Profile, VerticalElement
from roader.model.provenance import Provenance, Source
from roader.model.schema import LaneGroup, Road, RoadModel
from roader.model.serialize import to_plain

REV_MAJOR = "1"
REV_MINOR = "8"
_EPS = 1e-6


def _attrs(el: ET.Element, **values: Any) -> ET.Element:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            el.set(key, "true" if value else "false")
        elif isinstance(value, float):
            el.set(key, num(value))
        else:
            el.set(key, str(value))
    return el


def _sub(parent: ET.Element, tag: str, **values: Any) -> ET.Element:
    return _attrs(ET.SubElement(parent, tag), **values)


def _user_data(parent: ET.Element, value: str, payload: Any) -> None:
    el = _sub(parent, "userData", code="roader", value=value)
    el.text = json.dumps(to_plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stations(length: float, *groups: list[float]) -> list[float]:
    """0 plus every interior station from groups, merged within 1e-6 m."""
    out = [0.0]
    for s in sorted(s for group in groups for s in group):
        if _EPS < s < length - _EPS and s - out[-1] > _EPS:
            out.append(float(s))
    return out


def _road_type(road_class: str) -> str:
    return "motorway" if road_class.startswith("motorway") else "unknown"


def _links(model: RoadModel, road_id: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(predecessors, successors) of a road as (other road id, contact point)."""
    pred: list[tuple[str, str]] = []
    succ: list[tuple[str, str]] = []
    for link in sorted(model.links, key=lambda lk: lk.id):
        if link.from_road == road_id:
            (succ if link.from_end == "end" else pred).append((link.to_road, link.to_end))
        if link.to_road == road_id:
            (succ if link.to_end == "end" else pred).append((link.from_road, link.from_end))
    return pred, succ


def _plan_view(road_el: ET.Element, road: Road) -> None:
    plan = _sub(road_el, "planView")
    for e in road.alignment.elements:
        geom = _sub(plan, "geometry", s=e.s0, x=e.x0, y=e.y0, hdg=e.heading0, length=e.length)
        if e.kind == "line":
            _sub(geom, "line")
        elif e.kind == "arc":
            _sub(geom, "arc", curvature=e.k0)
        else:
            _sub(geom, "spiral", curvStart=e.k0, curvEnd=e.k1)


def _elevation(road_el: ET.Element, road: Road) -> None:
    prof = _sub(road_el, "elevationProfile")
    for e in road.profile.elements:
        _sub(prof, "elevation", s=e.s0, a=e.z0, b=e.g0, c=(e.g1 - e.g0) / (2.0 * e.length), d=0.0)


def _lateral(road_el: ET.Element, road: Road) -> None:
    xs = road.cross_section
    stations = _stations(road.length, xs.slope_left.breakpoints(), xs.slope_right.breakpoints())
    lateral = _sub(road_el, "lateralProfile")
    rolls = [math.atan(0.5 * (xs.slope_left(s) + xs.slope_right(s))) for s in stations]
    for i, s in enumerate(stations):
        b = (rolls[i + 1] - rolls[i]) / (stations[i + 1] - s) if i + 1 < len(stations) else 0.0
        _sub(lateral, "superelevation", s=s, a=rolls[i], b=b, c=0.0, d=0.0)
    reach = math.ceil(max([k.value for k in xs.width_right.knots] + [0.0])) + 1.0
    for s in stations:
        crown = 0.5 * (xs.slope_right(s) - xs.slope_left(s))
        _sub(lateral, "shape", s=s, t=-reach, a=-crown * reach, b=crown, c=0.0, d=0.0)
        _sub(lateral, "shape", s=s, t=0.0, a=0.0, b=-crown, c=0.0, d=0.0)


def _group_at(groups: list[LaneGroup], side: str, s: float) -> LaneGroup | None:
    for g in sorted(groups, key=lambda g: (g.s0, g.s1, g.count)):
        if g.side == side and g.s0 <= s <= g.s1:
            return g
    return None


def _side_lanes(groups: list[LaneGroup], side: str, s_mid: float) -> list[tuple[str, float]]:
    """Lane types and nominal widths from the reference line outward."""
    group = _group_at(groups, side, s_mid)
    if group is None or group.count <= 0:
        return [("driving", 1.0)]
    lanes: list[tuple[str, float]] = []
    if group.shoulder_inner_m > 0.0:
        lanes.append(("shoulder", group.shoulder_inner_m))
    lanes += [("driving", group.lane_width_m)] * group.count
    if group.shoulder_outer_m > 0.0:
        lanes.append(("shoulder", group.shoulder_outer_m))
    return lanes


def lane_layout(road: Road) -> list[dict[str, Any]]:
    """Lane sections as plain data: s, and per side a list of (id, type, a, b) widths."""
    xs = road.cross_section
    length = road.length
    lane_breaks = [v for g in xs.lanes for v in (g.s0, g.s1)]
    stations = _stations(length, xs.width_left.breakpoints(), xs.width_right.breakpoints(), lane_breaks)
    sections = []
    for i, s in enumerate(stations):
        end = stations[i + 1] if i + 1 < len(stations) else length
        section: dict[str, Any] = {"s": s, "left": [], "right": []}
        for side, width, sign in (("left", xs.width_left, 1), ("right", xs.width_right, -1)):
            wa, wb = width(s), width(end)
            if max(wa, wb) <= 0.0:
                continue
            slope = (wb - wa) / (end - s)
            lanes = _side_lanes(xs.lanes, side, 0.5 * (s + end))
            total = sum(w for _t, w in lanes)
            for n, (kind, nominal) in enumerate(lanes, start=1):
                f = nominal / total
                section[side].append((sign * n, kind, wa * f, slope * f))
        sections.append(section)
    return sections


def _lanes(road_el: ET.Element, road: Road) -> None:
    lanes_el = _sub(road_el, "lanes")
    sections = lane_layout(road)
    for i, section in enumerate(sections):
        prev_ids = {lane[0] for side in ("left", "right") for lane in sections[i - 1][side]} if i > 0 else set()
        next_ids = (
            {lane[0] for side in ("left", "right") for lane in sections[i + 1][side]} if i + 1 < len(sections) else set()
        )
        sec_el = _sub(lanes_el, "laneSection", s=section["s"])
        for side in ("left", "center", "right"):
            side_el = _sub(sec_el, side)
            if side == "center":
                _sub(side_el, "lane", id=0, type="none", level=False)
                continue
            for lane_id, kind, a, b in sorted(section[side], key=lambda lane: -lane[0]):
                direction = "reversed" if side == "left" and road.oneway else None
                lane_el = _sub(side_el, "lane", id=lane_id, type=kind, level=False, direction=direction)
                if lane_id in prev_ids or lane_id in next_ids:
                    link_el = _sub(lane_el, "link")
                    if lane_id in prev_ids:
                        _sub(link_el, "predecessor", id=lane_id)
                    if lane_id in next_ids:
                        _sub(link_el, "successor", id=lane_id)
                _sub(lane_el, "width", sOffset=0.0, a=a, b=b, c=0.0, d=0.0)


def _objects(road_el: ET.Element, road: Road, model: RoadModel) -> None:
    barriers = sorted(road.cross_section.barriers, key=lambda b: b.id)
    structures = sorted((st for st in model.structures if st.carried_road == road.id), key=lambda st: st.id)
    if not barriers and not structures:
        return
    objects = _sub(road_el, "objects")
    for barrier in barriers:
        sign = 1.0 if barrier.side == "left" else -1.0
        t = sign * (barrier.offset_m + 0.5 * barrier.base_width_m)
        length = barrier.s1 - barrier.s0
        obj = _sub(
            objects, "object", id=barrier.id, name=barrier.kind, type="barrier", s=barrier.s0, t=t,
            zOffset=0.0, orientation="none", height=barrier.height_m, width=barrier.base_width_m,
        )
        _sub(
            obj, "repeat", s=barrier.s0, length=length, distance=0.0, tStart=t, tEnd=t,
            heightStart=barrier.height_m, heightEnd=barrier.height_m, zOffsetStart=0.0, zOffsetEnd=0.0,
            widthStart=barrier.base_width_m, widthEnd=barrier.base_width_m,
        )
    for st in (st for st in structures if st.kind == "tunnel_portal"):
        el = _sub(objects, "tunnel", s=st.s_start, length=st.s_end - st.s_start, name=st.inventory_id or st.id,
                  id=st.id, type="standard")
        _user_data(el, f"structure:{st.id}", st)
    for st in (st for st in structures if st.kind != "tunnel_portal"):
        kind = "steel" if st.family == "truss" else "concrete"
        el = _sub(objects, "bridge", s=st.s_start, length=st.s_end - st.s_start, name=st.inventory_id or st.id,
                  id=st.id, type=kind)
        _user_data(el, f"structure:{st.id}", st)


def _extent(model: RoadModel) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for road in model.roads:
        length = road.length
        count = max(2, math.ceil(length / 10.0) + 1)
        stations = [length * i / (count - 1) for i in range(count)]
        stations += [e.s0 for e in road.alignment.elements]
        for s in stations:
            x, y, _h, _k = road.alignment.evaluate(s)
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    return max(ys), min(ys), max(xs), min(xs)


def xodr_tree(model: RoadModel) -> ET.Element:
    root = ET.Element("OpenDRIVE")
    north, south, east, west = _extent(model)
    header = _sub(
        root, "header", revMajor=REV_MAJOR, revMinor=REV_MINOR, name=model.unit.unit_id,
        version=model.unit.software_version, north=north, south=south, east=east, west=west, vendor="Roader",
    )
    geo = ET.SubElement(header, "geoReference")
    geo.text = model.frame.proj
    _user_data(
        header,
        f"unit:{model.unit.unit_id}",
        {
            **unit_header(model),
            "links": sorted(model.links, key=lambda lk: lk.id),
            "crossings": sorted(model.crossings, key=lambda c: c.id),
        },
    )

    for road in sorted(model.roads, key=lambda r: r.id):
        road_el = _sub(root, "road", name=road.name, length=road.length, id=road.id, junction="-1", rule="RHT")
        pred, succ = _links(model, road.id)
        if pred or succ:
            link_el = _sub(road_el, "link")
            if pred:
                other, contact = pred[0]
                _sub(link_el, "predecessor", elementType="road", elementId=other, contactPoint=contact)
            if succ:
                other, contact = succ[0]
                _sub(link_el, "successor", elementType="road", elementId=other, contactPoint=contact)
        _sub(road_el, "type", s=0.0, type=_road_type(road.road_class))
        _plan_view(road_el, road)
        _elevation(road_el, road)
        _lateral(road_el, road)
        _lanes(road_el, road)
        _objects(road_el, road, model)
        _user_data(road_el, f"road:{road.id}", road)

    for junction in sorted(model.junctions, key=lambda j: j.id):
        j_el = _sub(root, "junction", id=junction.id, name=junction.id, type="default")
        members = set(junction.road_ids)
        for link in sorted(model.links, key=lambda lk: lk.id):
            if link.from_road in members and link.to_road in members:
                _sub(j_el, "connection", id=link.id, incomingRoad=link.from_road, connectingRoad=link.to_road,
                     contactPoint=link.to_end)
        _user_data(j_el, f"junction:{junction.id}", junction)
    return root


def xodr_bytes(model: RoadModel) -> bytes:
    root = xodr_tree(model)
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + text + "\n").encode("utf-8")


def write_xodr(model: RoadModel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xodr_bytes(model))
    return path


# ---------------------------------------------------------------- read back


_READ_PROV = Provenance(Source.RULE, 0.0, rule_id="xodr")


def read_xodr_geometry(data: bytes) -> dict[str, tuple[Alignment, Profile]]:
    """Rebuild each road's L1 alignment and L2 profile from planView and elevationProfile."""
    root = ET.fromstring(data)
    out: dict[str, tuple[Alignment, Profile]] = {}
    for road_el in root.findall("road"):
        elements = []
        for g in road_el.find("planView").findall("geometry"):
            s, x, y = float(g.get("s")), float(g.get("x")), float(g.get("y"))
            hdg, length = float(g.get("hdg")), float(g.get("length"))
            kind_el = g[0]
            if kind_el.tag == "line":
                kind, k0, k1 = "line", 0.0, 0.0
            elif kind_el.tag == "arc":
                kind = "arc"
                k0 = k1 = float(kind_el.get("curvature"))
            else:
                kind, k0, k1 = "spiral", float(kind_el.get("curvStart")), float(kind_el.get("curvEnd"))
            elements.append(AlignElement(kind, s, length, x, y, hdg, k0, k1, _READ_PROV))
        records = road_el.find("elevationProfile").findall("elevation")
        road_length = float(road_el.get("length"))
        vertical = []
        for i, e in enumerate(records):
            s = float(e.get("s"))
            end = float(records[i + 1].get("s")) if i + 1 < len(records) else road_length
            length = end - s
            a, b, c = float(e.get("a")), float(e.get("b")), float(e.get("c"))
            vertical.append(VerticalElement(s, length, a, b, b + 2.0 * c * length, _READ_PROV))
        out[road_el.get("id")] = (Alignment(elements), Profile(vertical))
    return out
