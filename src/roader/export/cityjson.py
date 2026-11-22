"""CityJSON 2.0 writer: solids and semantics (FR-DM-2, FR-EX-3, FR-RH-2).

One CityObject per model element, with the model id as the CityObject id:

- Road (Transportation): each model road; its pavement meshes as CompositeSurface
  shells with TrafficArea on upward faces. CityJSON 2.0 does not allow Solid on
  Transportation objects, so the closed pavement solid is written as the
  CompositeSurface of its single closed shell (attribute ``closed_shell``).
- Road children (``parents`` = the road): side slopes (L4 earthwork) and junction
  areas (L6) the same way.
- Bridge: each structure (kind bridge or culvert), with BridgeConstructiveElement
  children for each deck/superstructure mesh (id = mesh id), each support
  (Solid, or MultiSolid when columns and cap are separate meshes) and each parapet.
  Tunnel portals map to Tunnel / TunnelConstructiveElement.
- CityFurniture: barriers that are not on a structure.
- TINRelief: the L7 terrain surface as a CompositeSurface of triangles.
- GenericCityObject: any mesh whose element id is not in the model (logged in
  ``attributes.unmatched``).

Vertices are integers with transform scale 0.001 and an integer-meter translate.
Coordinates are in the unit frame (D-007); see docs/EXPORTS.md for the CRS note.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from roader.export._common import (
    ElementInfo,
    check_meshes,
    element_confidence,
    element_index,
    mesh_info,
    rounded,
    unit_header,
)
from roader.model.mesh import Mesh
from roader.model.schema import RoadModel

SCALE = 0.001
LOD = "2"

# CityJSON 2.0 geometry types allowed per CityObject type (from the 2.0 schema).
_ANY = {"MultiPoint", "MultiLineString", "MultiSurface", "CompositeSurface", "Solid", "CompositeSolid", "MultiSolid"}
ALLOWED_GEOMETRY: dict[str, set[str]] = {
    "Road": {"MultiLineString", "MultiSurface", "CompositeSurface"},
    "Bridge": {"MultiSurface", "CompositeSurface", "Solid", "CompositeSolid"},
    "BridgePart": {"MultiSurface", "CompositeSurface", "Solid", "CompositeSolid"},
    "BridgeConstructiveElement": _ANY,
    "BridgeInstallation": _ANY,
    "Tunnel": {"MultiSurface", "CompositeSurface", "Solid", "CompositeSolid"},
    "TunnelConstructiveElement": _ANY,
    "TINRelief": {"CompositeSurface"},
    "CityFurniture": _ANY,
    "GenericCityObject": _ANY,
}
SECOND_LEVEL = {"BridgePart", "BridgeConstructiveElement", "BridgeInstallation", "TunnelConstructiveElement"}
TRANSPORTATION_SEMANTICS = {"TrafficArea", "AuxiliaryTrafficArea", "TransportationMarking", "TransportationHole"}

_UP_FACE = 0.7  # |normal z| above this counts as an upward traffic face


class _Vertices:
    """Global integer vertex pool, first-seen order (deterministic for sorted meshes)."""

    def __init__(self, translate: np.ndarray) -> None:
        self.translate = translate
        self.index: dict[tuple[int, int, int], int] = {}
        self.items: list[list[int]] = []

    def add_mesh(self, mesh: Mesh) -> list[int]:
        ints = np.rint((mesh.vertices - self.translate) / SCALE).astype(np.int64)
        out = []
        for row in ints:
            key = (int(row[0]), int(row[1]), int(row[2]))
            if key not in self.index:
                self.index[key] = len(self.items)
                self.items.append(list(key))
            out.append(self.index[key])
        return out


def _faces(mesh: Mesh, vmap: list[int]) -> list[list[list[int]]]:
    faces = []
    for a, b, c in mesh.faces:
        tri = [vmap[int(a)], vmap[int(b)], vmap[int(c)]]
        if len(set(tri)) < 3:
            raise ValueError(f"mesh {mesh.id} has a face collapsing below 1 mm (FR-SV-5)")
        faces.append([tri])
    return faces


def _up_faces(mesh: Mesh) -> np.ndarray:
    v = mesh.vertices
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    length[length == 0.0] = 1.0
    return (n[:, 2] / length) > _UP_FACE


def _extent(meshes: list[Mesh]) -> list[float]:
    v = np.vstack([m.vertices for m in meshes])
    lo, hi = v.min(axis=0), v.max(axis=0)
    return [rounded(x, 3) for x in (*lo, *hi)]


def _attributes(info: ElementInfo | None, meshes: list[Mesh], **extra: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if info is not None:
        attrs["element_id"] = info.id
        if info.prov is not None:
            attrs["source"] = info.prov.source.value
            if info.prov.rule_id:
                attrs["rule_id"] = info.prov.rule_id
            if info.prov.flags:
                attrs["provenance_flags"] = sorted(info.prov.flags)
        confidence = element_confidence(info)
        if confidence is not None:
            attrs["confidence"] = rounded(confidence, 6)
    if meshes:
        attrs["mesh_ids"] = [m.id for m in meshes]
        attrs["mesh_sources"] = sorted({m.source.value for m in meshes})
        if "source" not in attrs and len(attrs["mesh_sources"]) == 1:
            attrs["source"] = attrs["mesh_sources"][0]
        layers = sorted({m.layer for m in meshes})
        attrs["layer"] = layers[0] if len(layers) == 1 else layers
    for key, value in extra.items():
        if value is not None:
            attrs[key] = value
    return attrs


def cityjson_document(model: RoadModel, meshes: list[Mesh]) -> dict[str, Any]:
    meshes = check_meshes(meshes)
    index = element_index(model)
    if meshes:
        lo = np.floor(np.vstack([m.vertices for m in meshes]).min(axis=0))
    else:
        lo = np.zeros(3)
    pool = _Vertices(lo)
    vmaps = {m.id: pool.add_mesh(m) for m in meshes}

    by_element: dict[str, list[Mesh]] = {}
    for mesh in meshes:
        by_element.setdefault(mesh.element_id, []).append(mesh)

    def composite(mesh: Mesh, semantic: str | None, function: str | None = None) -> dict[str, Any]:
        geom: dict[str, Any] = {"type": "CompositeSurface", "lod": LOD, "boundaries": _faces(mesh, vmaps[mesh.id])}
        if semantic is not None:
            surface: dict[str, Any] = {"type": semantic}
            if function:
                surface["function"] = function
            up = _up_faces(mesh)
            geom["semantics"] = {"surfaces": [surface], "values": [0 if flag else None for flag in up]}
        return geom

    def solid(group: list[Mesh]) -> dict[str, Any]:
        shells = [[_faces(m, vmaps[m.id])] for m in group]
        if len(shells) == 1:
            return {"type": "Solid", "lod": LOD, "boundaries": shells[0]}
        return {"type": "MultiSolid", "lod": LOD, "boundaries": shells}

    objects: dict[str, dict[str, Any]] = {}

    def put(oid: str, obj: dict[str, Any], group: list[Mesh]) -> None:
        if oid in objects:
            raise ValueError(f"duplicate CityObject id {oid}")
        if group:
            obj["geographicalExtent"] = _extent(group)
        objects[oid] = obj

    def link(parent: str, child: str) -> None:
        objects[child].setdefault("parents", []).append(parent)
        objects[parent].setdefault("children", []).append(child)

    claimed: set[str] = set()

    for road in sorted(model.roads, key=lambda r: r.id):
        group = by_element.get(road.id, [])
        claimed.update(m.id for m in group)
        geoms = [composite(m, "TrafficArea" if m.layer == "L3" else None) for m in group]
        attrs = _attributes(
            index[road.id], group, name=road.name, road_class=road.road_class, kind=road.kind,
            oneway=road.oneway, length_m=rounded(road.length, 3), closed_shell=all(m.closed for m in group),
        )
        put(road.id, {"type": "Road", "attributes": attrs, **({"geometry": geoms} if geoms else {})}, group)

    for slope in sorted(model.side_slopes, key=lambda s: s.id):
        group = by_element.get(slope.id, [])
        claimed.update(m.id for m in group)
        geoms = [composite(m, None) for m in group]
        attrs = _attributes(
            index[slope.id], group, function="earthwork", road_id=slope.road_id, side=slope.side,
            slope_kind=slope.kind, ratio=rounded(slope.ratio, 6), s0=rounded(slope.s0, 3), s1=rounded(slope.s1, 3),
            closed_shell=all(m.closed for m in group),
        )
        put(slope.id, {"type": "Road", "attributes": attrs, **({"geometry": geoms} if geoms else {})}, group)
        if slope.road_id in objects:
            link(slope.road_id, slope.id)

    for junction in sorted(model.junctions, key=lambda j: j.id):
        group = by_element.get(junction.id, [])
        claimed.update(m.id for m in group)
        geoms = [composite(m, "TrafficArea") for m in group]
        attrs = _attributes(index[junction.id], group, function="junction", road_ids=sorted(junction.road_ids))
        put(junction.id, {"type": "Road", "attributes": attrs, **({"geometry": geoms} if geoms else {})}, group)

    for structure in sorted(model.structures, key=lambda s: s.id):
        tunnel = structure.kind == "tunnel_portal"
        top_type, part_type = ("Tunnel", "TunnelConstructiveElement") if tunnel else ("Bridge", "BridgeConstructiveElement")
        attrs = _attributes(
            index[structure.id], [], kind=structure.kind, family=structure.family,
            carried_road=structure.carried_road, crossed_roads=sorted(structure.crossed_roads),
            s_start=rounded(structure.s_start, 3), s_end=rounded(structure.s_end, 3),
            year_built=structure.year_built, inventory_id=structure.inventory_id, review=structure.review,
            deck_depth_m=rounded(structure.deck.depth_m, 3), deck_source=structure.deck.prov.source.value,
        )
        put(structure.id, {"type": top_type, "attributes": attrs}, [])
        for mesh in by_element.get(structure.id, []):
            claimed.add(mesh.id)
            child = {
                "type": part_type,
                "attributes": _attributes(index[structure.id], [mesh], function=mesh.part),
                "geometry": [solid([mesh]) if mesh.closed else composite(mesh, None)],
            }
            put(mesh.id, child, [mesh])
            link(structure.id, mesh.id)
        for support in sorted(structure.supports, key=lambda s: s.id):
            group = by_element.get(support.id, [])
            claimed.update(m.id for m in group)
            child = {
                "type": part_type,
                "attributes": _attributes(
                    index[support.id], group, function=support.kind, station=rounded(support.station, 3),
                    skew_rad=rounded(support.skew_rad, 9), n_columns=support.n_columns,
                ),
            }
            if group:
                child["geometry"] = [solid(group)]
            put(support.id, child, group)
            link(structure.id, support.id)

    for road in sorted(model.roads, key=lambda r: r.id):
        for barrier in sorted(road.cross_section.barriers, key=lambda b: b.id):
            info = index[barrier.id]
            group = by_element.get(barrier.id, [])
            claimed.update(m.id for m in group)
            attrs = _attributes(
                info, group, function=barrier.kind, road_id=road.id, side=barrier.side,
                s0=rounded(barrier.s0, 3), s1=rounded(barrier.s1, 3), height_m=rounded(barrier.height_m, 3),
            )
            on_structure = info.structure_id is not None
            if on_structure:
                tunnel = objects[info.structure_id]["type"] == "Tunnel"
                kind = "TunnelConstructiveElement" if tunnel else "BridgeConstructiveElement"
            else:
                kind = "CityFurniture"
            obj: dict[str, Any] = {"type": kind, "attributes": attrs}
            if group:
                obj["geometry"] = [solid(group)]
            put(barrier.id, obj, group)
            if on_structure:
                link(info.structure_id, barrier.id)

    terrain = [m for m in meshes if m.layer == "L7"]
    if terrain:
        claimed.update(m.id for m in terrain)
        tid = terrain[0].element_id
        prov = model.terrain.prov if model.terrain is not None else None
        attrs = _attributes(ElementInfo(tid, "terrain", prov), terrain, function="terrain")
        put(tid, {"type": "TINRelief", "attributes": attrs, "geometry": [composite(m, None) for m in terrain]}, terrain)

    for mesh in meshes:
        if mesh.id in claimed:
            continue
        info = mesh_info(mesh, index)
        attrs = _attributes(info, [mesh], function=mesh.part, unmatched=info is None, element_ref=mesh.element_id)
        geom = solid([mesh]) if mesh.closed else composite(mesh, None)
        put(mesh.id, {"type": "GenericCityObject", "attributes": attrs, "geometry": [geom]}, [mesh])

    for obj in objects.values():
        if "children" in obj:
            obj["children"] = sorted(obj["children"])

    header = unit_header(model)
    metadata: dict[str, Any] = {
        "identifier": model.unit.unit_id,
        "title": f"Roader unit {model.unit.unit_id}",
        "roaderFrame": header["frame"],
        "roaderModelHash": header["model_hash"],
    }
    if meshes:
        metadata["geographicalExtent"] = _extent(meshes)
    return {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [SCALE, SCALE, SCALE], "translate": [float(x) for x in lo]},
        "metadata": metadata,
        "CityObjects": objects,
        "vertices": pool.items,
    }


def cityjson_bytes(model: RoadModel, meshes: list[Mesh]) -> bytes:
    doc = cityjson_document(model, meshes)
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_cityjson(model: RoadModel, meshes: list[Mesh], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cityjson_bytes(model, meshes))
    return path


def validate_cityjson(doc: dict[str, Any]) -> list[str]:
    """Structural checks used by tests and publish: returns a list of problems (empty = valid)."""
    problems: list[str] = []
    if doc.get("type") != "CityJSON" or doc.get("version") != "2.0":
        problems.append("type/version")
    transform = doc.get("transform", {})
    if len(transform.get("scale", [])) != 3 or len(transform.get("translate", [])) != 3:
        problems.append("transform")
    vertices = doc.get("vertices", [])
    for v in vertices:
        if len(v) != 3 or not all(isinstance(c, int) for c in v):
            problems.append("non-integer vertex")
            break
    n = len(vertices)
    objects = doc.get("CityObjects", {})

    def walk(node: Any, depth: int) -> int:
        if depth == 0:
            if not isinstance(node, int) or not 0 <= node < n:
                problems.append(f"bad vertex index {node}")
            return 1
        return sum(walk(item, depth - 1) for item in node)

    depth_of = {"MultiSurface": 3, "CompositeSurface": 3, "Solid": 4, "MultiSolid": 5, "CompositeSolid": 5}
    for oid, obj in objects.items():
        kind = obj.get("type")
        if kind not in ALLOWED_GEOMETRY:
            problems.append(f"{oid}: type {kind} not in Transportation/Bridge/Tunnel/Relief/Generic modules")
            continue
        for geom in obj.get("geometry", []):
            gtype = geom.get("type")
            if gtype not in ALLOWED_GEOMETRY[kind]:
                problems.append(f"{oid}: geometry {gtype} not allowed for {kind}")
                continue
            walk(geom["boundaries"], depth_of[gtype])
            semantics = geom.get("semantics")
            if semantics:
                surfaces = semantics["surfaces"]
                for s in surfaces:
                    if kind == "Road" and s["type"] not in TRANSPORTATION_SEMANTICS:
                        problems.append(f"{oid}: semantic {s['type']} not valid for Road")
                values = semantics["values"]
                if gtype in {"MultiSurface", "CompositeSurface"} and len(values) != len(geom["boundaries"]):
                    problems.append(f"{oid}: semantics length")
                for value in values:
                    if value is not None and not 0 <= value < len(surfaces):
                        problems.append(f"{oid}: semantic index {value}")
        if kind in SECOND_LEVEL and not obj.get("parents"):
            problems.append(f"{oid}: {kind} needs a parent")
        for parent in obj.get("parents", []):
            if parent not in objects or oid not in objects[parent].get("children", []):
                problems.append(f"{oid}: parent {parent} does not list it")
        for child in obj.get("children", []):
            if child not in objects or oid not in objects[child].get("parents", []):
                problems.append(f"{oid}: child {child} does not point back")
        extent = obj.get("geographicalExtent")
        if extent is not None and (len(extent) != 6 or any(not math.isfinite(x) for x in extent)):
            problems.append(f"{oid}: geographicalExtent")
    return problems
