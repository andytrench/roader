"""Binary glTF 2.0 (.glb) writer, standard library plus numpy (FR-EX-3, FR-AS-10).

One node per Mesh, named with the stable mesh id, carrying extras {element_id,
layer, part, material, source, mesh_id, confidence} (FR-DM-3, AC-H6). One material
per semantic slot, metallic-roughness PBR with flat base colors (FR-AS-6). The
provenance variant colors every primitive by its source instead (FR-RH-1).

Frame: positions are float32 in the tile-local frame, whose origin is the unit
frame origin (the D-007 transverse Mercator anchor). glTF is Y-up right-handed,
so a unit-frame point (x east, y north, z up) is written as (x, z, -y). This is a
proper rotation, so face winding (counter-clockwise from outside) is unchanged.
The geodetic anchor and PROJ string are in ``asset.extras``.

Meshes that carry FR-AS-3..5 attributes are written with NORMAL (rotated like the
positions), TEXCOORD_0 (uv0, metres) and TEXCOORD_1 (uv1 atlas). Per-corner
attributes are unwelded: one glTF vertex per distinct (position, normal, uv0, uv1)
in first-use order, so hard edges and UV seams survive. UVs are written as stored
(v not flipped to image space); ``asset.extras.texcoords`` says so.

LODs (FR-AS-1): with ``lods`` given, each level-0 node lists its coarser versions
through ``MSFT_lod`` (nodes named ``<mesh id>#lod<k>``, outside the scene roots, as
that extension requires), with ``MSFT_screencoverage`` thresholds in its extras.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from roader.export._common import (
    GENERATOR,
    PROVENANCE_COLORS,
    check_meshes,
    element_index,
    material_pbr,
    node_extras,
    unit_header,
    uv1_statement,
)
from roader.model.mesh import Mesh, corner_values
from roader.model.schema import RoadModel

AXIS_CONVERSION = "gltf(X, Y, Z) = unit(x, z, -y): X east, Y up, Z south"

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT = 5126
_UINT32 = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_TRIANGLES = 4
TEXCOORDS = "TEXCOORD_0 = uv0 (station m, offset m); TEXCOORD_1 = uv1 atlas in [0, 1]; v as in the model, not flipped"
# MSFT_screencoverage per level, finest first (one more entry than coarser levels).
SCREEN_COVERAGE = (0.25, 0.1, 0.04, 0.01, 0.0)


def to_gltf_axes(vertices: np.ndarray) -> np.ndarray:
    """Unit frame (x east, y north, z up) to glTF Y-up, as float32."""
    v = np.asarray(vertices, dtype=np.float64)
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype("<f4")


def unweld(mesh: Mesh) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """glTF vertex arrays of a mesh: positions (unit frame), attributes by glTF name, indices.

    Without per-corner attributes the mesh's own vertices and faces are used as they are.
    """
    present = {name: value for name, value in (("NORMAL", mesh.normals), ("TEXCOORD_0", mesh.uv0), ("TEXCOORD_1", mesh.uv1)) if value is not None}
    if not any(np.asarray(v).ndim == 3 for v in present.values()):
        return mesh.vertices, {k: np.asarray(v, dtype=np.float64) for k, v in present.items()}, mesh.faces
    width = {"NORMAL": 3, "TEXCOORD_0": 2, "TEXCOORD_1": 2}
    corners = [mesh.vertices[mesh.faces].reshape(-1, 3)]
    for name, value in present.items():
        corners.append(corner_values(mesh, value, width[name]).reshape(-1, width[name]))
    table = np.hstack(corners).astype("<f4")
    _keys, first, inverse = np.unique(table, axis=0, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    order = np.sort(first)
    index = rank[inverse.reshape(-1)].reshape(-1, 3)
    positions = mesh.vertices[mesh.faces].reshape(-1, 3)[order]
    attrs = {}
    col = 3
    for name in present:
        attrs[name] = table[order, col : col + width[name]].astype(np.float64)
        col += width[name]
    return positions, attrs, index


def _material_entry(name: str, rgba: tuple[float, ...], metallic: float, roughness: float) -> dict[str, Any]:
    return {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [float(c) for c in rgba],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
        "doubleSided": False,
    }


def glb_bytes(
    model: RoadModel,
    meshes: list[Mesh],
    *,
    color_by: str = "material",
    lods: Mapping[int, Sequence[Mesh]] | None = None,
) -> bytes:
    """Encode meshes as a .glb. color_by: "material" (FR-AS-6) or "source" (FR-RH-1).

    ``lods``: level -> meshes (FR-AS-1); level 0 is ``meshes`` itself, coarser levels
    are attached to the level-0 node of the same mesh id through MSFT_lod.
    """
    if color_by not in {"material", "source"}:
        raise ValueError(f"unknown color_by {color_by}")
    meshes = check_meshes(meshes)
    coarser = {level: {m.id: m for m in check_meshes(lods[level])} for level in sorted(lods or {}) if level > 0}
    index = element_index(model)

    if color_by == "material":
        slots = sorted({m.material for m in meshes})
        materials = [_material_entry(slot, *material_pbr(slot)) for slot in slots]
        material_of = {slot: i for i, slot in enumerate(slots)}
    else:
        sources = list(PROVENANCE_COLORS)
        materials = [
            _material_entry(f"provenance_{src.value}", PROVENANCE_COLORS[src], 0.0, 1.0) for src in sources
        ]
        material_of = {src.value: i for i, src in enumerate(sources)}

    blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    gltf_meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    def add_view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(buffer_views) - 1

    def add_float(data: np.ndarray, kind: str) -> int:
        view = add_view(data.tobytes(), _ARRAY_BUFFER)
        accessors.append({"bufferView": view, "componentType": _FLOAT, "count": len(data), "type": kind})
        return len(accessors) - 1

    def add_mesh(mesh: Mesh) -> int:
        verts, attrs, faces = unweld(mesh)
        positions = to_gltf_axes(verts)
        indices = faces.astype("<u4").reshape(-1)
        pos_view = add_view(positions.tobytes(), _ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": _FLOAT,
                "count": len(positions),
                "type": "VEC3",
                "min": [float(x) for x in positions.min(axis=0)],
                "max": [float(x) for x in positions.max(axis=0)],
            }
        )
        attributes = {"POSITION": len(accessors) - 1}
        if "NORMAL" in attrs:
            attributes["NORMAL"] = add_float(to_gltf_axes(attrs["NORMAL"]), "VEC3")
        for name in ("TEXCOORD_0", "TEXCOORD_1"):
            if name in attrs:
                attributes[name] = add_float(attrs[name].astype("<f4"), "VEC2")
        idx_view = add_view(indices.tobytes(), _ELEMENT_ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": _UINT32,
                "count": len(indices),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            }
        )
        idx_accessor = len(accessors) - 1
        key = mesh.material if color_by == "material" else mesh.source.value
        gltf_meshes.append(
            {
                "name": mesh.id,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": idx_accessor,
                        "material": material_of[key],
                        "mode": _TRIANGLES,
                    }
                ],
            }
        )
        return len(gltf_meshes) - 1

    roots = []
    for mesh in meshes:
        roots.append(len(nodes))
        nodes.append({"name": mesh.id, "mesh": add_mesh(mesh), "extras": node_extras(mesh, index)})
    for root in roots:
        node = nodes[root]
        chain = [(level, by_id[node["name"]]) for level, by_id in coarser.items() if node["name"] in by_id]
        if not chain:
            continue
        ids = []
        for level, mesh in chain:
            ids.append(len(nodes))
            nodes.append({"name": f"{mesh.id}#lod{level}", "mesh": add_mesh(mesh), "extras": {**node_extras(mesh, index), "lod": level}})
        node["extensions"] = {"MSFT_lod": {"ids": ids}}
        node["extras"] = {**node["extras"], "lod": 0, "MSFT_screencoverage": list(SCREEN_COVERAGE[: len(ids) + 1])}
    while len(blob) % 4:
        blob.append(0)

    header = unit_header(model)
    doc: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": GENERATOR,
            "extras": {
                **header,
                "axis_conversion": AXIS_CONVERSION,
                "units": "meters",
                "color_by": color_by,
                **_uv_extras(meshes),
            },
        },
        "scene": 0,
        "scenes": [{"name": model.unit.unit_id, "nodes": roots}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if coarser:
        doc["extensionsUsed"] = ["MSFT_lod"]
    json_chunk = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = bytearray(struct.pack("<III", _GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(json_chunk), _CHUNK_JSON) + json_chunk
    out += struct.pack("<II", len(bin_chunk), _CHUNK_BIN) + bin_chunk
    return bytes(out)


def _uv_extras(meshes: list[Mesh]) -> dict[str, Any]:
    """Texture-coordinate convention and the FR-AS-5 atlas statement, when UVs are present."""
    out: dict[str, Any] = {}
    if any(m.uv0 is not None or m.uv1 is not None for m in meshes):
        out["texcoords"] = TEXCOORDS
    if any(m.uv1 is not None for m in meshes):
        out["uv1_atlas"] = uv1_statement()
    return out


def write_glb(
    model: RoadModel,
    meshes: list[Mesh],
    path: Path,
    *,
    color_by: str = "material",
    lods: Mapping[int, Sequence[Mesh]] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(model, meshes, color_by=color_by, lods=lods))
    return path


_POINTS = 0
_UBYTE = 5121

# ASPRS class colors for the review overlay (FR-EX-4); unknown classes are grey.
CLASS_COLORS = {
    1: (170, 170, 170),  # unclassified, includes vehicles
    2: (150, 115, 70),  # ground
    3: (110, 160, 80),
    4: (80, 140, 60),
    5: (50, 110, 45),  # vegetation
    6: (200, 90, 70),  # building
    9: (70, 120, 200),  # water
    11: (60, 60, 66),  # road surface
    17: (235, 200, 60),  # bridge deck
}


def points_glb_bytes(model: RoadModel, xyz: np.ndarray, classification: np.ndarray, *, max_points: int = 1_500_000) -> bytes:
    """Lidar returns as one POINTS primitive colored by class, for review renders.

    Decimation takes every k-th point in input order, so output is deterministic.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    cls = np.asarray(classification).astype(np.int64)
    step = max(1, int(np.ceil(len(xyz) / max_points)))
    xyz, cls = xyz[::step], cls[::step]
    colors = np.full((len(cls), 4), 255, dtype=np.uint8)
    colors[:, :3] = 140
    for code, rgb in CLASS_COLORS.items():
        colors[cls == code, :3] = rgb
    positions = to_gltf_axes(xyz)
    pos = positions.tobytes()
    col = colors.tobytes()
    pad = -len(pos) % 4
    blob = pos + b"\0" * pad + col
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": GENERATOR, "extras": {**unit_header(model), "axis_conversion": AXIS_CONVERSION, "decimation_step": step}},
        "scene": 0,
        "scenes": [{"name": f"{model.unit.unit_id}_points", "nodes": [0]}],
        "nodes": [{"name": "lidar_points", "mesh": 0}],
        "meshes": [{"name": "lidar_points", "primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": _POINTS}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": len(positions),
                "type": "VEC3",
                "min": [float(x) for x in positions.min(axis=0)],
                "max": [float(x) for x in positions.max(axis=0)],
            },
            {"bufferView": 1, "componentType": _UBYTE, "normalized": True, "count": len(colors), "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": _ARRAY_BUFFER},
            {"buffer": 0, "byteOffset": len(pos) + pad, "byteLength": len(col), "target": _ARRAY_BUFFER},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    out = bytearray(struct.pack("<III", _GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(json_chunk), _CHUNK_JSON) + json_chunk
    out += struct.pack("<II", len(blob), _CHUNK_BIN) + blob
    return bytes(out)


def write_points_glb(model: RoadModel, xyz: np.ndarray, classification: np.ndarray, path: Path, *, max_points: int = 1_500_000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(points_glb_bytes(model, xyz, classification, max_points=max_points))
    return path


def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Split a .glb into its JSON document and BIN chunk, checking the container lengths."""
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC or version != 2 or length != len(data):
        raise ValueError("not a glTF 2.0 binary")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != _CHUNK_JSON:
        raise ValueError("first chunk is not JSON")
    doc = json.loads(data[20 : 20 + json_len])
    offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, offset)
    if bin_type != _CHUNK_BIN or offset + 8 + bin_len != len(data):
        raise ValueError("bad BIN chunk")
    return doc, data[offset + 8 : offset + 8 + bin_len]
