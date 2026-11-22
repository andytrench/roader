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
"""

from __future__ import annotations

import json
import struct
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
)
from roader.model.mesh import Mesh
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


def to_gltf_axes(vertices: np.ndarray) -> np.ndarray:
    """Unit frame (x east, y north, z up) to glTF Y-up, as float32."""
    v = np.asarray(vertices, dtype=np.float64)
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype("<f4")


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


def glb_bytes(model: RoadModel, meshes: list[Mesh], *, color_by: str = "material") -> bytes:
    """Encode meshes as a .glb. color_by: "material" (FR-AS-6) or "source" (FR-RH-1)."""
    if color_by not in {"material", "source"}:
        raise ValueError(f"unknown color_by {color_by}")
    meshes = check_meshes(meshes)
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

    for mesh in meshes:
        positions = to_gltf_axes(mesh.vertices)
        indices = mesh.faces.astype("<u4").reshape(-1)
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
        pos_accessor = len(accessors) - 1
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
                        "attributes": {"POSITION": pos_accessor},
                        "indices": idx_accessor,
                        "material": material_of[key],
                        "mode": _TRIANGLES,
                    }
                ],
            }
        )
        nodes.append({"name": mesh.id, "mesh": len(gltf_meshes) - 1, "extras": node_extras(mesh, index)})
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
            },
        },
        "scene": 0,
        "scenes": [{"name": model.unit.unit_id, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = bytearray(struct.pack("<III", _GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(json_chunk), _CHUNK_JSON) + json_chunk
    out += struct.pack("<II", len(bin_chunk), _CHUNK_BIN) + bin_chunk
    return bytes(out)


def write_glb(model: RoadModel, meshes: list[Mesh], path: Path, *, color_by: str = "material") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(model, meshes, color_by=color_by))
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
