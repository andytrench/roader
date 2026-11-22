"""Helpers shared by the export writers: mesh checks, element index, palettes, formatting.

Every writer iterates in sorted id order and formats floats with ``num`` so two
runs on the same inputs produce the same bytes (AC-H5).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from roader.model.mesh import Mesh, corner_values
from roader.model.provenance import Provenance, Source
from roader.model.schema import RoadModel
from roader.model.serialize import model_hash, to_plain
from roader.solids.attributes import UV1_ATLAS_PX, UV1_PAD_PX

GENERATOR = "roader-export"

# FR-RH-1 debug palette, in legend order. RGBA, linear.
PROVENANCE_COLORS: dict[Source, tuple[float, float, float, float]] = {
    Source.MEASURED: (0.1, 0.7, 0.2, 1.0),
    Source.INVENTORY: (0.15, 0.35, 0.9, 1.0),
    Source.VECTOR: (0.95, 0.8, 0.1, 1.0),
    Source.RULE: (0.9, 0.15, 0.15, 1.0),
}

# FR-AS-6 semantic material slots, flat base colors for M1: (RGBA, metallic, roughness).
MATERIALS: dict[str, tuple[tuple[float, float, float, float], float, float]] = {
    "asphalt": ((0.08, 0.08, 0.09, 1.0), 0.0, 0.9),
    "travel_lane_asphalt": ((0.08, 0.08, 0.09, 1.0), 0.0, 0.9),
    "deck_concrete": ((0.6, 0.6, 0.58, 1.0), 0.0, 0.8),
    "concrete_pavement": ((0.55, 0.55, 0.53, 1.0), 0.0, 0.85),
    "shoulder": ((0.25, 0.25, 0.25, 1.0), 0.0, 0.9),
    "deck": ((0.6, 0.6, 0.58, 1.0), 0.0, 0.8),
    "structural_concrete": ((0.65, 0.64, 0.6, 1.0), 0.0, 0.8),
    "steel": ((0.45, 0.47, 0.5, 1.0), 1.0, 0.4),
    "soil_slope": ((0.35, 0.28, 0.18, 1.0), 0.0, 1.0),
    "slope_paving": ((0.5, 0.48, 0.44, 1.0), 0.0, 0.9),
    "terrain": ((0.3, 0.36, 0.2, 1.0), 0.0, 1.0),
}


def material_pbr(slot: str) -> tuple[tuple[float, float, float, float], float, float]:
    """PBR parameters for a slot; unknown slots get a stable color from the slot name."""
    if slot in MATERIALS:
        return MATERIALS[slot]
    digest = hashlib.sha1(slot.encode("utf-8")).digest()
    rgb = tuple(round(0.2 + 0.6 * b / 255.0, 3) for b in digest[:3])
    return (rgb[0], rgb[1], rgb[2], 1.0), 0.0, 0.9


def num(value: float, digits: int = 10) -> str:
    """Deterministic decimal text: repr of the rounded float, no negative zero."""
    value = round(float(value), digits)
    if value == 0.0 or not math.isfinite(value):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value {value}")
        return "0.0"
    return repr(value)


def rounded(value: float, digits: int = 6) -> float:
    value = round(float(value), digits)
    return 0.0 if value == 0.0 else value


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, compact, floats rounded as in the native model (D-011)."""
    text = json.dumps(to_plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def safe_name(text: str) -> str:
    """Identifier-safe name ([A-Za-z_][A-Za-z0-9_]*) for USD prims and file names."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not name or name[0].isdigit():
        name = "_" + name
    return name


def frame_record(model: RoadModel) -> dict[str, Any]:
    f = model.frame
    return {
        "proj": f.proj,
        "anchor_lon": rounded(f.anchor_lon, 9),
        "anchor_lat": rounded(f.anchor_lat, 9),
        "vertical_datum": f.vertical_datum,
        "geoid_model": f.geoid_model,
    }


def unit_header(model: RoadModel) -> dict[str, Any]:
    """Anchor and identity block written into every format (FR-EX-3, FR-AS-10)."""
    return {
        "unit_id": model.unit.unit_id,
        "model_hash": model_hash(model),
        "frame": frame_record(model),
        "tile_origin_unit_frame": [0.0, 0.0, 0.0],
    }


def check_meshes(meshes: Iterable[Mesh]) -> list[Mesh]:
    """Meshes sorted by id; rejects empty, duplicate or malformed meshes (FR-AS-10).

    Exports never repair geometry (spec section 0): a bad mesh is a generator bug.
    """
    out = sorted(meshes, key=lambda m: m.id)
    seen: set[str] = set()
    for mesh in out:
        if mesh.id in seen:
            raise ValueError(f"duplicate mesh id {mesh.id}")
        seen.add(mesh.id)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"empty mesh {mesh.id}")
        if not np.all(np.isfinite(mesh.vertices)):
            raise ValueError(f"non-finite vertex in mesh {mesh.id}")
        if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
            raise ValueError(f"face index out of range in mesh {mesh.id}")
        if not mesh.element_id:
            raise ValueError(f"mesh {mesh.id} has no element_id (FR-DM-3)")
    return out


# ---------------------------------------------------------------- element index


@dataclass(frozen=True)
class ElementInfo:
    """What a mesh element_id refers to in the model."""

    id: str
    kind: str  # road | structure | support | barrier | side_slope | junction | terrain
    prov: Provenance | None
    road_id: str | None = None  # owning road (road, barrier, side_slope)
    structure_id: str | None = None  # owning structure (structure, support, parapet)
    obj: Any = None


def element_index(model: RoadModel) -> dict[str, ElementInfo]:
    """Model element ids to ElementInfo. Barriers that are parapets of a structure
    belong to that structure; others belong to their road."""
    index: dict[str, ElementInfo] = {}
    for road in model.roads:
        index[road.id] = ElementInfo(road.id, "road", None, road_id=road.id, obj=road)
        for barrier in road.cross_section.barriers:
            structure_id = _parapet_owner(model, road.id, barrier.s0, barrier.s1, barrier.kind)
            index[barrier.id] = ElementInfo(
                barrier.id, "barrier", barrier.prov, road_id=road.id, structure_id=structure_id, obj=barrier
            )
    for slope in model.side_slopes:
        index[slope.id] = ElementInfo(slope.id, "side_slope", slope.prov, road_id=slope.road_id, obj=slope)
    for structure in model.structures:
        index[structure.id] = ElementInfo(
            structure.id, "structure", structure.prov, road_id=structure.carried_road,
            structure_id=structure.id, obj=structure,
        )
        for support in structure.supports:
            index[support.id] = ElementInfo(
                support.id, "support", support.prov, structure_id=structure.id, obj=support
            )
    for junction in model.junctions:
        index[junction.id] = ElementInfo(junction.id, "junction", junction.prov, obj=junction)
    return index


def _parapet_owner(model: RoadModel, road_id: str, s0: float, s1: float, kind: str) -> str | None:
    """Structure carrying road_id whose span holds most of [s0, s1]."""
    for structure in sorted(model.structures, key=lambda st: st.id):
        if structure.carried_road != road_id:
            continue
        overlap = min(s1, structure.s_end) - max(s0, structure.s_start)
        if overlap > 0.0 and (kind == "parapet" or overlap >= 0.5 * (s1 - s0)):
            return structure.id
    return None


def road_confidence(road: Any) -> float:
    """Lowest confidence among a road's L1/L2 elements (used where one number is needed)."""
    values = [e.prov.confidence for e in road.alignment.elements]
    values += [e.prov.confidence for e in road.profile.elements]
    return min(values) if values else 0.0


def element_confidence(info: ElementInfo | None) -> float | None:
    if info is None:
        return None
    if info.prov is not None:
        return float(info.prov.confidence)
    if info.kind == "road":
        return float(road_confidence(info.obj))
    return None


def mesh_info(mesh: Mesh, index: dict[str, ElementInfo]) -> ElementInfo | None:
    """Element of a mesh; L7 meshes map to the terrain patch whatever their id."""
    if mesh.layer == "L7":
        return ElementInfo(mesh.element_id, "terrain", None)
    return index.get(mesh.element_id)


def node_extras(mesh: Mesh, index: dict[str, ElementInfo]) -> dict[str, Any]:
    """Per-primitive metadata shared by glTF and USD (FR-DM-3, AC-H6, FR-RH-2)."""
    confidence = element_confidence(mesh_info(mesh, index))
    extras: dict[str, Any] = {
        "element_id": mesh.element_id,
        "layer": mesh.layer,
        "part": mesh.part,
        "material": mesh.material,
        "source": mesh.source.value,
        "mesh_id": mesh.id,
    }
    if confidence is not None:
        extras["confidence"] = rounded(confidence, 6)
    density = uv1_texel_density(mesh)
    if density is not None:
        extras["uv1_texel_px_per_m"] = rounded(density, 3)
    return extras


def uv1_statement() -> dict[str, int]:
    """FR-AS-5: the atlas every uv1 set is packed for."""
    return {"resolution_px": UV1_ATLAS_PX, "padding_px": UV1_PAD_PX}


def uv1_texel_density(mesh: Mesh) -> float | None:
    """Lightmap texels per metre of the mesh's uv1 on the stated atlas (FR-AS-5)."""
    if mesh.uv1 is None or not len(mesh.faces):
        return None
    uv = corner_values(mesh, mesh.uv1, 2)
    t = mesh.vertices[mesh.faces]
    area3 = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    area_uv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    if area3 <= 0.0:
        return None
    return float(UV1_ATLAS_PX * math.sqrt(area_uv / area3))
