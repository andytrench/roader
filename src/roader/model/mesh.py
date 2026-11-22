"""Mesh contract between solids, assets, validate and export.

A Mesh is one solid body generated from one model element (FR-DM-3). It is never
the source of truth; regenerate it from the model.

Shading and texture attributes are optional and may be given per vertex or per
face corner. Generated solids use per-corner arrays: a closed solid shares each
vertex between faces on both sides of a crease or a UV seam, so the attribute
cannot live on the vertex without splitting it (which would open the shell).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roader.model.provenance import Source

LAYERS = {"L3", "L4", "L5", "L6", "L7"}


@dataclass
class Mesh:
    id: str  # stable: f"{element_id}:{part}" or with an index suffix
    element_id: str  # model element that produced it (FR-DM-3)
    layer: str  # one of LAYERS
    part: str  # pavement | deck | parapet | barrier | abutment | pier_column | pier_cap | embankment | terrain | proxy
    material: str  # semantic material slot (FR-AS-6)
    source: Source  # provenance of the producing element (AC-H6, FR-RH-1)
    vertices: np.ndarray  # (N, 3) float64, unit frame
    faces: np.ndarray  # (M, 3) int64, counter-clockwise seen from outside
    closed: bool = True  # False only for the open L7 terrain surface
    # Optional attributes; see corner_values for the two accepted layouts.
    # FR-AS-3 shading normals, unit length: (N, 3) per vertex or (M, 3, 3) per face corner.
    normals: np.ndarray | None = None
    # FR-AS-4 tiling UV set 0 in metres, (station along, offset across): (N, 2) or (M, 3, 2).
    uv0: np.ndarray | None = None
    # FR-AS-5 unique UV set 1, a non-overlapping atlas in [0, 1]: (N, 2) or (M, 3, 2).
    uv1: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer}")
        self.source = Source(self.source)
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        for name in ("normals", "uv0", "uv1"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=np.float64))


def corner_values(mesh: Mesh, value: np.ndarray, width: int) -> np.ndarray:
    """An attribute as (M, 3, width) per face corner, whichever way it is stored.

    Raises ValueError when the shape fits neither layout (shapes are not checked on
    construction, so a mesh edited with ``dataclasses.replace`` stays constructible).
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (len(mesh.faces), 3, width):
        return arr
    if arr.shape == (len(mesh.vertices), width):
        return arr[mesh.faces]
    raise ValueError(f"{mesh.id}: attribute shape {arr.shape} is neither per vertex ({len(mesh.vertices)}, {width}) nor per corner ({len(mesh.faces)}, 3, {width})")


def signed_volume(mesh: Mesh) -> float:
    v = mesh.vertices
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    origin = v.mean(axis=0) if len(v) else np.zeros(3)
    return float(np.einsum("ij,ij->i", a - origin, np.cross(b - origin, c - origin)).sum() / 6.0)
