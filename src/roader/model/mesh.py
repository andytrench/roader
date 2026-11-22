"""Mesh contract between solids, assets, validate and export.

A Mesh is one solid body generated from one model element (FR-DM-3). It is never
the source of truth; regenerate it from the model.
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
    part: str  # pavement | deck | parapet | barrier | abutment | pier_column | pier_cap | embankment | terrain
    material: str  # semantic material slot (FR-AS-6)
    source: Source  # provenance of the producing element (AC-H6, FR-RH-1)
    vertices: np.ndarray  # (N, 3) float64, unit frame
    faces: np.ndarray  # (M, 3) int64, counter-clockwise seen from outside
    closed: bool = True  # False only for the open L7 terrain surface

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer}")
        self.source = Source(self.source)
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)


def signed_volume(mesh: Mesh) -> float:
    v = mesh.vertices
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    origin = v.mean(axis=0) if len(v) else np.zeros(3)
    return float(np.einsum("ij,ij->i", a - origin, np.cross(b - origin, c - origin)).sum() / 6.0)
