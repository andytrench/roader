"""Stage 10: solid generation from the model (section 7). Lane `solids`.

Public API: generate(model: RoadModel, chord_tol_m: float = 0.01) -> list[Mesh].
Solids are constructed, never recovered or repaired.
"""

from __future__ import annotations

from roader.model.mesh import Mesh
from roader.model.schema import RoadModel
from roader.solids.assemble import assemble
from roader.solids.bodies import road_bodies
from roader.solids.grid import PlanVertices
from roader.solids.terrain import TerrainSampler, terrain_mesh

TERRAIN_ELEMENT_ID = "terrain"

__all__ = ["TERRAIN_ELEMENT_ID", "generate"]


def generate(model: RoadModel, chord_tol_m: float = 0.01) -> list[Mesh]:
    """Analytic export profile at the finest LOD (FR-SG-1..6, FR-SV-7).

    One closed Mesh per body, ids f"{element_id}:{part}" with an index suffix when an
    element yields several, then the open L7 terrain surface last.
    """
    if not chord_tol_m > 0.0:
        raise ValueError("chord_tol_m must be positive")
    sampler = TerrainSampler(model.terrain) if model.terrain is not None else None
    pv = PlanVertices()
    bodies = []
    for road in model.roads:
        bodies.extend(road_bodies(model, road, sampler, chord_tol_m, pv))
    result = assemble(bodies, pv, sampler)
    meshes = result.meshes
    if sampler is not None and model.terrain is not None:
        meshes.append(terrain_mesh(sampler, result.footprints, result.ring, TERRAIN_ELEMENT_ID, model.terrain.prov.source))
    return meshes
