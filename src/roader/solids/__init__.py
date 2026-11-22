"""Stage 10: solid generation from the model (section 7). Lane `solids`.

Public API:

- generate(model, chord_tol_m=0.01, profile="analytic") -> list[Mesh]
- generate_lods(model, profile="analytic") -> dict[int, list[Mesh]]

Solids are constructed, never recovered or repaired. Every closed mesh carries
model-derived shading normals and UV sets 0 and 1 (FR-AS-3..5); the terrain carries
normals and UV set 1.
"""

from __future__ import annotations

from roader.model.mesh import Mesh
from roader.model.schema import RoadModel
from roader.solids.assemble import assemble
from roader.solids.attributes import RoadFrame, finish, terrain_attributes
from roader.solids.bodies import road_bodies
from roader.solids.grid import PlanVertices
from roader.solids.junctions import JunctionPlan
from roader.solids.terrain import TerrainSampler, terrain_meshes

TERRAIN_ELEMENT_ID = "terrain"
PROFILES = ("analytic", "realtime")
LOD_CHORD_TOL = {0: 0.01, 1: 0.05, 2: 0.25, 3: 1.0}  # m, FR-AS-1

__all__ = ["LOD_CHORD_TOL", "PROFILES", "TERRAIN_ELEMENT_ID", "generate", "generate_lods"]


def generate(model: RoadModel, chord_tol_m: float = 0.01, profile: str = "analytic") -> list[Mesh]:
    """Solids of one model in an export profile (FR-SG-1..6, FR-SV-7, section 7 table).

    "analytic": touching solids share conforming faces and never overlap.
    "realtime": bodies embed REALTIME_EMBED_M into what carries them (pavement into
    its embankment, barriers into pavement or deck, supports up into the deck, columns
    into the cap, embankments into the ground) and outlines are inset by a few tenths
    of a millimetre per body class, so no two solids have coplanar overlapping faces.

    One closed Mesh per body, ids f"{element_id}:{part}" with an index suffix when an
    element yields several, then the open L7 terrain surface last, one Mesh per
    connected piece. Ids and order are the same in both profiles.
    """
    if not chord_tol_m > 0.0:
        raise ValueError("chord_tol_m must be positive")
    if profile not in PROFILES:
        raise ValueError(f"unknown export profile {profile!r}; expected one of {PROFILES}")
    realtime = profile == "realtime"
    sampler = TerrainSampler(model.terrain) if model.terrain is not None else None
    pv = PlanVertices()
    plan = JunctionPlan(model, pv)  # L6 areas replace the road sweeps they cover (FR-SG-3)
    bodies = []
    for road in model.roads:
        bodies.extend(plan.clip(road_bodies(model, plan.road(road), sampler, chord_tol_m, pv, realtime=realtime)))
    bodies.extend(plan.bodies(sampler, chord_tol_m))
    result = assemble(bodies, pv, sampler, conforming=not realtime)
    frames = {road.id: RoadFrame(road) for road in model.roads}
    meshes = [finish(raw, pv, frames, realtime=realtime) for raw in result.raw]
    if sampler is not None and model.terrain is not None:
        terrain = terrain_meshes(
            sampler, result.footprints, result.ring, model.terrain.id, model.terrain.prov.source, chord_tol_m
        )
        meshes += [terrain_attributes(m) for m in terrain]
    return meshes


def generate_lods(model: RoadModel, profile: str = "analytic") -> dict[int, list[Mesh]]:
    """FR-AS-1 level-of-detail chain: level -> meshes at chord tolerance LOD_CHORD_TOL[level].

    Every level is a complete rendering of the model, so each solid is watertight on
    its own and voids under decks stay open at every level; no proxy is generated.
    """
    return {level: generate(model, chord_tol_m=tol, profile=profile) for level, tol in LOD_CHORD_TOL.items()}
