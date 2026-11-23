"""Stage 10: solid generation from the model (section 7). Lane `solids`.

Public API:

- generate(model, chord_tol_m=0.01, profile="analytic", failures=None) -> list[Mesh]
- generate_lods(model, profile="analytic") -> dict[int, list[Mesh]]

Solids are constructed, never recovered or repaired. Every closed mesh carries
model-derived shading normals and UV sets 0 and 1 (FR-AS-3..5); the terrain carries
normals and UV set 1.
"""

from __future__ import annotations

import dataclasses

from roader.model.mesh import Mesh
from roader.model.schema import ELEMENT_FAILED, Flag, RoadModel
from roader.solids.assemble import assemble
from roader.solids.attributes import RoadFrame, finish, terrain_attributes
from roader.solids.bodies import road_bodies
from roader.solids.failure import ElementFailure, describe
from roader.solids.grid import PlanVertices
from roader.solids.junctions import JunctionPlan
from roader.solids.terrain import TerrainSampler, terrain_meshes

TERRAIN_ELEMENT_ID = "terrain"
PROFILES = ("analytic", "realtime")
LOD_CHORD_TOL = {0: 0.01, 1: 0.05, 2: 0.25, 3: 1.0}  # m, FR-AS-1
MAX_RETRIES = 50

__all__ = ["LOD_CHORD_TOL", "PROFILES", "TERRAIN_ELEMENT_ID", "generate", "generate_lods"]


def generate(model: RoadModel, chord_tol_m: float = 0.01, profile: str = "analytic", *, failures: list[Flag] | None = None) -> list[Mesh]:
    """Solids of one model in an export profile (FR-SG-1..6, FR-SV-7, section 7 table).

    "analytic": touching solids share conforming faces and never overlap.
    "realtime": bodies embed REALTIME_EMBED_M into what carries them (pavement into
    its embankment, barriers into pavement or deck, supports up into the deck, columns
    into the cap, embankments into the ground) and outlines are inset by a few tenths
    of a millimetre per body class, so no two solids have coplanar overlapping faces.

    One closed Mesh per body, ids f"{element_id}:{part}" with an index suffix when an
    element yields several, then the open L7 terrain surface last, one Mesh per
    connected piece. Ids and order are the same in both profiles.

    ``failures``: None raises on the first element that cannot be built. A list makes
    generation tolerant (FR-EX-1, D-119): a road whose sweep fails is retried with its
    structures squared (support skew 0), then dropped; a junction area that fails is
    left out so its roads are swept through it; a terrain surface that fails is left out;
    any other element whose bodies fail is dropped. Each replacement appends an ``ELEMENT_FAILED`` flag naming the element,
    the requirement and the exception, so the unit is held.
    """
    if not chord_tol_m > 0.0:
        raise ValueError("chord_tol_m must be positive")
    if profile not in PROFILES:
        raise ValueError(f"unknown export profile {profile!r}; expected one of {PROFILES}")
    if failures is None:
        return _generate(model, chord_tol_m, profile, None, set(), set())
    work, dropped, squared = model, set(), set()
    for _ in range(MAX_RETRIES):
        try:
            return _generate(work, chord_tol_m, profile, failures, dropped, squared)
        except ElementFailure as exc:
            eid = exc.element_id
            if any(jn.id == eid for jn in work.junctions):
                work = dataclasses.replace(work, junctions=[jn for jn in work.junctions if jn.id != eid])
                failures.append(Flag(ELEMENT_FAILED, eid, "FR-SG-3", f"junction area left out, roads swept through it; {describe(exc)}"))
            elif eid not in dropped:
                dropped.add(eid)
                failures.append(Flag(ELEMENT_FAILED, eid, "FR-SG-1", f"element dropped from the solids; {describe(exc)}"))
            else:
                raise
    raise RuntimeError(f"solids: more than {MAX_RETRIES} element failures")


def _squared(model: RoadModel, road_id: str) -> RoadModel:
    """The model with the structures carried by ``road_id`` square to it (support skew 0)."""
    structures = [
        dataclasses.replace(st, supports=[dataclasses.replace(sp, skew_rad=0.0) for sp in st.supports]) if st.carried_road == road_id else st
        for st in model.structures
    ]
    return dataclasses.replace(model, structures=structures)


def _generate(model: RoadModel, chord_tol_m: float, profile: str, failures: list[Flag] | None, dropped: set[str], squared: set[str]) -> list[Mesh]:
    """One rendering; elements in ``dropped`` are left out, roads in ``squared`` swept with square structures."""
    realtime = profile == "realtime"
    sampler = TerrainSampler(model.terrain) if model.terrain is not None else None
    pv = PlanVertices()
    plan = JunctionPlan(model, pv)  # L6 areas replace the road sweeps they cover (FR-SG-3)
    bodies = []
    notes: list[Flag] = []  # geometry limited to be swept (D-141), kept only when the road's bodies are used
    for road in model.roads:
        if road.id in dropped:
            continue
        try:
            source = _squared(model, road.id) if road.id in squared else model
            bodies.extend(plan.clip(road_bodies(source, plan.road(road), sampler, chord_tol_m, pv, realtime=realtime, limits=notes)))
        except Exception as exc:
            if failures is None:
                raise
            if road.id in squared or not any(st.carried_road == road.id for st in model.structures):
                raise ElementFailure(road.id, exc) from exc
            try:
                bodies.extend(plan.clip(road_bodies(_squared(model, road.id), plan.road(road), sampler, chord_tol_m, pv, realtime=realtime, limits=notes)))
            except Exception as again:  # noqa: BLE001
                raise ElementFailure(road.id, again) from exc
            failures.append(Flag(ELEMENT_FAILED, road.id, "FR-ST-4", f"structures on the road squared (support skew 0); {describe(exc)}"))
            squared.add(road.id)  # retries keep the same form
    bodies.extend(plan.bodies(sampler, chord_tol_m, realtime=realtime))
    bodies = plan.trim_fills(bodies)  # gore side slopes stop at the pavement edge (D-071)
    bodies = [b for b in bodies if b.element_id not in dropped]
    result = assemble(bodies, pv, sampler, conforming=not realtime)
    frames = {road.id: RoadFrame(road) for road in model.roads}
    meshes = [finish(raw, pv, frames, realtime=realtime) for raw in result.raw]
    if sampler is not None and model.terrain is not None:
        try:
            terrain = terrain_meshes(
                sampler, result.footprints, result.ring, model.terrain.id, model.terrain.prov.source, chord_tol_m
            )
        except Exception as exc:
            if failures is None:
                raise
            failures.append(Flag(ELEMENT_FAILED, model.terrain.id, "FR-SG-6", f"terrain surface dropped; {describe(exc)}"))
            terrain = []
        meshes += [terrain_attributes(m) for m in terrain]
    if failures is not None:
        seen = {(f.code, f.element_id, f.message) for f in failures}
        for f in notes:
            if (f.code, f.element_id, f.message) not in seen:
                seen.add((f.code, f.element_id, f.message))
                failures.append(f)
    return meshes


def generate_lods(model: RoadModel, profile: str = "analytic") -> dict[int, list[Mesh]]:
    """FR-AS-1 level-of-detail chain: level -> meshes at chord tolerance LOD_CHORD_TOL[level].

    Every level is a complete rendering of the model, so each solid is watertight on
    its own and voids under decks stay open at every level; no proxy is generated.
    """
    return {level: generate(model, chord_tol_m=tol, profile=profile) for level, tol in LOD_CHORD_TOL.items()}
