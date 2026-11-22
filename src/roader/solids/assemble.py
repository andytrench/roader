"""Bodies to closed meshes: each body is the prism between two level functions over its cells.

Top and bottom faces come from the cell triangulation, so stacked bodies (pavement
on embankment, parapet on deck, deck on abutment) share identical faces. Side walls
stand on the plan edges of a body's outline; every wall edge is cut into strips at
the top and bottom levels of all bodies meeting on that edge, plus the terrain, so
bodies side by side share identical wall faces too (FR-SV-7). Each vertical edge
carries every level present at its plan vertex, so no wall has a T-junction.

In the real-time profile (``conforming=False``) bodies overlap by design, so each
is closed on its own levels only; the terrain still cuts ground-contact walls.
Faces are tagged TOP, BOTTOM or WALL (walls with their outline edge) so normals and
UVs can follow the model (``roader.solids.attributes``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from roader.model.mesh import Mesh
from roader.model.provenance import Source
from roader.solids.grid import Cell, PlanVertices, Tri, ladder
from roader.solids.terrain import TerrainSampler

LEVEL_MERGE_M = 1e-6
TOP, BOTTOM, WALL = 0, 1, 2

Key = tuple[int, float]  # plan vertex id, elevation


@dataclass
class Body:
    mesh_id: str
    element_id: str
    layer: str
    part: str
    material: str
    source: Source
    cells: list[Cell]
    bottom: Callable[[int], float]
    top: Callable[[int], float]
    ground: bool | None = None  # None: in contact if the bottom is below terrain everywhere
    bottom_tris: list[Tri] | None = None  # replaces the cell triangulation for the bottom face
    creases: frozenset[str] = frozenset()  # u-line keys where the top surface has a model crease (FR-AS-3)
    inset_m: float = 0.0  # real-time profile: plan inset of the outline (FR-SV-7)
    zb: dict[int, float] = field(default_factory=dict)
    zt: dict[int, float] = field(default_factory=dict)


def boundary_edges(cells: list[Cell]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for cell in cells:
        loop = cell.loop
        edges.extend((loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop)))
    present = set(edges)
    return [(a, b) for a, b in edges if (b, a) not in present]


def chain_ring(edges: list[tuple[int, int]]) -> list[int]:
    nxt: dict[int, int] = {}
    for a, b in edges:
        if a in nxt:
            raise ValueError("outline touches itself")
        nxt[a] = b
    start = edges[0][0]
    ring = [start]
    while nxt[ring[-1]] != start:
        ring.append(nxt[ring[-1]])
        if len(ring) > len(edges):
            raise ValueError("outline is not a single ring")
    if len(ring) != len(edges):
        raise ValueError("outline is not a single ring")
    return ring


class Levels:
    """Elevations used at each plan vertex, merged when closer than LEVEL_MERGE_M."""

    def __init__(self) -> None:
        self.raw: dict[int, list[float]] = {}
        self.rep: dict[int, dict[float, float]] = {}
        self.vals: dict[int, list[float]] = {}

    def add(self, p: int, z: float) -> None:
        self.raw.setdefault(p, []).append(z)

    def finalize(self) -> None:
        for p, zs in self.raw.items():
            mapping: dict[float, float] = {}
            reps: list[float] = []
            prev = None
            for z in sorted(set(zs)):
                if prev is None or z - prev > LEVEL_MERGE_M:
                    reps.append(z)
                mapping[z] = reps[-1]
                prev = z
            self.rep[p] = mapping
            self.vals[p] = reps

    def canon(self, p: int, z: float) -> float:
        return self.rep[p][z]


@dataclass
class RawMesh:
    """One body's triangles before attributes: vertex keys, faces and what each face is."""

    body: Body
    plan: np.ndarray  # (N,) plan vertex id per mesh vertex
    vertices: np.ndarray  # (N, 3) plan position and level of each mesh vertex
    faces: np.ndarray  # (M, 3) indices into vertices
    kind: np.ndarray  # (M,) TOP, BOTTOM or WALL
    wall_edge: np.ndarray  # (M, 2) plan edge (a, b) of a wall face along the outline, else (-1, -1)


@dataclass
class Assembly:
    raw: list[RawMesh]
    footprints: list[list[tuple[float, float]]]
    ring: list[tuple[float, float, float]]

    @property
    def meshes(self) -> list[Mesh]:
        return [plain_mesh(r, r.vertices) for r in self.raw]


Face = tuple[Key, Key, Key]


def assemble(bodies: list[Body], pv: PlanVertices, sampler: TerrainSampler | None, *, conforming: bool = True) -> Assembly:
    """Close every body into a shell.

    conforming (analytic profile): levels and wall cuts are shared, so touching bodies
    share identical faces (FR-SV-7). Otherwise (real-time profile) each body is closed
    on its own levels, and the terrain cuts only the walls of ground-contact bodies.
    """
    shared = Levels()
    own = [shared if conforming else Levels() for _ in bodies]
    terrain: dict[int, float] = {}
    for n, body in enumerate(bodies):
        levels = own[n]
        verts, bottom_only = _body_vertices(body)
        for p in verts + bottom_only:
            zb = body.bottom(p)
            zt = body.top(p) if p not in bottom_only else zb
            if not (math.isfinite(zb) and math.isfinite(zt)):
                raise ValueError(f"{body.mesh_id}: non-finite level at ({pv.x[p]:.2f}, {pv.y[p]:.2f})")
            if zt < zb - 1e-9:
                raise ValueError(f"{body.mesh_id}: top below bottom at ({pv.x[p]:.2f}, {pv.y[p]:.2f})")
            body.zb[p], body.zt[p] = zb, max(zt, zb)
            levels.add(p, zb)
            levels.add(p, body.zt[p])
            if sampler is not None and p not in terrain:
                terrain[p] = sampler(pv.x[p], pv.y[p])
        if body.ground is None:
            below = [body.zb[p] < terrain[p] for p in verts] if sampler is not None else [False]
            if any(below) and not all(below):
                raise ValueError(f"{body.mesh_id}: bottom crosses the terrain; set footing below ground")
            body.ground = all(below)
        if sampler is not None and (conforming or body.ground):
            for p in verts + bottom_only:
                levels.add(p, terrain[p])
    for levels in {id(lv): lv for lv in own}.values():
        levels.finalize()
    for n, body in enumerate(bodies):
        body.zb = {p: own[n].canon(p, z) for p, z in body.zb.items()}
        body.zt = {p: own[n].canon(p, z) for p, z in body.zt.items()}
    if conforming:
        terrain = {p: shared.canon(p, z) for p, z in terrain.items()}

    faces: list[list[Face]] = [[] for _ in bodies]
    kinds: list[list[int]] = [[] for _ in bodies]
    wall_edges: list[list[tuple[int, int]]] = [[] for _ in bodies]
    walls: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for n, body in enumerate(bodies):
        zb, zt = body.zb, body.zt
        out, kind = faces[n], kinds[n]
        for cell in body.cells:
            for a, b, c in cell.tris:
                if zt[a] == zb[a] and zt[b] == zb[b] and zt[c] == zb[c]:
                    raise ValueError(f"{body.mesh_id}: zero thickness over a whole triangle")
                out.append(((a, zt[a]), (b, zt[b]), (c, zt[c])))
                kind.append(TOP)
                if body.bottom_tris is None:
                    out.append(((a, zb[a]), (c, zb[c]), (b, zb[b])))
                    kind.append(BOTTOM)
        for a, b, c in body.bottom_tris or []:
            out.append(((a, zb[a]), (c, zb[c]), (b, zb[b])))
            kind.append(BOTTOM)
        wall_edges[n] = [(-1, -1)] * len(out)
        group = 0 if conforming else n
        for a, b in boundary_edges(body.cells):
            walls.setdefault((group, min(a, b), max(a, b)), []).append((n, a, b))

    for (group, p, q), entries in walls.items():
        levels = shared if conforming else own[group]
        cut_terrain = terrain if conforming or bodies[group].ground else {}
        _walls(p, q, entries, bodies, faces, kinds, wall_edges, levels, cut_terrain, pv)

    raw = [_raw(body, out, kind, edges, pv) for body, out, kind, edges in zip(bodies, faces, kinds, wall_edges, strict=True)]

    footprints: list[list[tuple[float, float]]] = []
    ground_edges: list[tuple[int, int]] = []
    for body in bodies:
        if body.ground:
            footprints.extend([(pv.x[p], pv.y[p]) for p in cell.loop] for cell in body.cells)
            ground_edges.extend(boundary_edges(body.cells))
    present = set(ground_edges)
    ring: list[tuple[float, float, float]] = []
    seen: set[int] = set()
    for a, b in ground_edges:
        if (b, a) in present:
            continue
        for p in (a, b):
            if p not in seen:
                seen.add(p)
                ring.append((pv.x[p], pv.y[p], terrain[p]))
    return Assembly(raw, footprints, ring)


def _body_vertices(body: Body) -> tuple[list[int], list[int]]:
    """Vertices of the cells, then vertices used only by an overriding bottom face."""
    seen: dict[int, None] = {}
    for cell in body.cells:
        for p in cell.loop:
            seen.setdefault(p, None)
        for tri in cell.tris:
            for p in tri:
                seen.setdefault(p, None)
    extra: dict[int, None] = {}
    for tri in body.bottom_tris or []:
        for p in tri:
            if p not in seen:
                extra.setdefault(p, None)
    return list(seen), list(extra)


def _walls(
    p: int,
    q: int,
    entries: list[tuple[int, int, int]],
    bodies: list[Body],
    faces: list[list[Face]],
    kinds: list[list[int]],
    wall_edges: list[list[tuple[int, int]]],
    levels: Levels,
    terrain: dict[int, float],
    pv: PlanVertices,
) -> None:
    ranges = []
    cuts: set[tuple[float, float]] = set()
    for n, _a, _b in entries:
        body = bodies[n]
        bot = (body.zb[p], body.zb[q])
        top = (body.zt[p], body.zt[q])
        ranges.append((bot, top))
        cuts.add(bot)
        cuts.add(top)
    if p in terrain and q in terrain:
        tp, tq = terrain[p], terrain[q]
        if any(bot[0] < tp < top[0] or bot[1] < tq < top[1] for bot, top in ranges):
            cuts.add((tp, tq))
    order = sorted(cuts)
    for lo, hi in pairwise(order):
        if hi[1] < lo[1]:
            raise ValueError(f"levels cross along the wall at ({pv.x[p]:.2f}, {pv.y[p]:.2f})")
    index = {cut: k for k, cut in enumerate(order)}

    strips: list[list[tuple[Key, Key, Key]]] = []
    for lo, hi in pairwise(order):
        left = [(p, z) for z in levels.vals[p] if lo[0] <= z <= hi[0]]
        right = [(q, z) for z in levels.vals[q] if lo[1] <= z <= hi[1]]

        def rel(key: Key, lo_z: float, hi_z: float) -> float:
            return (key[1] - lo_z) / (hi_z - lo_z) if hi_z > lo_z else 0.0

        def shorter(ln: Key, _r: Key, _l: Key, rn: Key, lo=lo, hi=hi) -> bool:
            return rel(ln, lo[0], hi[0]) <= rel(rn, lo[1], hi[1])

        strips.append(ladder(left, right, shorter))  # type: ignore[arg-type]

    for (n, a, b), (bot, top) in zip(entries, ranges, strict=True):
        dx, dy = pv.x[b] - pv.x[a], pv.y[b] - pv.y[a]
        for k in range(index[bot], index[top]):
            for tri in strips[k]:
                faces[n].append(tri if _wall_normal_dot(tri, pv, dy, -dx) > 0.0 else (tri[0], tri[2], tri[1]))
                kinds[n].append(WALL)
                wall_edges[n].append((a, b))


def _wall_normal_dot(tri: tuple[Key, Key, Key], pv: PlanVertices, ox: float, oy: float) -> float:
    (a, za), (b, zb), (c, zc) = tri
    ux, uy, uz = pv.x[b] - pv.x[a], pv.y[b] - pv.y[a], zb - za
    vx, vy, vz = pv.x[c] - pv.x[a], pv.y[c] - pv.y[a], zc - za
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    return nx * ox + ny * oy


def _raw(body: Body, tris: list[Face], kind: list[int], edges: list[tuple[int, int]], pv: PlanVertices) -> RawMesh:
    index: dict[Key, int] = {}
    faces = np.empty((len(tris), 3), dtype=np.int64)
    for n, tri in enumerate(tris):
        for k, key in enumerate(tri):
            idx = index.get(key)
            if idx is None:
                idx = index[key] = len(index)
            faces[n, k] = idx
    vertices = np.array([(pv.x[p], pv.y[p], z) for p, z in index], dtype=np.float64).reshape(-1, 3)
    plan = np.array([p for p, _z in index], dtype=np.int64)
    return RawMesh(body, plan, vertices, faces, np.array(kind, dtype=np.int64), np.array(edges, dtype=np.int64).reshape(-1, 2))


def plain_mesh(raw: RawMesh, vertices: np.ndarray, **attributes: np.ndarray | None) -> Mesh:
    body = raw.body
    return Mesh(
        id=body.mesh_id,
        element_id=body.element_id,
        layer=body.layer,
        part=body.part,
        material=body.material,
        source=body.source,
        vertices=vertices,
        faces=raw.faces,
        **attributes,
    )
