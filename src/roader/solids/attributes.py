"""Shading normals and UV sets derived from the model (FR-AS-3, FR-AS-4, FR-AS-5).

All three are stored per face corner (``Mesh`` docstring): a solid keeps one vertex
where a crease or UV seam passes, and the corners on each side differ.

Normals (FR-AS-3). Corners are smoothed across an edge only when the model says the
surface continues there: top faces across station rows and across u-lines that are
not creases of the body (``Body.creases``: barrier arrises, the slope break at an
embankment's pavement edge); bottom faces everywhere; walls along one outline run
(all along-road edges, or all edges of one end wall). Top, bottom and wall faces
never share a normal, and outline corners between along-road and end walls are hard.
No angle threshold is used.

UV set 0 (FR-AS-4), metres:

- top faces on a road grid: (distance along the offset line, distance across the
  section). Along is ``s - u * psi(s)`` with ``psi`` the heading change since the
  start of the UV segment: its station derivative is ``1 - kappa * u``, the length
  of the offset line, so there is no stretch along curves (D-039). Shear grows with
  ``psi``, so a segment ends after ``PSI_MAX`` of turning (stretch across
  ``1 - cos`` about 3%); a straight road is one segment and fully continuous.
  Across is the arc length along the station row, signed from the reference line
  (or from the body's first u-line when the row does not cross it), so barrier
  faces and embankment slopes do not stretch.
- walls: (horizontal distance along the outline, elevation).
- bottoms and anything off a road grid: an isometric projection onto the face plane.

UV set 1 (FR-AS-5): charts are the top (station, across), the bottom (station,
offset, or plan x, y off the grid) and the walls (perimeter along the outline,
elevation), cut into chunks of at most ``CHUNK_M`` and shelf-packed with
``UV1_PAD_PX`` of padding on a ``UV1_ATLAS_PX`` square atlas at one scale per mesh,
so texel density is uniform within a mesh and stated in the exports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from roader.model.mesh import Mesh
from roader.model.schema import Road
from roader.solids.assemble import (
    BOTTOM,
    TOP,
    WALL,
    RawMesh,
    boundary_edges,
    chain_ring,
    plain_mesh,
)
from roader.solids.grid import PlanVertices

PSI_MAX = 0.25  # rad of turning per UV0 segment
UV0_SAMPLE_M = 1.0  # station step used to place UV0 segment ends
CHUNK_M = 32.0  # longest UV1 chart side along the road or outline, m
UV1_ATLAS_PX = 1024
UV1_PAD_PX = 2
MITER_LIMIT = 4.0
ROW_ORIGIN_TOL_M = 1e-3  # a top row touching the reference line within this is measured from it

ALONG, ACROSS, OTHER = 0, 1, 2


class RoadFrame:
    """Heading, curvature and UV0 segments of one road."""

    def __init__(self, road: Road) -> None:
        self.road = road
        self.length = road.length
        self._cache: dict[float, tuple[float, float]] = {}
        starts = [0.0]
        ref = self.heading(0.0)
        n = max(1, math.ceil(self.length / UV0_SAMPLE_M))
        prev = 0.0
        for k in range(1, n + 1):
            s = self.length * k / n
            if abs(math.remainder(self.heading(s) - ref, 2.0 * math.pi)) > PSI_MAX:
                starts.append(prev)
                ref = self.heading(prev)
            prev = s
        self.starts = np.array(starts)
        self.refs = np.array([self.heading(s) for s in starts])

    def state(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), self.length)
        hit = self._cache.get(s)
        if hit is None:
            _x, _y, heading, k = self.road.alignment.evaluate(s)
            hit = self._cache[s] = (heading, k)
        return hit

    def heading(self, s: float) -> float:
        return self.state(s)[0]

    def segment(self, s: np.ndarray) -> np.ndarray:
        return np.maximum(np.searchsorted(self.starts, s, side="right") - 1, 0)


# ---------------------------------------------------------------- entry points


def finish(raw: RawMesh, pv: PlanVertices, frames: Mapping[str, RoadFrame], *, realtime: bool = False) -> Mesh:
    """Mesh of one body with normals, uv0 and uv1; real-time bodies are inset first."""
    body = raw.body
    vertices = raw.vertices.copy()
    plan = raw.plan
    s = np.array([pv.s[p] for p in plan], dtype=np.float64)
    u = np.array([pv.u[p] for p in plan], dtype=np.float64)
    road = [pv.road[p] for p in plan]
    row = np.array([pv.row[p] for p in plan], dtype=np.int64)
    if realtime and body.inset_m > 0.0:
        _inset(raw, pv, frames, vertices, s, u, road, body.inset_m)
    faces = raw.faces
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(fn, axis=1)
    fn = fn / np.where(norm > 0.0, norm, 1.0)[:, None]

    edge_type = _edge_types(raw.wall_edge, pv)
    normals = _normals(raw, pv, tri, fn, edge_type)
    uv0, top_across, top_ok = _uv0(raw, vertices, s, u, road, row, frames, fn, edge_type)
    uv1 = _uv1(raw, pv, vertices, s, u, row, top_across, top_ok)
    return plain_mesh(raw, vertices, normals=normals, uv0=uv0, uv1=uv1)


def terrain_attributes(mesh: Mesh) -> Mesh:
    """Smooth per-vertex normals and a plan-projection atlas (uv1) for the open L7 surface.

    The terrain has no tiling uv0: it is not a road surface (FR-AS-4), and engines
    texture it in world space.
    """
    v, f = mesh.vertices, mesh.faces
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros_like(v)
    for k in range(3):
        np.add.at(acc, f[:, k], fn)
    n = acc / np.linalg.norm(acc, axis=1)[:, None]
    lo = v[:, :2].min(axis=0)
    extent = float((v[:, :2].max(axis=0) - lo).max())
    pad = UV1_PAD_PX / UV1_ATLAS_PX
    uv1 = pad + (v[:, :2] - lo) * ((1.0 - 2.0 * pad) / extent)
    mesh.normals = n
    mesh.uv1 = uv1
    return mesh


# ---------------------------------------------------------------- real-time inset


def _inset(
    raw: RawMesh, pv: PlanVertices, frames: Mapping[str, RoadFrame], vertices: np.ndarray,
    s: np.ndarray, u: np.ndarray, road: list[str | None], inset: float,
) -> None:
    """Move the outline inward in plan by ``inset`` (mitred), keeping elevations.

    Walls stay vertical and tops keep their model elevation; only the plan outline
    moves. (s, u) of moved grid vertices follow to first order for the UVs.
    """
    edges = boundary_edges(raw.body.cells)
    incoming: dict[int, tuple[int, int]] = {}
    outgoing: dict[int, tuple[int, int]] = {}
    for a, b in edges:
        outgoing.setdefault(a, (a, b))
        incoming.setdefault(b, (a, b))
    shift: dict[int, tuple[float, float]] = {}
    for p in sorted(set(incoming) & set(outgoing)):
        n1 = _left_normal(pv, *incoming[p])
        n2 = _left_normal(pv, *outgoing[p])
        denom = 1.0 + n1[0] * n2[0] + n1[1] * n2[1]
        scale = inset / max(denom, 2.0 / MITER_LIMIT**2)
        shift[p] = ((n1[0] + n2[0]) * scale, (n1[1] + n2[1]) * scale)
    for idx, p in enumerate(raw.plan):
        d = shift.get(int(p))
        if d is None:
            continue
        vertices[idx, 0] += d[0]
        vertices[idx, 1] += d[1]
        rid = road[idx]
        if rid is None or not math.isfinite(s[idx]):
            continue
        heading, k = frames[rid].state(float(s[idx]))
        c, sn = math.cos(heading), math.sin(heading)
        du = -sn * d[0] + c * d[1]
        ds = (c * d[0] + sn * d[1]) / (1.0 - k * u[idx])
        s[idx] += ds
        u[idx] += du


def _left_normal(pv: PlanVertices, a: int, b: int) -> tuple[float, float]:
    dx, dy = pv.x[b] - pv.x[a], pv.y[b] - pv.y[a]
    length = math.hypot(dx, dy)
    return -dy / length, dx / length


# ---------------------------------------------------------------- normals


def _edge_types(wall_edge: np.ndarray, pv: PlanVertices) -> np.ndarray:
    out = np.full(len(wall_edge), OTHER, dtype=np.int64)
    for f, (a, b) in enumerate(wall_edge.tolist()):
        if a < 0 or pv.row[a] < 0 or pv.row[b] < 0 or pv.road[a] != pv.road[b]:
            continue
        out[f] = ACROSS if pv.row[a] == pv.row[b] else ALONG
    return out


def _normals(raw: RawMesh, pv: PlanVertices, tri: np.ndarray, fn: np.ndarray, edge_type: np.ndarray) -> np.ndarray:
    faces, kind = raw.faces, raw.kind
    m = len(faces)
    creases = raw.body.creases
    plan = raw.plan
    on_crease = {int(i) for i, p in enumerate(plan) if creases & pv.keys.get(int(p), set())}

    def crease(v: int, w: int) -> bool:
        if v not in on_crease or w not in on_crease:
            return False
        pa, pb = int(plan[v]), int(plan[w])
        return pv.row[pa] != pv.row[pb] and bool(creases & pv.keys[pa] & pv.keys[pb])

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fid = np.tile(np.arange(m), 3)
    pos = np.repeat(np.arange(3), m)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, e, fid, pos = key[order], e[order], fid[order], pos[order]
    same = np.all(key[1:] == key[:-1], axis=1)
    rows, cols = [], []
    wall_edge = raw.wall_edge
    for j in np.nonzero(same)[0].tolist():
        f, g = int(fid[j]), int(fid[j + 1])
        kf = kind[f]
        if kf != kind[g]:
            continue
        v, w = int(e[j, 0]), int(e[j, 1])
        if kf == TOP and crease(v, w):
            continue
        if kf == WALL:
            ef, eg = wall_edge[f], wall_edge[g]
            same_edge = {int(ef[0]), int(ef[1])} == {int(eg[0]), int(eg[1])}
            if not same_edge and (edge_type[f] != edge_type[g] or edge_type[f] == OTHER):
                continue
        for x in (v, w):
            rows.append(f * 3 + _corner(faces[f], x))
            cols.append(g * 3 + _corner(faces[g], x))
    n_corner = 3 * m
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_corner, n_corner))
    _n, label = connected_components(graph, directed=False)
    weight = _corner_angles(tri).reshape(-1)
    acc = np.zeros((label.max() + 1, 3))
    np.add.at(acc, label, np.repeat(fn, 3, axis=0) * weight[:, None])
    length = np.linalg.norm(acc, axis=1)
    acc /= np.where(length > 0.0, length, 1.0)[:, None]
    return acc[label].reshape(m, 3, 3)


def _corner(face: np.ndarray, v: int) -> int:
    return int(np.nonzero(face == v)[0][0])


def _corner_angles(tri: np.ndarray) -> np.ndarray:
    out = np.empty(tri.shape[:2])
    for k in range(3):
        a = tri[:, (k + 1) % 3] - tri[:, k]
        b = tri[:, (k + 2) % 3] - tri[:, k]
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        out[:, k] = np.arccos(np.clip(cos, -1.0, 1.0))
    return out


# ---------------------------------------------------------------- uv0


def _row_across(raw: RawMesh, vertices: np.ndarray, u: np.ndarray, row: np.ndarray, road: list[str | None]) -> dict[int, float]:
    """Arc length along each station row of the top surface, from the row's first u-line."""
    top = np.unique(raw.faces[raw.kind == TOP])
    groups: dict[tuple[str, int], list[int]] = {}
    for v in top.tolist():
        if row[v] >= 0 and road[v] is not None:
            groups.setdefault((road[v], int(row[v])), []).append(v)
    out: dict[int, float] = {}
    for verts in groups.values():
        verts.sort(key=lambda v: (u[v], v))
        uu = u[verts]
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(uu), np.diff(vertices[verts, 2])))])
        # Measured from the reference line when the row crosses or touches it (a real-time inset moves a
        # reference-line vertex by a fraction of a millimetre, D-092), else from the first u-line.
        origin = float(np.interp(0.0, uu, arc)) if uu[0] <= ROW_ORIGIN_TOL_M and uu[-1] >= -ROW_ORIGIN_TOL_M else -float(uu[0])
        for v, a in zip(verts, arc.tolist(), strict=True):
            out[v] = a - origin
    return out


def _uv0(
    raw: RawMesh, vertices: np.ndarray, s: np.ndarray, u: np.ndarray, road: list[str | None], row: np.ndarray,
    frames: Mapping[str, RoadFrame], fn: np.ndarray, edge_type: np.ndarray,
) -> tuple[np.ndarray, dict[int, float], np.ndarray]:
    faces, kind = raw.faces, raw.kind
    m = len(faces)
    uv = np.empty((m, 3, 2))
    across = _row_across(raw, vertices, u, row, road)
    grid = np.array([r is not None and math.isfinite(sv) for r, sv in zip(road, s, strict=True)])
    road_of = np.array([r or "" for r in road], dtype=object)
    same_road = grid[faces].all(axis=1) & (road_of[faces[:, 0]] == road_of[faces[:, 1]]) & (road_of[faces[:, 0]] == road_of[faces[:, 2]])
    top_ok = (kind == TOP) & same_road & np.array([all(int(v) in across for v in f) for f in faces])
    for f in np.nonzero(top_ok)[0].tolist():
        frame = frames[road[int(faces[f, 0])]]
        seg = int(frame.segment(np.array([s[faces[f]].mean()]))[0])
        ref = float(frame.refs[seg])
        for k, v in enumerate(faces[f].tolist()):
            psi = math.remainder(frame.heading(float(s[v])) - ref, 2.0 * math.pi)
            uv[f, k] = (s[v] - u[v] * psi, across[v])
    wall = np.nonzero(kind == WALL)[0]
    for f in wall.tolist():
        a, b = (int(x) for x in raw.wall_edge[f])
        ia = [k for k, v in enumerate(faces[f].tolist()) if raw.plan[v] == a]
        ref_v = int(faces[f, ia[0]]) if ia else int(faces[f, 0])
        other = next((int(v) for v in faces[f] if raw.plan[v] == b), None)
        if edge_type[f] == ALONG:
            frame = frames[road[ref_v]]
            h_ref = _along(frame, s[ref_v], u[ref_v])
            sign = 1.0 if other is None or _along(frame, s[other], u[other]) >= h_ref else -1.0
        elif edge_type[f] == ACROSS:
            h_ref = float(u[ref_v])
            sign = 1.0 if other is None or u[other] >= h_ref else -1.0
        else:
            h_ref, sign = 0.0, 1.0
        base = vertices[ref_v, :2]
        for k, v in enumerate(faces[f].tolist()):
            d = math.hypot(vertices[v, 0] - base[0], vertices[v, 1] - base[1])
            uv[f, k] = (h_ref + sign * d, vertices[v, 2])
    rest = np.nonzero(~top_ok & (kind != WALL))[0]
    if len(rest):
        uv[rest] = _planar(vertices[faces[rest]], fn[rest])
    return uv, across, top_ok


def _along(frame: RoadFrame, s: float, u: float) -> float:
    seg = int(frame.segment(np.array([s]))[0])
    psi = math.remainder(frame.heading(float(s)) - float(frame.refs[seg]), 2.0 * math.pi)
    return float(s - u * psi)


def _planar(tri: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Isometric projection of each triangle onto its own plane."""
    x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    horizontal = np.abs(fn[:, 2]) > 0.999
    e1 = np.where(horizontal[:, None], x - fn * fn[:, :1], np.cross(z, fn))
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(fn, e1)
    return np.stack([np.einsum("mkj,mj->mk", tri, e1), np.einsum("mkj,mj->mk", tri, e2)], axis=2)


# ---------------------------------------------------------------- uv1


def _uv1(
    raw: RawMesh, pv: PlanVertices, vertices: np.ndarray, s: np.ndarray, u: np.ndarray, row: np.ndarray,
    top_across: dict[int, float], top_ok: np.ndarray,
) -> np.ndarray:
    faces, kind = raw.faces, raw.kind
    m = len(faces)
    local = np.empty((m, 3, 2))
    chart = np.empty(m, dtype=object)

    top = np.nonzero(kind == TOP)[0]
    if len(top) and top_ok[top].all():
        for f in top.tolist():
            local[f] = [(s[v], top_across[int(v)]) for v in faces[f]]
            chart[f] = "t"
    else:
        local[top] = vertices[faces[top]][:, :, :2]
        chart[top] = "t"
    bottom = np.nonzero(kind == BOTTOM)[0]
    grid = np.isfinite(s) & np.isfinite(u)
    if len(bottom) and grid[faces[bottom]].all() and len({pv.road[int(p)] for p in raw.plan[np.unique(faces[bottom])]}) == 1:
        local[bottom] = np.stack([s[faces[bottom]], u[faces[bottom]]], axis=2)
    else:
        local[bottom] = vertices[faces[bottom]][:, :, :2]
    chart[bottom] = "b"

    wall = np.nonzero(kind == WALL)[0]
    if len(wall):
        perimeter = _perimeter(raw, pv, vertices)
        for f in wall.tolist():
            a, b = (int(x) for x in raw.wall_edge[f])
            h0, n = perimeter.get((a, b), (0.0, f))
            base = None
            for k, v in enumerate(faces[f].tolist()):
                if raw.plan[v] == a:
                    base = vertices[v, :2]
            for k, v in enumerate(faces[f].tolist()):
                d = 0.0 if base is None else math.hypot(vertices[v, 0] - base[0], vertices[v, 1] - base[1])
                local[f, k] = (h0 + d, vertices[v, 2])
            chart[f] = f"w{n}"

    # Chunks: faces of one chart grouped by their centroid along the chart's first axis.
    centre = local[:, :, 0].mean(axis=1)
    groups: dict[tuple[str, int], list[int]] = {}
    for name in sorted(set(chart.tolist())):
        idx = np.nonzero(chart == name)[0]
        start = local[idx, :, 0].min()
        for f, c in zip(idx.tolist(), centre[idx].tolist(), strict=True):
            groups.setdefault((name, int((c - start) // CHUNK_M)), []).append(f)
    boxes = []
    keys = sorted(groups)
    for key in keys:
        idx = np.array(groups[key])
        pts = local[idx].reshape(-1, 2)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        boxes.append((lo, hi - lo))
    origin, scale = _pack([size for _lo, size in boxes])
    uv1 = np.empty((m, 3, 2))
    for key, (lo, _size), pos in zip(keys, boxes, origin, strict=True):
        idx = np.array(groups[key])
        uv1[idx] = (local[idx] - lo) * scale + pos
    return uv1


def _perimeter(raw: RawMesh, pv: PlanVertices, vertices: np.ndarray) -> dict[tuple[int, int], tuple[float, int]]:
    """Distance along the outline ring at the start of each wall edge, and the chart index."""
    edges = boundary_edges(raw.body.cells)
    try:
        ring = chain_ring(edges)
    except ValueError:
        return {(a, b): (0.0, k) for k, (a, b) in enumerate(edges)}
    # Final (possibly inset) plan positions per plan vertex.
    pos: dict[int, np.ndarray] = {}
    for idx, p in enumerate(raw.plan.tolist()):
        pos.setdefault(p, vertices[idx, :2])
    out: dict[tuple[int, int], tuple[float, int]] = {}
    h = 0.0
    for k, a in enumerate(ring):
        b = ring[(k + 1) % len(ring)]
        out[(a, b)] = (h, 0)
        pa = pos.get(a, np.array([pv.x[a], pv.y[a]]))
        pb = pos.get(b, np.array([pv.x[b], pv.y[b]]))
        h += float(np.hypot(*(pb - pa)))
    return out


def _pack(sizes: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Shelf-pack boxes (metres) into the unit square with UV1_PAD_PX padding.

    Returns each box's lower-left corner in uv and the uniform scale (uv per metre).
    The atlas side in metres is the smallest one that fits, found by bisection.
    """
    sizes = [np.maximum(np.asarray(sz, dtype=np.float64), 1e-6) for sz in sizes]
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0], i))
    pad_frac = UV1_PAD_PX / UV1_ATLAS_PX

    def place(side: float) -> list[np.ndarray] | None:
        pad = pad_frac * side
        x = y = shelf = pad
        out: list[np.ndarray | None] = [None] * len(sizes)
        shelf_h = 0.0
        for i in order:
            w, h = sizes[i]
            if w + 2 * pad > side:
                return None
            if x + w + pad > side:
                x = pad
                shelf += shelf_h + pad
                shelf_h = 0.0
            out[i] = np.array([x, shelf])
            x += w + pad
            shelf_h = max(shelf_h, h)
        y = shelf + shelf_h + pad
        return out if y <= side else None  # type: ignore[return-value]

    area = sum(float(w * h) for w, h in sizes)
    lo = math.sqrt(area)
    hi = max(lo, 1.0)
    while place(hi) is None:
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if place(mid) is None:
            lo = mid
        else:
            hi = mid
    corners = place(hi)
    assert corners is not None
    return [c / hi for c in corners], 1.0 / hi
