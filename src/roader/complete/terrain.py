"""L7 terrain completion (FR-OC-3, FR-SG-6 input).

The terrain patch is the bare earth the solids stand on, so it has to be complete
and has to stay out of every solid that does not stand on it:

1. Observed ground: median height of ground-classified returns per grid node.
2. Unobservable ground is removed: under every deck, and under every road out to
   the toe of its fill embankment (the measured slope surface is the embankment,
   not the ground). The toe is where the measured side profile stops falling at
   the fill rate.
3. Every gap is filled from observed nodes by linear interpolation (nearest node
   outside their hull), so ground under a bridge comes from the ground around it,
   and ground under an embankment runs from toe to toe.
4. Nodes under a pavement slab, a deck or a pier cap are lowered below it; those
   bodies are not ground-contact solids and terrain must not cross them.

Each node's origin is recorded in ``TerrainPatch.fill``. Support footings are
then seated below the completed terrain (``seat_supports``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import griddata

from roader.conflate.polyline import Polyline
from roader.io.points import PointSet
from roader.model.provenance import Provenance, Source
from roader.model.schema import RoadModel, TerrainPatch
from roader.rules import RuleBook

CELL_M = 2.0
GROUND_CLASS = 2
MIN_GROUND_POINTS = 2
TOE_SEARCH_M = 40.0
TOE_STEP_M = 1.0
STATION_STEP_M = 5.0
NONE_REACH_M = 1.0  # beyond the pavement edge on sides without a fill
LOWER_REACH_CELLS = 3.0  # interpolated nodes this many cells beyond a pavement are lowered below its underside
LOWER_MARGIN_M = 0.1  # terrain below a slab underside or cap bottom by at least this
CAP_CLEARANCE_M = 0.3
FOOTING_MARGIN_M = 0.05

MEASURED, GAP, UNDER_STRUCTURE, UNDER_ROAD, LOWERED = 0, 1, 2, 3, 4


class Grid:
    def __init__(self, x0: float, y0: float, cell: float, nx: int, ny: int, z: np.ndarray) -> None:
        self.x0, self.y0, self.cell, self.nx, self.ny = x0, y0, cell, nx, ny
        self.z = z  # (ny, nx), NaN where unknown
        jj, ii = np.mgrid[0:ny, 0:nx]
        self.xy = np.column_stack([x0 + ii.ravel() * cell, y0 + jj.ravel() * cell])

    def nearest(self, xy: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint((xy[:, 0] - self.x0) / self.cell).astype(np.int64), 0, self.nx - 1)
        j = np.clip(np.rint((xy[:, 1] - self.y0) / self.cell).astype(np.int64), 0, self.ny - 1)
        return self.z[j, i]

    def bilinear(self, xy: np.ndarray) -> np.ndarray:
        fx = np.clip((xy[:, 0] - self.x0) / self.cell, 0.0, self.nx - 1.0)
        fy = np.clip((xy[:, 1] - self.y0) / self.cell, 0.0, self.ny - 1.0)
        i = np.minimum(np.floor(fx).astype(np.int64), self.nx - 2)
        j = np.minimum(np.floor(fy).astype(np.int64), self.ny - 2)
        tx, ty = fx - i, fy - j
        z = self.z
        a = z[j, i] * (1 - tx) + z[j, i + 1] * tx
        b = z[j + 1, i] * (1 - tx) + z[j + 1, i + 1] * tx
        return a * (1 - ty) + b * ty


def _observed_ground(points: PointSet, cell: float) -> Grid:
    xy = points.xyz[:, :2]
    x0 = math.floor(xy[:, 0].min() / cell) * cell
    y0 = math.floor(xy[:, 1].min() / cell) * cell
    nx = math.ceil((xy[:, 0].max() - x0) / cell) + 1
    ny = math.ceil((xy[:, 1].max() - y0) / cell) + 1
    g = points.classification == GROUND_CLASS
    i = np.rint((xy[g, 0] - x0) / cell).astype(np.int64)
    j = np.rint((xy[g, 1] - y0) / cell).astype(np.int64)
    key = j * nx + i
    z = points.xyz[g, 2]
    order = np.lexsort((z, key))
    k, zs = key[order], z[order]
    uk, start, cnt = np.unique(k, return_index=True, return_counts=True)
    med = 0.5 * (zs[start + (cnt - 1) // 2] + zs[start + cnt // 2])
    grid = np.full(nx * ny, np.nan)
    ok = cnt >= MIN_GROUND_POINTS
    grid[uk[ok]] = med[ok]
    return Grid(x0, y0, cell, nx, ny, grid.reshape(ny, nx))


def _pw(piece, s: np.ndarray) -> np.ndarray:
    return np.interp(s, [k.s for k in piece.knots], [k.value for k in piece.knots])


def _fill_ratio(model: RoadModel, road_id: str, side: str, s: float) -> float | None:
    for ss in model.side_slopes:
        if ss.road_id == road_id and ss.side == side and ss.s0 - 1e-6 <= s <= ss.s1 + 1e-6:
            return ss.ratio if ss.kind == "fill" and ss.ratio > 0 else None
    return None


def _reach(model: RoadModel, road, line: Polyline, ground: Grid, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Station samples and the distance beyond the pavement edge that is embankment, not ground."""
    sign = 1.0 if side == "left" else -1.0
    piece = road.cross_section.width_left if side == "left" else road.cross_section.width_right
    st = np.arange(0.0, road.length + 1e-9, STATION_STEP_M)
    d = np.arange(0.0, TOE_SEARCH_M + 1e-9, TOE_STEP_M)
    out = np.full(len(st), NONE_REACH_M)
    for n, s in enumerate(st):
        ratio = _fill_ratio(model, road.id, side, float(s))
        if ratio is None:
            continue
        w = piece(float(s))
        z_edge = road.surface_z(float(s), sign * w)
        z = ground.nearest(line.offset_point(np.full(len(d), s), sign * (w + d)))
        toe = TOE_SEARCH_M
        for k in range(1, len(d) - 2):
            if not (np.isfinite(z[k]) and np.isfinite(z[k + 2])):
                continue
            falling = (z[k] - z[k + 2]) / (d[k + 2] - d[k])
            if falling < 0.5 / ratio and z[k] < z_edge - 0.3:
                toe = d[k]
                break
        out[n] = toe + 1.0
    # A toe is never closer than its neighbours' minus one step: no notches in the zone.
    smooth = np.maximum(out, np.maximum(np.roll(out, 1), np.roll(out, -1)))
    smooth[0], smooth[-1] = max(out[0], out[1]) if len(out) > 1 else out[0], max(out[-1], out[-2]) if len(out) > 1 else out[-1]
    return st, smooth


def complete_terrain(points: PointSet, model: RoadModel, rules: RuleBook, cell: float = CELL_M) -> TerrainPatch:
    ground = _observed_ground(points, cell)
    z0 = ground.z.ravel().copy()
    code = np.where(np.isfinite(z0), MEASURED, GAP).astype(np.int64)
    exclude = np.zeros(len(z0), dtype=bool)
    lines = {r.id: Polyline.from_alignment(r.alignment) for r in model.roads}
    proj = {}
    for road in model.roads:
        s, u, _ok = lines[road.id].project(ground.xy, max_dist=TOE_SEARCH_M + 30.0)
        proj[road.id] = (np.nan_to_num(s, nan=-1e9), np.nan_to_num(u, nan=1e9))

    # Unobservable ground: under decks and under roads out to the embankment toe.
    for st in model.structures:
        road = model.road(st.carried_road)
        s, u = proj[road.id]
        xs = road.cross_section
        pad = st.deck.overhang_m + 1.0
        deck = (s >= st.s_start - 0.5) & (s <= st.s_end + 0.5) & (u <= _pw(xs.width_left, s) + pad) & (u >= -_pw(xs.width_right, s) - pad)
        exclude |= deck
        code[deck] = UNDER_STRUCTURE
    for road in model.roads:
        s, u = proj[road.id]
        xs = road.cross_section
        inside = (s >= -cell) & (s <= road.length + cell)
        for side, sign, piece in (("left", 1.0, xs.width_left), ("right", -1.0, xs.width_right)):
            st, reach = _reach(model, road, lines[road.id], ground, side)
            a = sign * u
            zone = inside & (a >= -0.5) & (a <= _pw(piece, s) + np.interp(s, st, reach))
            exclude |= zone
            code[zone & (code != UNDER_STRUCTURE)] = UNDER_ROAD

    known = np.isfinite(z0) & ~exclude
    z = z0.copy()
    z[exclude] = np.nan
    missing = ~np.isfinite(z)
    if missing.any():
        if known.sum() >= 3:
            z[missing] = griddata(ground.xy[known], z0[known], ground.xy[missing], method="linear")
        still = ~np.isfinite(z)
        if still.any():
            z[still] = griddata(ground.xy[known], z0[known], ground.xy[still], method="nearest")

    # One binomial pass: bumps between nodes would make embankment toes jump.
    z = _smooth(z.reshape(ground.ny, ground.nx)).ravel()

    # Lower nodes under bodies that do not stand on the ground.
    for road in model.roads:
        s, u = proj[road.id]
        xs = road.cross_section
        wl, wr = _pw(xs.width_left, s), _pw(xs.width_right, s)
        # The re-triangulation band joins nodes up to two cells apart (FR-SG-6), so interpolated nodes
        # that far beyond the edge are held at the edge's underside too (D-074); measured ones as before.
        reach = np.where(code == MEASURED, 1.5, LOWER_REACH_CELLS) * cell
        near = (s >= -reach) & (s <= road.length + reach) & (u <= wl + reach) & (u >= -wr - reach)
        idx = np.flatnonzero(near)
        if idx.size == 0:
            continue
        sc = np.clip(s[idx], 0.0, road.length)
        uc = np.clip(u[idx], -wr[idx], wl[idx])
        under = np.array([road.surface_z(float(a), float(b)) for a, b in zip(sc, uc, strict=True)]) - _pw(xs.pavement_thickness, sc)
        limit = under - LOWER_MARGIN_M
        low = z[idx] > limit
        z[idx[low]] = limit[low]
        code[idx[low]] = LOWERED
    cap_d = max((sp.cap_depth_m for st in model.structures for sp in st.supports), default=0.0)
    for st in model.structures:
        road = model.road(st.carried_road)
        s, u = proj[road.id]
        xs = road.cross_section
        pad = st.deck.overhang_m + 1.5 * cell
        idx = np.flatnonzero((s >= st.s_start - cell) & (s <= st.s_end + cell) & (u <= _pw(xs.width_left, s) + pad) & (u >= -_pw(xs.width_right, s) - pad))
        if idx.size == 0:
            continue
        sc = np.clip(s[idx], st.s_start, st.s_end)
        soffit = np.array([road.profile.z(float(a)) for a in sc]) - st.deck.depth_m
        limit = soffit - cap_d - CAP_CLEARANCE_M
        low = z[idx] > limit
        z[idx[low]] = limit[low]
        code[idx[low]] = LOWERED

    n_meas = int((code == MEASURED).sum())
    prov = Provenance(
        Source.MEASURED,
        0.8 * n_meas / len(code),
        n_points=n_meas,
        observed_fraction=n_meas / len(code),
        flags=["FR-OC-3:completed", f"FR-OC-3:filled_cells:{len(code) - n_meas}"],
    )
    return TerrainPatch(
        float(ground.x0),
        float(ground.y0),
        float(cell),
        int(ground.nx),
        int(ground.ny),
        [round(float(v), 4) for v in z],
        prov,
        [int(c) for c in code],
    )


def _smooth(z: np.ndarray) -> np.ndarray:
    """3 x 3 binomial filter with edge replication."""
    p = np.pad(z, 1, mode="edge")
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    rows = k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]
    return k[0] * rows[:-2] + k[1] * rows[1:-1] + k[2] * rows[2:]


def terrain_grid(patch: TerrainPatch) -> Grid:
    return Grid(patch.origin_x, patch.origin_y, patch.cell_m, patch.nx, patch.ny, np.asarray(patch.z, dtype=np.float64).reshape(patch.ny, patch.nx))


def seat_supports(model: RoadModel, rules: RuleBook) -> None:
    """Footings below the completed terrain everywhere under the support block (FR-OC-3)."""
    if model.terrain is None:
        return
    grid = terrain_grid(model.terrain)
    foot, _prov = rules.fill("footing.depth")
    for st in model.structures:
        road = model.road(st.carried_road)
        line = Polyline.from_alignment(road.alignment)
        xs = road.cross_section
        for sp in st.supports:
            if sp.kind == "abutment":
                near_start = abs(sp.station - st.s_start) <= abs(sp.station - st.s_end)
                a, b = (st.s_start, st.s_start + sp.cap_width_m) if near_start else (st.s_end - sp.cap_width_m, st.s_end)
            else:
                a, b = sp.station - 0.5 * sp.cap_width_m, sp.station + 0.5 * sp.cap_width_m
            half_l = xs.width_left(sp.station) + st.deck.overhang_m + 2.0
            half_r = xs.width_right(sp.station) + st.deck.overhang_m + 2.0
            ss, uu = np.meshgrid(np.linspace(a, b, 5), np.arange(-half_r, half_l + 1e-9, 0.5))
            ss, uu = ss.ravel(), uu.ravel()
            ss = ss + uu * math.tan(sp.skew_rad)
            lowest = float(np.min(grid.bilinear(line.offset_point(ss, uu))))
            target = lowest - foot - FOOTING_MARGIN_M
            if sp.footing_z > target:
                sp.footing_z = target
                sp.prov.flags.append("FR-OC-3:footing_below_terrain")
