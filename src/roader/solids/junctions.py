"""L6 junction solids conforming with the road sweeps (FR-SG-3, FR-SV-7, D-046).

A junction area replaces the road bodies over each participant's station interval:
pavement, embankment and barrier cells of the road grid inside the interval are
dropped, and the area is built as a pavement slab (plus, over terrain, a fill
block down to the ground) on a triangulation of its outline. Along every seam
the slab uses the road grid's own vertices at the seam station, with the road's
own top and underside elevations there, so the faces where the slab meets the
road sweep are shared exactly; the assembler cuts the shared walls at every
level present (``roader.solids.assemble``).

Seam stations are added to the road's sweep stations by a no-op knot on a copy of
the road's pavement thickness, so ``road_bodies`` is used unchanged.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from roader.model.geometry import Knot, Piecewise
from roader.model.schema import JunctionArea, Road, RoadModel
from roader.solids.assemble import Body
from roader.solids.bodies import (
    INSET_ROAD_M,
    INSET_SUPPORT_M,
    MAT_PAVEMENT,
    MAT_SOIL,
    REALTIME_EMBED_M,
    mesh_id,
    pavement_bottom,
)
from roader.solids.failure import element
from roader.solids.grid import Cell, PlanVertices
from roader.solids.stations import STATION_MERGE_M
from roader.solids.terrain import TerrainSampler

CLIPPED_PARTS = {"pavement", "embankment", "barrier", "parapet"}
ROW_TOL_M = 1e-6
SKEW_SNAP_M = 0.05  # a skewed deck-end row in the outline matches the grid row within this (D-073)
OUTLINE_SNAP_M = 0.01  # outline vertices this close to a seam row end are that end (D-011 rounding of the road)
STEINER_M = 2.5  # interior vertex spacing of the area triangulation
STEINER_CLEAR = 0.4  # Steiner points keep STEINER_CLEAR * STEINER_M from the boundary and each other
MIN_FILL_M = 0.05  # fill block never thinner than this under the slab
RT_GROUND_CLEAR_M = 1e-3  # real-time fill top stays this far above the terrain and below the slab top
EPS_AREA = 1e-9
EPS_OVERLAP_M2 = 1e-6  # plan overlap of two cells below this is contact, not overlap


def zones(model: RoadModel) -> dict[str, list[tuple[float, float]]]:
    """Per road, the station intervals taken over by junction areas (from the seam sides)."""
    lengths = {r.id: r.length for r in model.roads}
    out: dict[str, list[tuple[float, float]]] = {}
    for jn in model.junctions:
        # A participant without seams lies wholly inside the area (a shared area, D-070).
        for rid in sorted({sm.road_id for sm in jn.seams} | set(jn.road_ids)):
            if rid not in lengths:
                raise ValueError(f"junction {jn.id}: seam on missing road {rid}")
            lo = [sm.s for sm in jn.seams if sm.road_id == rid and sm.side > 0]
            hi = [sm.s for sm in jn.seams if sm.road_id == rid and sm.side < 0]
            if any(sm.side == 0 for sm in jn.seams if sm.road_id == rid):
                raise ValueError(f"junction {jn.id}: seam on {rid} has no side (D-046)")
            a = min(lo) if lo else 0.0
            b = max(hi) if hi else lengths[rid]
            if b - a <= STATION_MERGE_M:
                raise ValueError(f"junction {jn.id}: empty interval on {rid}")
            out.setdefault(rid, []).append((a, b))
    return {rid: sorted(v) for rid, v in out.items()}


def _in_zone(s: float, ranges: list[tuple[float, float]]) -> bool:
    return any(a + 1e-9 < s < b - 1e-9 for a, b in ranges)


class JunctionPlan:
    """Wraps road body generation: seam stations in, zone cells out, then the junction bodies."""

    def __init__(self, model: RoadModel, pv: PlanVertices) -> None:
        self.model = model
        self.pv = pv
        self.zones = zones(model) if model.junctions else {}
        self.seam_s: dict[str, list[float]] = {}
        for jn in model.junctions:
            for sm in jn.seams:
                self.seam_s.setdefault(sm.road_id, []).append(sm.s)
        self.ranges: dict[str, tuple[int, int]] = {}
        self._current: tuple[str, int] | None = None
        self._xy: dict[tuple[float, float], int] = {}  # outline vertices shared by touching areas (D-021)
        self._floor: dict[int, float] = {}  # seam vertices at a deck end: the soffit there (D-073)

    def road(self, road: Road) -> Road:
        """The road to sweep: a copy with a sweep station at every seam."""
        self._current = (road.id, len(self.pv.x))
        stations = self.seam_s.get(road.id)
        if not stations:
            return road
        pw = road.cross_section.pavement_thickness
        knots = list(pw.knots)
        for s in sorted(set(stations)):
            if any(abs(k.s - s) <= 1e-9 for k in knots):
                continue
            prov = knots[0].prov if knots else None
            knots.append(Knot(s, pw(s), prov))  # type: ignore[arg-type]
        knots.sort(key=lambda k: k.s)
        xs = dataclasses.replace(road.cross_section, pavement_thickness=Piecewise(knots))
        return dataclasses.replace(road, cross_section=xs)

    def clip(self, bodies: list[Body]) -> list[Body]:
        """Drop the road's cells inside its junction intervals; split and renumber the bodies they cut."""
        assert self._current is not None
        rid, n0 = self._current
        self.ranges[rid] = (n0, len(self.pv.x))
        ranges = self.zones.get(rid)
        if not ranges:
            return bodies
        pv = self.pv
        # A cell lies between two station lines; skewed lines (deck ramps) carry their station at the centre line.
        line_s = {pv.row[p]: pv.s[p] for p in range(n0, len(pv.x)) if pv.road[p] == rid and pv.u[p] == 0.0}

        def station(p: int) -> float:
            return line_s.get(pv.row[p], pv.s[p])

        pieces: list[tuple[Body, list[list[Cell]]]] = []
        renumber: set[tuple[str, str]] = set()
        for body in bodies:
            mids = [sum(station(p) for p in c.loop) / len(c.loop) for c in body.cells]
            hit = [math.isfinite(m) and _in_zone(m, ranges) for m in mids]
            if not any(hit):
                pieces.append((body, [body.cells]))
                continue
            renumber.add((body.element_id, body.part))
            if body.part not in CLIPPED_PARTS:
                raise ValueError(f"{body.mesh_id}: a junction area reaches this {body.part} (FR-SG-3)")
            # Cells between zone k-1 and zone k form run k.
            by_run: dict[int, list[Cell]] = {}
            for c, m, h in zip(body.cells, mids, hit, strict=True):
                if not h:
                    by_run.setdefault(sum(b <= m for _a, b in ranges), []).append(c)
            pieces.append((body, [by_run[k] for k in sorted(by_run)]))
        counts: dict[tuple[str, str], int] = {}
        for body, runs in pieces:
            key = (body.element_id, body.part)
            counts[key] = counts.get(key, 0) + len(runs)
        seen: dict[tuple[str, str], int] = {}
        out: list[Body] = []
        for body, runs in pieces:
            key = (body.element_id, body.part)
            if key not in renumber:
                out.append(body)
                continue
            for run in runs:
                k = seen.get(key, 0)
                seen[key] = k + 1
                out.append(dataclasses.replace(body, mesh_id=mesh_id(body.element_id, body.part, k, counts[key]), cells=run, zb={}, zt={}))
        return out

    def bodies(self, sampler: TerrainSampler | None, tol: float, *, realtime: bool = False) -> list[Body]:
        """Area bodies; ``realtime`` builds them as the road bodies are in that profile (D-047):
        the slab embeds into its fill, the fill into the ground, and both outlines are inset."""
        areas = sorted(self.model.junctions, key=lambda j: j.id)
        outlines = []
        for jn in areas:
            with element(jn.id):
                outlines.append(self._outline(jn))
        pv = self.pv
        # A seam vertex of one area on another's outline (a shared stretch starting at the seam, D-154)
        # is one vertex: the other uses the seam's grid vertex and its top and underside.
        seam_vertex = {(round(pv.x[p], 6), round(pv.y[p], 6)): p for _r, _h, exact in outlines for p in exact}
        seam_z = {p: zz for _r, _h, exact in outlines for p, zz in exact.items()}

        def same(p: int) -> int:
            return seam_vertex.get((round(pv.x[p], 6), round(pv.y[p], 6)), p)

        outlines = [([same(p) for p in ring], [[same(p) for p in h] for h in holes], exact) for ring, holes, exact in outlines]
        users: dict[int, list[JunctionArea]] = {}
        for jn, (ring, holes, exact) in zip(areas, outlines, strict=True):
            for p in [*ring, *(q for h in holes for q in h)]:
                if p not in exact:
                    users.setdefault(p, []).append(jn)
        # Where areas touch (a shared split, D-070) both use one top and one underside, so their walls conform.
        common: dict[int, tuple[float, float]] = {}
        for p, jns in users.items():
            if p in seam_z:
                common[p] = seam_z[p]
            elif len(jns) > 1:
                top = sum(j.lifted_z(pv.x[p], pv.y[p]) for j in jns) / len(jns)
                common[p] = (top, top - max(j.thickness_at(pv.x[p], pv.y[p]) for j in jns))
        out: list[Body] = []
        for jn, (ring, holes, exact) in zip(areas, outlines, strict=True):
            with element(jn.id):
                out.extend(self._junction(jn, ring, holes, {**common, **exact}, sampler, realtime))
        return out

    def trim_fills(self, bodies: list[Body]) -> list[Body]:
        """Drop embankment cells beyond a road's pavement edge that overlap another element's body (D-071).

        Side slopes of roads that run close together, or beside an L6 area, would otherwise
        overlap in the gore. Only cells wholly outside the pavement edge are dropped, so the
        pavement keeps its fill; both of two overlapping slopes stop at their edges.
        """
        pv = self.pv
        model = self.model
        roads = {r.id: r for r in model.roads}
        owner_of: dict[str, str] = {ss.id: ss.road_id for ss in model.side_slopes}
        for r in model.roads:
            owner_of.update({b.id: r.id for b in r.cross_section.barriers})
        for st in model.structures:
            owner_of[st.id] = st.carried_road
            owner_of.update({sp.id: st.carried_road for sp in st.supports})

        def owner(body: Body) -> str:
            return owner_of.get(body.element_id, body.element_id)

        def zrange(body: Body, cell: Cell) -> tuple[float, float]:
            return min(body.bottom(p) for p in cell.loop), max(body.top(p) for p in cell.loop)

        def outer(road: Road, cell: Cell) -> bool:
            xs = road.cross_section
            for p in cell.loop:
                u, st = pv.u[p], pv.s[p]
                if not math.isfinite(u) or abs(u) < (xs.width_left(st) if u >= 0.0 else xs.width_right(st)) - 1e-6:
                    return False
            return True

        candidates: list[tuple[int, int]] = []  # (body, cell) of outer embankment cells
        for bi, body in enumerate(bodies):
            if body.layer == "L4" and body.part == "embankment" and owner(body) in roads:
                road = roads[owner(body)]
                candidates.extend((bi, ci) for ci, cell in enumerate(body.cells) if outer(road, cell))
        if not candidates:
            return bodies
        geoms, refs = [], []
        for bi, body in enumerate(bodies):
            for ci, cell in enumerate(body.cells):
                geoms.append(Polygon([(pv.x[p], pv.y[p]) for p in cell.loop]))
                refs.append((bi, ci))
        index = {ref: k for k, ref in enumerate(refs)}
        tree = shapely.STRtree(geoms)
        drop: dict[int, set[int]] = {}
        for bi, ci in candidates:
            mine = owner(bodies[bi])
            g = geoms[index[(bi, ci)]]
            lo, hi = zrange(bodies[bi], bodies[bi].cells[ci])
            for k in tree.query(g):
                bj, cj = refs[int(k)]
                if owner(bodies[bj]) == mine or g.intersection(geoms[int(k)]).area <= EPS_OVERLAP_M2:
                    continue
                lo2, hi2 = zrange(bodies[bj], bodies[bj].cells[cj])
                if lo < hi2 and lo2 < hi:
                    drop.setdefault(bi, set()).add(ci)
                    break
        return [
            dataclasses.replace(body, cells=[c for ci, c in enumerate(body.cells) if ci not in drop[bi]]) if bi in drop else body
            for bi, body in enumerate(bodies)
        ]

    # ------------------------------------------------------------ one area

    def _seam_row(self, jn: JunctionArea, seam) -> list[int]:
        pv = self.pv
        if seam.road_id not in self.ranges:
            raise ValueError(f"junction {jn.id}: road {seam.road_id} was not swept")
        n0, n1 = self.ranges[seam.road_id]
        s_arr = np.asarray(pv.s[n0:n1])
        near = np.flatnonzero(np.abs(s_arr - seam.s) <= STATION_MERGE_M * 1.01)
        if not len(near):
            raise ValueError(f"junction {jn.id}: no sweep station at {seam.road_id}@{seam.s:.3f}")
        s_row = s_arr[near[np.argmin(np.abs(s_arr[near] - seam.s))]]
        row = [n0 + int(i) for i in near if s_arr[i] == s_row and seam.u0 - ROW_TOL_M <= pv.u[n0 + int(i)] <= seam.u1 + ROW_TOL_M]
        row.sort(key=lambda p: pv.u[p])
        if len(row) >= 2 and abs(pv.u[row[0]] - seam.u0) <= 1e-5 and abs(pv.u[row[-1]] - seam.u1) <= 1e-5:
            return row
        # A skewed station line (a deck end or its ramp, D-073): the grid row through the centre-line vertex.
        centre = [n0 + int(i) for i in near if pv.u[n0 + int(i)] == 0.0 and pv.road[n0 + int(i)] == seam.road_id]
        if centre:
            k = pv.row[min(centre, key=lambda p: abs(pv.s[p] - seam.s))]
            line = sorted((p for p in range(n0, n1) if pv.row[p] == k and pv.road[p] == seam.road_id), key=lambda p: pv.u[p])
            right = [p for p in line if "wr" in pv.keys[p]]
            left = [p for p in line if "wl" in pv.keys[p]]
            if right and left:
                lo, hi = pv.u[right[0]], pv.u[left[-1]]
                row = [p for p in line if lo - ROW_TOL_M <= pv.u[p] <= hi + ROW_TOL_M]
                if len(row) >= 2:
                    return row
        raise ValueError(f"junction {jn.id}: sweep row at {seam.road_id}@{seam.s:.3f} does not span the seam")

    def _outline(self, jn: JunctionArea) -> tuple[list[int], list[list[int]], dict[int, tuple[float, float]]]:
        """Outline and hole vertex ids, with the road grid's own rows at the seams and their (top, underside)."""
        pv = self.pv
        roads = {r.id: r for r in self.model.roads}
        ring_xy = list(jn.outline)
        n = len(ring_xy)
        refs: list[int | tuple[float, float]] = list(ring_xy)
        exact: dict[int, tuple[float, float]] = {}  # seam vertex -> (top, underside) of the road there
        replaced: list[tuple[int, int, list[int], object]] = []
        for seam in jn.seams:
            row = self._seam_row(jn, seam)
            road = roads[seam.road_id]
            for p in row:
                exact[p] = (road.surface_z(pv.s[p], pv.u[p]), pavement_bottom(road, pv.s[p], pv.u[p]))
            for st in self.model.structures:
                if st.carried_road == seam.road_id and min(abs(st.s_start - seam.s), abs(st.s_end - seam.s)) <= 1e-6:
                    # Beside a deck end the fill reaches at least down to the soffit, onto the abutment top.
                    for p in row:
                        self._floor[p] = road.surface_z(pv.s[p], pv.u[p]) - st.deck.depth_m
            # The outline carries the straight row; a skewed grid row replaces it (D-073).
            ends = [road.alignment.offset_point(seam.s, seam.u0), road.alignment.offset_point(seam.s, seam.u1)]
            ia, ib = (_nearest(ring_xy, e) for e in ends)
            if ia is None or ib is None:  # an outline that already follows the skewed row (a deck end)
                ia, ib = (_nearest(ring_xy, (pv.x[p], pv.y[p]), SKEW_SNAP_M) for p in (row[0], row[-1]))
            if ia is None or ib is None:
                raise ValueError(f"junction {jn.id}: seam {seam.road_id}@{seam.s:.3f} is not on the outline")
            if (ib - ia) % n <= (ia - ib) % n:
                replaced.append((ia, ib, row, seam))
            else:
                replaced.append((ib, ia, row[::-1], seam))
        drop: set[int] = set()
        insert: dict[int, list[int]] = {}
        for a, b, row, seam in replaced:
            k = a
            while True:
                drop.add(k)
                if k == b:
                    break
                k = (k + 1) % n
            insert[a] = row
            road = roads[seam.road_id]
            for end, step in ((row[0], -1), (row[-1], 1)):
                k = (a if step < 0 else b) + step
                s_end = pv.s[end]
                lo, hi = sorted((seam.s, s_end))
                while (k % n) not in drop:
                    x, y = ring_xy[k % n]
                    # Outline vertices within the snap distance of a row end are that end (FR-SV-5);
                    # those between the straight and the skewed row lie on the road's kept side (D-073).
                    near_end = math.hypot(x - pv.x[end], y - pv.y[end]) <= OUTLINE_SNAP_M
                    if not near_end and hi - lo > OUTLINE_SNAP_M:
                        sv = road.alignment.project(x, y)[0]
                        near_end = lo + 1e-6 < sv < hi - 1e-6
                    if not near_end:
                        break
                    drop.add(k % n)
                    k += step
        ring: list[int] = []
        for k in range(n):
            if k in insert:
                ring.extend(insert[k])
            elif k not in drop:
                x, y = refs[k]  # type: ignore[misc]
                ring.append(self._vertex(float(x), float(y)))
        holes = [[self._vertex(float(x), float(y)) for x, y in h] for h in jn.holes]
        return ring, holes, exact

    def _junction(
        self,
        jn: JunctionArea,
        ring: list[int],
        holes: list[list[int]],
        exact: dict[int, tuple[float, float]],
        sampler: TerrainSampler | None,
        realtime: bool = False,
    ) -> list[Body]:
        pv = self.pv
        embed = REALTIME_EMBED_M if realtime else 0.0
        tris = _triangulate(pv, ring, holes, self._steiner(jn, ring, holes))

        def top(p: int) -> float:
            if p in exact:
                return exact[p][0]
            return jn.lifted_z(pv.x[p], pv.y[p])

        def under(p: int) -> float:
            if p in exact:
                return exact[p][1]
            x, y = pv.x[p], pv.y[p]
            return jn.lifted_z(x, y) - jn.thickness_at(x, y)

        cells = [Cell([a, b, c], [(a, b, c)]) for a, b, c in tris]
        src = jn.prov.source
        def keel(p: int) -> float:
            return under(p) - embed

        slab = Body(f"{jn.id}:pavement", jn.id, "L6", "pavement", MAT_PAVEMENT, src, cells, keel, top, ground=False, inset_m=INSET_ROAD_M)
        if sampler is None:
            return [slab]
        verts = {p for c in cells for p in c.loop}
        ground = {p: sampler(pv.x[p], pv.y[p]) for p in verts}
        if all(ground[p] >= under(p) - MIN_FILL_M for p in verts):
            slab.ground = None
            return [slab]

        def fill_bottom(p: int) -> float:
            return min(ground[p], under(p) - MIN_FILL_M, self._floor.get(p, math.inf)) - embed

        def fill_top(p: int) -> float:
            # Real-time walls are cut by the terrain: where it rises above the underside the fill
            # rises with it (still inside the slab), so the cut never crosses the fill top.
            if not realtime:
                return under(p)
            return max(under(p), min(ground[p] + RT_GROUND_CLEAR_M, top(p) - RT_GROUND_CLEAR_M))

        fill = Body(f"{jn.id}:embankment", jn.id, "L6", "embankment", MAT_SOIL, src, cells, fill_bottom, fill_top, ground=True, inset_m=INSET_SUPPORT_M)
        return [slab, fill]

    def _vertex(self, x: float, y: float) -> int:
        p = self._xy.get((x, y))
        if p is None:
            p = self._xy[(x, y)] = self.pv.add(x, y)
        return p

    def _steiner(self, jn: JunctionArea, ring: list[int], holes: list[list[int]]) -> list[tuple[float, float]]:
        """Interior points: a square lattice plus the participants' reference lines, clear of the boundary."""
        pv = self.pv
        poly = Polygon([(pv.x[p], pv.y[p]) for p in ring], [[(pv.x[p], pv.y[p]) for p in h] for h in holes])
        inner = poly.buffer(-STEINER_CLEAR * STEINER_M)
        if inner.is_empty:
            return []
        shapely.prepare(inner)
        cand: list[tuple[float, float]] = []
        roads = {r.id: r for r in self.model.roads}
        for rid in jn.road_ids:
            road = roads.get(rid)
            if road is None:
                continue
            for a, b in self.zones.get(rid, []):
                m = max(1, math.ceil((b - a) / STEINER_M))
                cand.extend(road.alignment.offset_point(a + (b - a) * j / m, 0.0) for j in range(m + 1))
        x0, y0, x1, y1 = poly.bounds
        gx = np.arange(math.floor(x0 / STEINER_M), math.ceil(x1 / STEINER_M) + 1) * STEINER_M
        gy = np.arange(math.floor(y0 / STEINER_M), math.ceil(y1 / STEINER_M) + 1) * STEINER_M
        cand.extend((float(x), float(y)) for y in gy for x in gx)
        out: list[tuple[float, float]] = []
        cell = STEINER_CLEAR * STEINER_M
        taken: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for x, y in cand:
            if not inner.contains(Point(x, y)):
                continue
            key = (math.floor(x / cell), math.floor(y / cell))
            near = [q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in taken.get((key[0] + dx, key[1] + dy), [])]
            if any(math.hypot(q[0] - x, q[1] - y) < cell for q in near):
                continue
            taken.setdefault(key, []).append((x, y))
            out.append((x, y))
        return out


def _nearest(ring, p, tol: float = OUTLINE_SNAP_M) -> int | None:
    best, bi = math.inf, None
    for i, q in enumerate(ring):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if d < best:
            best, bi = d, i
    return bi if best <= tol else None


# ---------------------------------------------------------------- triangulation


def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _incircle(a, b, c, d) -> float:
    """> 0 when d is inside the circle through counter-clockwise a, b, c."""
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    return (
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    )


def _triangulate(pv: PlanVertices, ring: list[int], holes: list[list[int]], steiner: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Constrained Delaunay triangles of the outline, refined by Steiner points with Lawson flips.

    Outline edges have one triangle each and are never flipped, so they stay as given;
    the result is counter-clockwise in plan.
    """
    lookup: dict[tuple[float, float], int] = {}
    for p in [*ring, *[q for h in holes for q in h]]:
        key = (pv.x[p], pv.y[p])
        if key in lookup and lookup[key] != p:
            raise ValueError("junction outline repeats a vertex")
        lookup[key] = p
    poly = Polygon([(pv.x[p], pv.y[p]) for p in ring], [[(pv.x[p], pv.y[p]) for p in h] for h in holes])
    if not poly.is_valid:
        raise ValueError(f"junction outline is not a valid polygon: {shapely.is_valid_reason(poly)}")
    tris: list[list[int] | None] = []
    for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)):
        a, b, c = (lookup[(float(x), float(y))] for x, y in shapely.get_coordinates(tri)[:3])
        tris.append([a, b, c] if pv.area2(a, b, c) > 0.0 else [a, c, b])
    edges: dict[tuple[int, int], int] = {}

    def link(t: int) -> None:
        a, b, c = tris[t]  # type: ignore[misc]
        edges[(a, b)] = t
        edges[(b, c)] = t
        edges[(c, a)] = t

    def unlink(t: int) -> None:
        a, b, c = tris[t]  # type: ignore[misc]
        for e in ((a, b), (b, c), (c, a)):
            if edges.get(e) == t:
                del edges[e]

    for t in range(len(tris)):
        link(t)

    def xy(p: int) -> tuple[float, float]:
        return pv.x[p], pv.y[p]

    def legalize(a: int, b: int, p: int) -> None:
        stack = [(a, b)]
        while stack:
            a, b = stack.pop()
            t1 = edges.get((a, b))
            t2 = edges.get((b, a))
            if t1 is None or t2 is None:
                continue
            tri1, tri2 = tris[t1], tris[t2]
            q = next(v for v in tri1 if v not in (a, b))  # type: ignore[union-attr]
            if q != p:
                continue
            d = next(v for v in tri2 if v not in (a, b))  # type: ignore[union-attr]
            if _incircle(xy(a), xy(b), xy(p), xy(d)) <= 1e-12:
                continue
            pa, pb, pp, pd = xy(a), xy(b), xy(p), xy(d)
            if _orient(*pp, *pd, *pa) * _orient(*pp, *pd, *pb) >= 0.0:
                continue  # non-convex quad: the flip would fold
            unlink(t1)
            unlink(t2)
            tris[t1] = [a, d, p]
            tris[t2] = [d, b, p]
            link(t1)
            link(t2)
            stack.extend([(a, d), (d, b)])

    for x, y in steiner:
        found = None
        for t, tri in enumerate(tris):
            if tri is None:
                continue
            a, b, c = tri
            o1 = _orient(pv.x[a], pv.y[a], pv.x[b], pv.y[b], x, y)
            o2 = _orient(pv.x[b], pv.y[b], pv.x[c], pv.y[c], x, y)
            o3 = _orient(pv.x[c], pv.y[c], pv.x[a], pv.y[a], x, y)
            if min(o1, o2, o3) > 1e-9:
                found = t
                break
        if found is None:
            continue  # on an edge or outside: skip the point
        a, b, c = tris[found]  # type: ignore[misc]
        p = pv.add(x, y)
        unlink(found)
        tris[found] = [a, b, p]
        link(found)
        tris.append([b, c, p])
        link(len(tris) - 1)
        tris.append([c, a, p])
        link(len(tris) - 1)
        legalize(a, b, p)
        legalize(b, c, p)
        legalize(c, a, p)

    out = [tuple(t) for t in tris if t is not None]
    for a, b, c in out:
        if pv.area2(a, b, c) <= EPS_AREA:
            raise ValueError("junction triangulation produced a degenerate triangle")
    return out  # type: ignore[return-value]
