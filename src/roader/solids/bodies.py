"""Body definitions for one road: pavement, barriers, embankments, and the structures it carries.

Construction follows FR-SG-1 and FR-SG-4: every body is a closed prism over cells of
the road grid between two model surfaces (pavement top and underside, deck top and
soffit, parapet profile, embankment slope and terrain). Over a structure span the
pavement is replaced by the deck and no embankment is generated, so the void under
the deck stays open (G3, AC-H3).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import shapely
from shapely.geometry import Polygon

from roader.model.provenance import Source
from roader.model.schema import (
    Barrier,
    Flag,
    Road,
    RoadModel,
    SideSlope,
    Structure,
    Support,
)
from roader.solids.assemble import Body, boundary_edges, chain_ring
from roader.solids.grid import Cell, PlanVertices, RoadGrid, ULine
from roader.solids.stations import (
    PRIORITY_OTHER,
    PRIORITY_STRUCTURE,
    RAMP_FACTOR,
    RAMP_MIN_M,
    STATION_MERGE_M,
    STRUCTURE_TOL_M,
    SkewKnot,
    StationLine,
    end_limited,
    merge_breakpoints,
    skew_at,
    skew_knots,
    skew_offsets,
    stations,
    support_block,
    toe_samples,
    toe_skew_limit,
)
from roader.solids.terrain import TerrainSampler

MIN_FILL_M = 1e-3
FOLD_LIMIT = 0.1  # a u-line on the inside of a curve keeps 1 - k u at least this (D-142); twice the grid's FOLD_MARGIN
DECK_EDGE_MIN_M = 0.5  # a deck edge stays this far from its road's centre line (D-141)
# Real-time export profile (FR-SV-7, D-038): resting bodies embed into what carries
# them, and every outline is inset in plan by a per-class amount so faces that touch
# in the analytic profile end up at least 0.2 mm apart (coplanar means within 0.1 mm).
REALTIME_EMBED_M = 0.25
BARRIER_CLEAR_M = 0.025  # off a deck, a barrier stops this far above the pavement keel
INSET_SUPPORT_M = 2e-4  # embankments, abutments, pier caps, columns
INSET_ROAD_M = 4e-4  # pavement, deck
INSET_TOP_M = 6e-4  # barriers, parapets
MAX_REACH_M = 500.0
SHAPE_MIN_M = 1e-3
COLUMN_SIDES = 8

# FR-AS-6 semantic material slots.
MAT_PAVEMENT = "travel_lane_asphalt"
MAT_DECK = "deck_concrete"
MAT_STRUCTURAL = "structural_concrete"
MAT_STEEL = "steel"
MAT_SOIL = "soil_slope"

_TRUST = {Source.MEASURED: 0, Source.INVENTORY: 1, Source.VECTOR: 2, Source.RULE: 3}


def least_trusted(sources: list[Source]) -> Source:
    return max(sources, key=lambda src: _TRUST[Source(src)])


def road_source(road: Road) -> Source:
    """The road's own provenance when set; otherwise the least trusted parameter (D-027)."""
    if road.prov is not None:
        return road.prov.source
    xs = road.cross_section
    sources = [e.prov.source for e in road.alignment.elements]
    sources += [e.prov.source for e in road.profile.elements]
    for pw in (xs.width_left, xs.width_right, xs.slope_left, xs.slope_right, xs.pavement_thickness):
        sources += [k.prov.source for k in pw.knots]
    return least_trusted(sources)


def mesh_id(element_id: str, part: str, k: int, n: int) -> str:
    return f"{element_id}:{part}" if n == 1 else f"{element_id}:{part}:{k}"


def subtract(a: float, b: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(a, b)]
    for s0, s1 in spans:
        nxt = []
        for p0, p1 in pieces:
            if s1 <= p0 or s0 >= p1:
                nxt.append((p0, p1))
                continue
            if s0 > p0:
                nxt.append((p0, s0))
            if s1 < p1:
                nxt.append((s1, p1))
        pieces = nxt
    return [(p0, p1) for p0, p1 in pieces if p1 - p0 > STATION_MERGE_M]


def _const(value: float) -> Callable[[float], float]:
    return lambda _s: value


def pavement_bottom(road: Road, s: float, u: float) -> float:
    return road.surface_z(s, u) - road.cross_section.pavement_thickness(s)


def _barrier_offsets(b: Barrier) -> tuple[list[float], float]:
    """Lateral breaklines (inner base, inner top, outer top, outer base) and the slope width."""
    d = 0.5 * (b.base_width_m - b.top_width_m)
    o = b.offset_m
    if d > SHAPE_MIN_M:
        return [o, o + d, o + d + b.top_width_m, o + b.base_width_m], d
    return [o, o + b.base_width_m], 0.0


def _barrier_shape(b: Barrier) -> Callable[[float], float]:
    """Height fraction of the barrier profile at |u|: a trapezoid on the surface below."""
    offs, d = _barrier_offsets(b)
    if d == 0.0:
        return lambda a: 1.0
    a0, a1, a2, a3 = offs

    def shape(a: float) -> float:
        if a <= a0 or a >= a3:
            return 0.0
        if a < a1:
            return (a - a0) / d
        if a <= a2:
            return 1.0
        return (a3 - a) / d

    return shape


def _toe(road: Road, sign: float, ratio: float, sampler: TerrainSampler, ss_id: str) -> Callable[[float], float]:
    xs = road.cross_section
    width = xs.width_left if sign > 0 else xs.width_right
    step = max(0.25, 0.5 * sampler.patch.cell_m)

    def toe(s: float) -> float:
        w = width(s)
        z_edge = pavement_bottom(road, s, sign * w)
        # Same arithmetic as Alignment.offset_point, with the station evaluated once.
        px, py, heading, _k = road.alignment.evaluate(s)
        nx_, ny_ = -math.sin(heading), math.cos(heading)

        def g(d: float) -> float:
            u = sign * (w + d)
            return z_edge - d / ratio - sampler(px + u * nx_, py + u * ny_)

        if g(0.0) <= MIN_FILL_M:
            raise ValueError(f"FR-SG: embankment {ss_id}: pavement underside is not above terrain at s={s:.2f}")
        lo, hi = 0.0, step
        while g(hi) > 0.0:
            lo, hi = hi, hi + step
            if hi > MAX_REACH_M:
                raise ValueError(f"FR-SG: embankment {ss_id}: no toe within {MAX_REACH_M} m at s={s:.2f}")
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return sign * (w + 0.5 * (lo + hi))

    return toe


def _fill_top(road: Road, sign: float, ratio: float) -> Callable[[float, float], float]:
    xs = road.cross_section
    width = xs.width_left if sign > 0 else xs.width_right

    def top(s: float, u: float) -> float:
        w = width(s)
        a = sign * u
        if a <= w:
            return pavement_bottom(road, s, u)
        return pavement_bottom(road, s, sign * w) - (a - w) / ratio

    return top


def _fold_limited(road: Road, lines: list[ULine], limits: list[Flag] | None) -> list[ULine]:
    """u-lines kept short of the centre of curvature on the inside of a curve (FR-SG-2, D-142).

    A line (an embankment toe on a tight ramp) whose offset u would reach 1 - k u <
    FOLD_LIMIT is held at (1 - FOLD_LIMIT) / |k| there; each limited line is flagged once.
    Lines that cannot come near the centre are returned unchanged.
    """
    kmax = max((max(abs(e.k0), abs(e.k1)) for e in road.alignment.elements), default=0.0)
    if kmax <= 1e-12:
        return lines
    out = []
    for line in lines:
        probe = [line.s0 + (line.s1 - line.s0) * i / 20 for i in range(21)]
        if max(abs(line.func(s)) for s in probe) * kmax < 0.5 * (1.0 - FOLD_LIMIT):
            out.append(line)
            continue
        hits: list[float] = []

        def func(s: float, f=line.func, hits=hits) -> float:
            u = f(s)
            k = road.alignment.evaluate(min(max(s, 0.0), road.length))[3]
            if u * k > 0.0 and abs(u * k) > 1.0 - FOLD_LIMIT:
                hits.append(s)
                return math.copysign((1.0 - FOLD_LIMIT) / abs(k), u)
            return u

        out.append(ULine(line.key, func, line.s0, line.s1))
        if limits is not None and any(func(s) != line.func(s) for s in probe):
            element = line.key.split(":")[1] if ":" in line.key else road.id
            limits.append(Flag(
                "FR-SG-2:fold_limited", element, "FR-SG-2",
                f"u-line {line.key} on {road.id} held at 1 - k u = {FOLD_LIMIT} on the inside of the curve over s={line.s0:.1f}-{line.s1:.1f}",
            ))
    return out


def _skew_field(
    road: Road,
    structures: list[Structure],
    lines: list[ULine],
    knots: list[SkewKnot],
    sweep: list[float],
    umax: float,
    limits: list[Flag] | None,
) -> list[StationLine]:
    """Station lines with the skew field limited near steep embankment toes and the road's ends (D-161, D-118).

    Each structure whose skew field had to be limited by a toe is flagged once.
    """
    tmax = max((abs(k.t) for k in knots), default=0.0)
    windows = []
    for st in structures:
        pad = RAMP_FACTOR * tmax * umax + RAMP_MIN_M + tmax * umax
        windows.append((st.s_start - pad, st.s_end + pad, st.id))
    toes = [(line.func, line.s0, line.s1) for line in lines if line.key.startswith("toe:")]
    caps = toe_skew_limit(toe_samples(toes, [(a, b) for a, b, _ in windows], tmax), sweep, umax, tmax) if tmax > 0.0 else [math.inf] * len(sweep)
    out, hit = [], {}
    for s, cap in zip(sweep, caps, strict=True):
        t = skew_at(knots, s)
        if abs(t) > cap:
            for a, b, st_id in windows:
                if a <= s <= b:
                    lo = hit.get(st_id, (math.inf, 0.0))
                    hit[st_id] = (min(lo[0], cap), max(lo[1], abs(t)))
            t = math.copysign(cap, t)
        out.append(StationLine(s, end_limited(t, s, road.length, umax)))
    if limits is not None:
        for st_id, (cap, t) in hit.items():
            limits.append(Flag(
                "FR-SG-2:skew_limited", st_id, "FR-SG-2",
                f"|tan skew| on {road.id} limited from {t:.2f} to {cap:.2f} where embankment toes run near the skewed station lines",
            ))
    return out


def road_bodies(
    model: RoadModel,
    road: Road,
    sampler: TerrainSampler | None,
    tol: float,
    pv: PlanVertices,
    *,
    realtime: bool = False,
    limits: list[Flag] | None = None,
) -> list[Body]:
    """Bodies of one road. ``realtime`` switches to the real-time profile's levels and insets.

    ``limits`` collects a flag wherever the model's geometry had to be limited to be swept (D-141).
    """
    xs = road.cross_section
    embed = REALTIME_EMBED_M if realtime else 0.0
    length = road.length
    if length <= 0.0 or not road.profile.elements:
        raise ValueError(f"road {road.id}: empty alignment or profile")
    structures = sorted((st for st in model.structures if st.carried_road == road.id), key=lambda st: (st.s_start, st.id))
    for st in structures:
        if st.kind != "bridge":
            raise NotImplementedError(f"structure {st.id}: kind {st.kind!r} is not generated in milestone 1")
        # Stations are rounded on construction (D-056): an end on the road end may pass the summed lengths by a rounding step.
        if not -STATION_MERGE_M <= st.s_start < st.s_end <= length + STATION_MERGE_M:
            raise ValueError(f"structure {st.id}: span outside road {road.id}")
    spans = [(st.s_start, st.s_end) for st in structures]
    # FR-CR-3: where this road rides another road's deck, that deck is its surface.
    carried = spans + [(sh.s_start, sh.s_end) for st in model.structures for sh in st.shared if sh.road_id == road.id]

    def on_structure(s: float) -> bool:
        return any(a - STATION_MERGE_M <= s <= b + STATION_MERGE_M for a, b in spans)

    lines = [
        ULine("c", _const(0.0), 0.0, length),
        ULine("wl", xs.width_left, 0.0, length),
        ULine("wr", lambda s: -xs.width_right(s), 0.0, length),
    ]
    bps: list[tuple[float, int]] = []
    for e in road.alignment.elements:
        bps += [(e.s0, PRIORITY_OTHER), (e.s0 + e.length, PRIORITY_OTHER)]
    for e in road.profile.elements:
        bps += [(e.s0, PRIORITY_OTHER), (e.s0 + e.length, PRIORITY_OTHER)]
    for pw in (xs.width_left, xs.width_right, xs.slope_left, xs.slope_right, xs.pavement_thickness):
        bps += [(s, PRIORITY_OTHER) for s in pw.breakpoints()]
    for a, b in carried[len(spans) :]:
        bps += [(a, PRIORITY_STRUCTURE), (b, PRIORITY_STRUCTURE)]

    # Barriers: rendered where they stand on the deck or on the pavement.
    barriers: list[tuple[Barrier, float, float]] = []
    for b in xs.barriers:
        s0, s1 = max(b.s0, 0.0), min(b.s1, length)
        if s1 - s0 <= STATION_MERGE_M:
            continue
        width = xs.width_left if b.side == "left" else xs.width_right
        samples = [s0, s1] + [k.s for k in width.knots if s0 < k.s < s1]
        if all(on_structure(s) or b.offset_m + b.base_width_m <= width(s) + 1e-9 for s in samples):
            barriers.append((b, s0, s1))
    for b, s0, s1 in barriers:
        sign = 1.0 if b.side == "left" else -1.0
        offs, _d = _barrier_offsets(b)
        for k, off in enumerate(offs):
            lines.append(ULine(f"b:{b.id}:{k}", _const(sign * off), s0, s1))

    for st in structures:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            width = xs.width_left if side == "left" else xs.width_right
            zone = max(
                (b.offset_m + b.base_width_m for b, s0, s1 in barriers if b.side == side and s0 < st.s_end and s1 > st.s_start),
                default=0.0,
            )
            overhang = st.deck.overhang(side)
            # D-141: a twin joint's negative overhang (FR-CR-3) never takes the edge within DECK_EDGE_MIN_M of the centre line.
            samples = [st.s_start, st.s_end] + [k.s for k in width.knots if st.s_start < k.s < st.s_end]
            least = min(max(width(s), zone) for s in samples) - DECK_EDGE_MIN_M
            if overhang < -least:
                if limits is not None:
                    limits.append(Flag(
                        "FR-SG-2:deck_edge_limited", st.id, "FR-CR-3",
                        f"{side} overhang {overhang:.2f} m would take the deck edge past its pavement; limited to {-least:.2f} m ({DECK_EDGE_MIN_M} m from the centre line)",
                    ))
                overhang = -least

            def edge(s: float, width=width, zone=zone, overhang=overhang, sign=sign) -> float:
                return sign * (max(width(s), zone) + overhang)

            lines.append(ULine(f"d{side[0]}:{st.id}", edge, st.s_start, st.s_end))
            for k, u in enumerate(skew_offsets(road, st, side, min(tol, STRUCTURE_TOL_M))):
                lines.append(ULine(f"k{side[0]}{k}:{st.id}", _const(u), st.s_start, st.s_end))
        bps += [(st.s_start, PRIORITY_STRUCTURE), (st.s_end, PRIORITY_STRUCTURE)]
        for sp in st.supports:
            a, b = support_block(st, sp)
            if not st.s_start - 1e-9 <= a < b <= st.s_end + 1e-9:
                raise ValueError(f"support {sp.id}: block [{a:.2f}, {b:.2f}] outside structure {st.id}")
            bps += [(a, PRIORITY_STRUCTURE), (b, PRIORITY_STRUCTURE), (sp.station, PRIORITY_STRUCTURE)]

    fills: list[tuple[SideSlope, float, list[tuple[float, float]]]] = []
    for ss in model.side_slopes:
        if ss.road_id != road.id or ss.kind == "none":
            continue
        if ss.kind != "fill":
            raise NotImplementedError(f"side slope {ss.id}: kind {ss.kind!r} is not generated in milestone 1")
        if sampler is None:
            raise ValueError(f"side slope {ss.id}: embankments need a terrain patch (L7)")
        if ss.ratio <= 0.0:
            raise ValueError(f"side slope {ss.id}: ratio must be positive")
        sign = 1.0 if ss.side == "left" else -1.0
        pieces = subtract(max(ss.s0, 0.0), min(ss.s1, length), carried)
        for k, (a, b) in enumerate(pieces):
            lines.append(ULine(f"toe:{ss.id}:{k}", _toe(road, sign, ss.ratio, sampler, ss.id), a, b))
        fills.append((ss, sign, pieces))

    lines = _fold_limited(road, lines, limits)
    for line in lines:
        bps += [(line.s0, PRIORITY_OTHER), (line.s1, PRIORITY_OTHER)]
    base = merge_breakpoints(bps, length)
    reach = {1.0: 0.0, -1.0: 0.0}  # farthest offset on each side
    for line in lines:
        for s in base:
            if line.s0 - STATION_MERGE_M <= s <= line.s1 + STATION_MERGE_M:
                u = line.func(min(max(s, line.s0), line.s1))
                side = 1.0 if u >= 0.0 else -1.0
                reach[side] = max(reach[side], abs(u))
    umax = 1.25 * max(reach.values()) + 1.0

    seen: dict[tuple[int, float], float] = {}

    def sides(a: float, b: float) -> tuple[float, float]:
        """Lateral reach on the left and the right of the lines active over [a, b]."""
        near = {1.0: 0.0, -1.0: 0.0}
        for n, line in enumerate(lines):
            if line.s1 < a - STATION_MERGE_M or line.s0 > b + STATION_MERGE_M:
                continue
            for s in (a, 0.5 * (a + b), b):
                key = (n, s)
                if key not in seen:
                    seen[key] = line.func(min(max(s, line.s0), line.s1))
                u = seen[key]
                side = 1.0 if u >= 0.0 else -1.0
                near[side] = max(near[side], abs(u))
        return 1.25 * near[1.0] + 1.0, 1.25 * near[-1.0] + 1.0
    knots = skew_knots(structures, umax)
    base = merge_breakpoints(bps + [(k.s, PRIORITY_OTHER) for k in knots if k.ramp], length)
    station_lines = _skew_field(road, structures, lines, knots, stations(road, base, tol, umax, knots, spans, sides), umax, limits)
    grid = RoadGrid(road, lines, station_lines, pv)

    def at(f: Callable[[float, float], float]) -> Callable[[int], float]:
        return lambda p: f(pv.s[p], pv.u[p])

    def top_fn(s: float, u: float) -> float:
        return road.surface_z(s, u)

    def bot_fn(s: float, u: float) -> float:
        return pavement_bottom(road, s, u)

    def keel_fn(s: float, u: float) -> float:
        return pavement_bottom(road, s, u) - embed

    src = road_source(road)
    bodies: list[Body] = []
    pieces = subtract(0.0, length, carried)
    for k, (a, b) in enumerate(pieces):
        bodies.append(
            Body(
                mesh_id(road.id, "pavement", k, len(pieces)), road.id, "L3", "pavement", MAT_PAVEMENT, src,
                grid.cells_between(a, b, "wr", "wl"), at(keel_fn), at(top_fn), ground=False, inset_m=INSET_ROAD_M,
            )
        )

    for ss, sign, ranges in fills:
        fill_top = _fill_top(road, sign, ss.ratio)
        for k, (a, b) in enumerate(ranges):
            toe = f"toe:{ss.id}:{k}"
            lo, hi = ("c", toe) if sign > 0 else (toe, "c")

            def bottom(p: int, fill_top=fill_top) -> float:
                # Terrain rising above the slope plane between two toe stations pinches the fill there, not through it.
                return min(sampler(pv.x[p], pv.y[p]), fill_top(pv.s[p], pv.u[p])) - embed  # type: ignore[misc]

            bodies.append(
                Body(
                    mesh_id(ss.id, "embankment", k, len(ranges)), ss.id, "L4", "embankment", MAT_SOIL, ss.prov.source,
                    grid.cells_between(a, b, lo, hi), bottom, at(fill_top), ground=True,
                    creases=frozenset({"wl" if sign > 0 else "wr"}), inset_m=INSET_SUPPORT_M,
                )
            )

    for b, s0, s1 in barriers:
        sign = 1.0 if b.side == "left" else -1.0
        n_lines = len(_barrier_offsets(b)[0])
        lo, hi = (f"b:{b.id}:0", f"b:{b.id}:{n_lines - 1}") if sign > 0 else (f"b:{b.id}:{n_lines - 1}", f"b:{b.id}:0")
        shape = _barrier_shape(b)

        def btop(s: float, u: float, shape=shape, h=b.height_m) -> float:
            return road.surface_z(s, u) + h * shape(abs(u))

        def bbottom(s: float, u: float) -> float:
            if not realtime:
                return road.surface_z(s, u)
            # Off a deck the pavement keel sits on the embankment top: go through both.
            depth = embed if on_structure(s) else xs.pavement_thickness(s) + embed - BARRIER_CLEAR_M
            return road.surface_z(s, u) - depth

        parapet = b.kind == "parapet" and on_structure(s0) and on_structure(s1)
        arris = frozenset({f"b:{b.id}:1", f"b:{b.id}:2"}) if n_lines == 4 else frozenset()
        bodies.append(
            Body(
                mesh_id(b.id, "parapet" if parapet else "barrier", 0, 1), b.id, "L5" if parapet else "L3",
                "parapet" if parapet else "barrier", MAT_STEEL if b.kind == "guardrail" else MAT_STRUCTURAL,
                b.prov.source, grid.cells_between(s0, s1, lo, hi), at(bbottom), at(btop), ground=False,
                creases=arris, inset_m=INSET_TOP_M,
            )
        )

    for st in structures:
        bodies.extend(_structure_bodies(st, road, grid, knots, pv, top_fn, embed))
    return bodies


def _structure_bodies(
    st: Structure,
    road: Road,
    grid: RoadGrid,
    knots: list,
    pv: PlanVertices,
    top_fn: Callable[[float, float], float],
    embed: float = 0.0,
) -> list[Body]:
    """Deck, abutments and piers; supports reach ``embed`` up into what they carry."""
    depth = st.deck.depth_m
    if depth <= 0.0:
        raise ValueError(f"structure {st.id}: deck depth must be positive")
    lo, hi = f"dr:{st.id}", f"dl:{st.id}"

    def soffit(p: int) -> float:
        return top_fn(pv.s[p], pv.u[p]) - depth

    def top(p: int) -> float:
        return top_fn(pv.s[p], pv.u[p])

    def support_top(p: int) -> float:
        return soffit(p) + embed

    bodies = [
        Body(
            f"{st.id}:deck", st.id, "L5", "deck", MAT_DECK, least_trusted([st.prov.source, st.deck.prov.source]),
            grid.cells_between(st.s_start, st.s_end, lo, hi), soffit, top, ground=False, inset_m=INSET_ROAD_M,
        )
    ]
    for sp in sorted(st.supports, key=lambda sp: (sp.station, sp.id)):
        a, b = support_block(st, sp)
        cells = grid.cells_between(a, b, lo, hi)
        if sp.kind == "abutment":
            bodies.append(
                Body(
                    f"{sp.id}:abutment", sp.id, "L5", "abutment", MAT_STRUCTURAL, sp.prov.source,
                    cells, _const_level(sp.footing_z), support_top, inset_m=INSET_SUPPORT_M,
                )
            )
            continue
        bodies.extend(_pier(sp, road, cells, grid, knots, pv, soffit, (lo, hi), embed))
    return bodies


def _const_level(z: float) -> Callable[[int], float]:
    return lambda _p: z


def _pier(
    sp: Support,
    road: Road,
    cells: list[Cell],
    grid: RoadGrid,
    knots: list,
    pv: PlanVertices,
    soffit: Callable[[int], float],
    keys: tuple[str, str],
    embed: float = 0.0,
) -> list[Body]:
    """Pier cap block under the soffit plus n_columns octagonal columns, aligned with the skew (FR-ST-4)."""
    if sp.n_columns < 1 or sp.column_width_m <= 0.0 or sp.cap_depth_m <= 0.0:
        raise ValueError(f"pier {sp.id}: needs columns, column width and cap depth")
    cap_verts = sorted({p for cell in cells for p in cell.loop})
    z_cap = min(soffit(p) for p in cap_verts) - sp.cap_depth_m
    if z_cap <= sp.footing_z:
        raise ValueError(f"pier {sp.id}: cap bottom below footing")

    ring = chain_ring(boundary_edges(cells))
    outline = Polygon([(pv.x[p], pv.y[p]) for p in ring])
    t = grid.sl[grid.index(sp.station)].t
    i = grid.index(sp.station)
    u_right, u_left = pv.u[grid.vid[(i, keys[0])]], pv.u[grid.vid[(i, keys[1])]]
    n = sp.n_columns
    radius = 0.5 * sp.column_width_m / math.cos(math.pi / COLUMN_SIDES)
    columns: list[list[int]] = []
    inner = outline.buffer(-0.01)
    for k in range(n):
        u = u_right + (k + 0.5) * (u_left - u_right) / n
        s = sp.station + t * u
        cx, cy = road.alignment.offset_point(s, u)
        ax, ay = road.alignment.offset_point(sp.station + t * (u + 0.5), u + 0.5)
        bx, by = road.alignment.offset_point(sp.station + t * (u - 0.5), u - 0.5)
        theta = math.atan2(ay - by, ax - bx)
        loop = []
        for j in range(COLUMN_SIDES):
            ang = theta + math.pi / COLUMN_SIDES + 2.0 * math.pi * j / COLUMN_SIDES
            loop.append(pv.add(cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
        if not inner.contains(Polygon([(pv.x[p], pv.y[p]) for p in loop])):
            raise ValueError(f"pier {sp.id}: column {k} is not inside the cap")
        columns.append(loop)

    holes = [[(pv.x[p], pv.y[p]) for p in loop] for loop in columns]
    lookup = {(pv.x[p], pv.y[p]): p for p in ring}
    for loop in columns:
        lookup.update({(pv.x[p], pv.y[p]): p for p in loop})
    bottom_tris = []
    for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(Polygon(outline.exterior.coords, holes))):
        a, b, c = (lookup[(float(x), float(y))] for x, y in shapely.get_coordinates(tri)[:3])
        bottom_tris.append((a, b, c) if pv.area2(a, b, c) > 0.0 else (a, c, b))
    fans = [[(loop[0], loop[j], loop[j + 1]) for j in range(1, COLUMN_SIDES - 1)] for loop in columns]
    for fan in fans:
        bottom_tris.extend(fan)

    bodies = [
        Body(
            f"{sp.id}:pier_cap", sp.id, "L5", "pier_cap", MAT_STRUCTURAL, sp.prov.source,
            cells, _const_level(z_cap), lambda p: soffit(p) + embed, ground=False, bottom_tris=bottom_tris,
            inset_m=INSET_SUPPORT_M,
        )
    ]
    for k, (loop, fan) in enumerate(zip(columns, fans, strict=True)):
        bodies.append(
            Body(
                mesh_id(sp.id, "pier_column", k, n), sp.id, "L5", "pier_column", MAT_STRUCTURAL, sp.prov.source,
                [Cell(loop, fan)], _const_level(sp.footing_z), _const_level(z_cap + embed), inset_m=INSET_SUPPORT_M,
            )
        )
    return bodies
