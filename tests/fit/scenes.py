"""Synthetic interchange pieces for the fit lane (FR-AF-3, FR-XS-4, loop ramps).

Every scene is a list of truth roads plus the lidar they would produce. Roads are
built from element specs; points are sampled on the true surfaces through the
true alignments, so curved roads, banking and twin carriageways are exact.
Where two pavements overlap in plan (a ramp leaving a mainline), only the first
road in the list is sampled, as one physical surface is.
"""

from __future__ import annotations

import math

import numpy as np

from roader.conflate.polyline import Polyline
from roader.io import VectorEdge, from_synthetic
from roader.model.geometry import (
    AlignElement,
    Alignment,
    Knot,
    Piecewise,
    Profile,
    VerticalElement,
    constant,
)
from roader.model.provenance import Provenance, Source
from roader.model.schema import CrossSection, Road, VectorRef
from roader.model.synthetic import SyntheticPoints

TRUTH = Provenance(Source.MEASURED, 1.0)


def alignment(x0: float, y0: float, h0: float, specs: list[tuple[str, float, float, float]]) -> Alignment:
    """Chain of (kind, length, k0, k1) elements from a start pose."""
    els, x, y, h, s0 = [], x0, y0, h0, 0.0
    for kind, length, k0, k1 in specs:
        el = AlignElement(kind, s0, length, x, y, h, k0, k1, TRUTH)
        els.append(el)
        x, y, h, _k = el.end_state()
        s0 += length
    return Alignment(els)


def profile(z0: float, g0: float, specs: list[tuple[float, float]]) -> Profile:
    """Chain of (length, end grade) elements: equal grades are tangents, different ones parabolic curves."""
    els, s0, z, g = [], 0.0, z0, g0
    for length, g1 in specs:
        el = VerticalElement(s0, length, z, g, g1, TRUTH)
        els.append(el)
        z, g, s0 = el.z(length), g1, s0 + length
    return Profile(els)


def piecewise(pairs: list[tuple[float, float]]) -> Piecewise:
    return Piecewise([Knot(s, v, TRUTH) for s, v in pairs])


def section(left: float, right: float, slope_left, slope_right) -> CrossSection:
    def pw(v):
        return v if isinstance(v, Piecewise) else constant(v, TRUTH)

    return CrossSection(pw(left), pw(right), pw(slope_left), pw(slope_right), constant(0.3, TRUTH))


def road(fid: str, kind: str, road_class: str, oneway: bool, al: Alignment, pr: Profile, xs: CrossSection, name: str = "") -> Road:
    return Road(
        id=f"rd_{fid}",
        name=name or fid,
        road_class=road_class,
        kind=kind,
        oneway=oneway,
        alignment=al,
        profile=pr,
        cross_section=xs,
        vector_refs=[VectorRef("synthetic", fid, {"highway": road_class})],
    )


def _surface_z(road: Road, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    zc = np.array([road.profile.z(float(v)) for v in s])
    xs = road.cross_section
    sl = np.array([xs.slope_left(float(v)) for v in s])
    sr = np.array([xs.slope_right(float(v)) for v in s])
    return zc + np.where(u >= 0.0, sl, sr) * u


def _half_widths(road: Road, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = road.cross_section
    return np.array([xs.width_left(float(v)) for v in s]), np.array([xs.width_right(float(v)) for v in s])


def points(
    roads: list[Road],
    *,
    spacing: float = 0.4,
    terrain_spacing: float = 0.8,
    terrain_drop: float = 1.5,
    seed: int = 11,
    extra=None,
    no_terrain=None,
) -> SyntheticPoints:
    """Pavement returns (3 cm noise) on every road, 1:4 fill slopes down to terrain, a few vehicles.

    ``extra``: optional (xyz, intensity, label) arrays appended before noise (medians, barriers).
    ``no_terrain``: optional callable (N, 2) xy -> bool mask of places with no terrain returns.
    """
    rng = np.random.default_rng(seed)
    lines = [Polyline.from_alignment(r.alignment) for r in roads]
    xs_, zs_, lab = [], [], []
    for n, (rd, line) in enumerate(zip(roads, lines, strict=True)):
        wl, wr = _half_widths(rd, np.array([0.0]))
        wmax = max(float(wl[0]), float(wr[0])) + 2.0
        s = np.arange(0.0, rd.length, spacing)
        u = np.arange(-wmax, wmax, spacing)
        ss, uu = (a.ravel() for a in np.meshgrid(s, u))
        ss = np.clip(ss + rng.uniform(-0.4, 0.4, ss.size) * spacing, 0.0, rd.length)
        wl, wr = _half_widths(rd, ss)
        keep = (uu <= wl) & (uu >= -wr)
        ss, uu = ss[keep], uu[keep]
        xy = line.offset_point(ss, uu)
        own = np.ones(len(xy), dtype=bool)
        for prev, pline in zip(roads[:n], lines[:n], strict=False):
            ps, pu, ok = pline.project(xy)
            pl, pr = _half_widths(prev, np.nan_to_num(ps))
            own &= ~(ok & (pu <= pl) & (pu >= -pr))
        xs_.append(xy[own])
        zs_.append(_surface_z(rd, ss[own], uu[own]))
        lab.append(np.full(int(own.sum()), "pavement"))
    pav_xy = np.vstack(xs_)
    lo, hi = pav_xy.min(axis=0) - 40.0, pav_xy.max(axis=0) + 40.0
    tx, ty = (a.ravel() for a in np.meshgrid(np.arange(lo[0], hi[0], terrain_spacing), np.arange(lo[1], hi[1], terrain_spacing)))
    txy = np.column_stack([tx, ty])
    base = min(float(np.min(z)) for z in zs_) - terrain_drop
    tz = np.full(len(txy), base)
    on_road = np.zeros(len(txy), dtype=bool)
    near = np.zeros(len(txy), dtype=bool)
    for rd, line in zip(roads, lines, strict=True):
        ts, tu, ok = line.project(txy, max_dist=50.0)
        near |= ok
        ts = np.nan_to_num(ts)
        tu = np.nan_to_num(tu)
        wl, wr = _half_widths(rd, ts)
        half = np.where(tu >= 0.0, wl, wr)
        beyond = np.abs(tu) - half
        on_road |= ok & (beyond <= 0.0)
        edge = _surface_z(rd, ts, np.where(tu >= 0.0, wl, -wr))
        slope = edge - np.maximum(beyond, 0.0) / 4.0
        tz = np.where(ok & (beyond > 0.0), np.maximum(tz, slope), tz)
    keep = near & ~on_road
    if no_terrain is not None:
        keep &= ~no_terrain(txy)
    xs_.append(txy[keep])
    zs_.append(tz[keep])
    lab.append(np.full(int(keep.sum()), "terrain"))
    xy = np.vstack(xs_)
    z = np.concatenate(zs_)
    labels = np.concatenate(lab)
    intensity = np.where(labels == "pavement", 30.0, 80.0)
    cls = np.full(len(z), 2)
    if extra is not None:
        exyz, eint, elab = extra
        xy = np.vstack([xy, exyz[:, :2]])
        z = np.concatenate([z, exyz[:, 2]])
        intensity = np.concatenate([intensity, eint])
        labels = np.concatenate([labels, elab])
        cls = np.concatenate([cls, np.where(elab == "barrier", 1, 2)])
    # Vehicles on the first road (FR-RS-3 must keep them out of every fit).
    first, fline = roads[0], lines[0]
    for s_car in np.linspace(0.15, 0.85, 4) * first.length:
        s = rng.uniform(s_car, s_car + 4.5, 120)
        u = rng.uniform(-2.5, -0.7, 120)
        cxy = fline.offset_point(s, u)
        xy = np.vstack([xy, cxy])
        z = np.concatenate([z, _surface_z(first, s, u) + rng.uniform(1.2, 1.5, 120)])
        intensity = np.concatenate([intensity, np.full(120, 60.0)])
        labels = np.concatenate([labels, np.full(120, "vehicle")])
        cls = np.concatenate([cls, np.full(120, 1)])
    z = z + rng.normal(0.0, 0.03, z.size)
    intensity = intensity + rng.normal(0.0, 5.0, z.size)
    n = z.size
    return SyntheticPoints(
        xyz=np.column_stack([xy, z]),
        classification=cls.astype(np.uint8),
        intensity=intensity,
        return_number=np.ones(n, dtype=np.uint8),
        point_source_id=np.ones(n, dtype=np.uint16),
        truth=labels,
    )


def vectors(roads: list[Road], *, seed: int = 3, bias: float = 1.0, noise: float = 0.2, step: float = 8.0, pin_ends: bool = True) -> list[VectorEdge]:
    """Input centerlines: truth with a slowly varying lateral bias and vertex noise.

    ``pin_ends`` keeps every end vertex exact, as OSM ways share their junction nodes.
    """
    rng = np.random.default_rng(seed)
    out = []
    for rd in roads:
        line = Polyline.from_alignment(rd.alignment)
        s = np.linspace(0.0, rd.length, max(2, math.ceil(rd.length / step) + 1))
        off = bias * np.sin(s / 90.0) + rng.normal(0.0, noise, len(s))
        if pin_ends:
            off *= np.clip(np.minimum(s, rd.length - s) / 40.0, 0.0, 1.0)
        xy = line.offset_point(s, off)
        ref = rd.vector_refs[0]
        tags = dict(ref.tags)
        if rd.oneway:
            tags["oneway"] = "yes"
        tags.setdefault("ref", "I 99" if rd.kind == "mainline" else "")
        if not tags["ref"]:
            del tags["ref"]
        out.append(VectorEdge("synthetic", ref.feature_id, xy, tags))
    return out


def reverse(rd: Road) -> Road:
    """The same road driven the other way: stations and offsets mirrored."""
    L = rd.length
    els, s0 = [], 0.0
    for e in reversed(rd.alignment.elements):
        x, y, h, _k = e.end_state()
        els.append(AlignElement(e.kind, s0, e.length, x, y, h + math.pi, -e.k1, -e.k0, e.prov))
        s0 += e.length
    pels, s0 = [], 0.0
    for e in reversed(rd.profile.elements):
        pels.append(VerticalElement(s0, e.length, e.z(e.length), -e.g1, -e.g0, e.prov))
        s0 += e.length

    def mirror(pw: Piecewise, sign: float) -> Piecewise:
        return Piecewise([Knot(L - k.s, sign * k.value, k.prov) for k in reversed(pw.knots)])

    xs = rd.cross_section
    out = CrossSection(
        mirror(xs.width_right, 1.0),
        mirror(xs.width_left, 1.0),
        mirror(xs.slope_right, -1.0),
        mirror(xs.slope_left, -1.0),
        xs.pavement_thickness,
    )
    return Road(rd.id, rd.name, rd.road_class, rd.kind, rd.oneway, Alignment(els), Profile(pels), out, rd.vector_refs)


# ---------------------------------------------------------------- scenes

MAIN_Z = 100.0


def mainline() -> Road:
    """Eastbound one-way carriageway, 2 lanes, crowned, 1300 m with a crest."""
    al = alignment(-400.0, 0.0, 0.0, [("line", 1300.0, 0.0, 0.0)])
    pr = profile(MAIN_Z, 0.012, [(300.0, 0.012), (200.0, -0.008), (800.0, -0.008)])
    return road("main", "mainline", "motorway", True, al, pr, section(5.5, 5.5, -0.02, 0.02), "I-99 eastbound")


LEAVE_R = 1500.0  # radius of the ramp's first arc, tangent to the parent at the node
SEPARATE_M = 180.0  # the ramp's far edge has left the parent's pavement by here (offset 10.8 m)


def _ramp_from(parent: Road, s_join: float, fid: str, turn: float, climb: float) -> Road:
    """A ramp leaving ``parent``'s centerline at ``s_join`` tangent to it, then curving to side ``turn``.

    Until it has left the parent's pavement it lies on the parent's surface: on an
    arc of LEAVE_R its offset is s^2 / 2R, so the parent's cross slope adds a
    parabola, and its profile is a vertical curve over [0, SEPARATE_M] (to a few
    mm). Then it climbs to ``climb`` and banks toward the inside of its curve.
    The parent must be on a tangent with constant cross slope over that stretch.
    """
    x, y, h, _k = parent.alignment.evaluate(s_join)
    side = 1.0 if turn > 0 else -1.0
    k0, k = side / LEAVE_R, turn / 200.0
    al = alignment(
        x, y, h,
        [("arc", SEPARATE_M, k0, k0), ("spiral", 120.0, k0, k), ("arc", 120.0, k, k), ("spiral", 60.0, k, 0.0), ("line", 60.0, 0.0, 0.0)],
    )
    pxs = parent.cross_section
    near_side = pxs.slope_left(s_join) if side > 0 else pxs.slope_right(s_join)
    g0 = parent.profile.grade(s_join)
    g1 = g0 + near_side * side * SEPARATE_M / LEAVE_R
    pr = profile(parent.profile.z(s_join), g0, [(SEPARATE_M, g1), (100.0, climb), (260.0, climb)])
    bank = -0.05 if turn > 0 else 0.05
    crossing = 90.0  # the parent's crown has left the ramp's inner half by here
    left0, right0 = pxs.slope_left(s_join), pxs.slope_right(s_join)
    sl = [(0.0, left0), (crossing, near_side)] if side < 0 else [(0.0, left0)]
    sr = [(0.0, right0), (crossing, near_side)] if side > 0 else [(0.0, right0)]
    tail = [(SEPARATE_M, near_side), (SEPARATE_M + 80.0, bank), (460.0, bank), (510.0, None)]
    sl = piecewise(sl + [(a, -0.02 if b is None else b) for a, b in tail])
    sr = piecewise(sr + [(a, 0.02 if b is None else b) for a, b in tail])
    return road(fid, "ramp", "motorway_link", True, al, pr, section(3.0, 3.0, sl, sr))


def diverge_merge_scene() -> list[Road]:
    """Mainline, an exit ramp leaving it to the right, and an entry ramp joining it from the right."""
    m = mainline()
    exit_ramp = _ramp_from(m, EXIT_S, "exit", -1.0, 0.035)
    back = reverse(m)
    entry = reverse(_ramp_from(back, back.length - ENTRY_S, "entry", 1.0, 0.03))
    return [m, exit_ramp, entry]


EXIT_S = 100.0
ENTRY_S = 1200.0


def loop_ramp(radius: float = 60.0, bank: float = 0.07) -> Road:
    """Cloverleaf loop: line, spiral, 3/4-turn arc (230 m at R = 60), spiral, line, turning left, banked down to the inside.

    The legs end where they would meet their mainlines, about 30 m apart.
    """
    k = 1.0 / radius
    arc = 230.0 * radius / 60.0
    al = alignment(0.0, 0.0, 0.0, [("line", 70.0, 0.0, 0.0), ("spiral", 50.0, 0.0, k), ("arc", arc, k, k), ("spiral", 50.0, k, 0.0), ("line", 10.0, 0.0, 0.0)])
    pr = profile(MAIN_Z, -0.03, [(120.0, -0.03), (160.0, 0.025), (arc - 30.0, 0.025)])
    e = 120.0 + arc
    sl = piecewise([(70.0, -0.02), (120.0, -bank), (e, -bank), (e + 50.0, -0.02)])
    sr = piecewise([(70.0, 0.02), (120.0, -bank), (e, -bank), (e + 50.0, 0.02)])
    return road("loop", "ramp", "motorway_link", True, al, pr, section(3.1, 3.1, sl, sr))


def twin_carriageways(median: float = 14.0, dz: float = 1.2) -> list[Road]:
    """Two one-way carriageways (two OSM ways) with a depressed median; the westbound one runs ``dz`` higher."""
    half = 0.5 * median + 5.5
    eb = road(
        "eb", "mainline", "motorway", True,
        alignment(-300.0, -half, 0.0, [("line", 600.0, 0.0, 0.0)]),
        profile(MAIN_Z, 0.01, [(600.0, 0.01)]),
        section(5.5, 5.5, 0.02, 0.02), "I-99 eastbound",
    )
    wb = road(
        "wb", "mainline", "motorway", True,
        alignment(300.0, half, math.pi, [("line", 600.0, 0.0, 0.0)]),
        profile(MAIN_Z + dz + 0.01 * 600.0, -0.01, [(600.0, -0.01)]),
        section(5.5, 5.5, 0.02, 0.02), "I-99 westbound",
    )
    return [eb, wb]


def median_band(eb: Road, wb: Road, kind: str, spacing: float = 0.3) -> tuple:
    """Extra returns between two straight twins along +x / -x: a barrier or a flush painted median.

    ``(xyz, intensity, label)`` for ``points(extra=...)`` and the no-terrain mask.
    """
    y0 = eb.alignment.evaluate(0.0)[1] + eb.cross_section.width_left(0.0)
    y1 = wb.alignment.evaluate(0.0)[1] - wb.cross_section.width_left(0.0)
    x0, x1 = -295.0, 295.0
    xs = np.arange(x0, x1, spacing)

    def edge_z(x):
        za = np.array([eb.surface_z(v + 300.0, eb.cross_section.width_left(0.0)) for v in x])
        zb = np.array([wb.surface_z(300.0 - v, wb.cross_section.width_left(0.0)) for v in x])
        return za, zb

    if kind == "barrier":
        yc = 0.5 * (y0 + y1)
        rows = []
        for dy, h in ((-0.28, 0.25), (-0.2, 0.5), (-0.12, 0.78), (0.0, 0.81), (0.12, 0.78), (0.2, 0.5), (0.28, 0.25)):
            za, zb = edge_z(xs)
            rows.append(np.column_stack([xs, np.full(xs.size, yc + dy), 0.5 * (za + zb) + h]))
        xyz = np.vstack(rows)
        return (xyz, np.full(len(xyz), 90.0), np.full(len(xyz), "barrier")), lambda xy: (xy[:, 1] > y0) & (xy[:, 1] < y1)
    ys = np.arange(y0 + 0.1, y1 - 0.05, spacing)
    gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
    za, zb = edge_z(gx)
    t = (gy - y0) / (y1 - y0)
    xyz = np.column_stack([gx, gy, za + t * (zb - za)])
    return (xyz, np.full(len(xyz), 60.0), np.full(len(xyz), "median")), lambda xy: (xy[:, 1] > y0) & (xy[:, 1] < y1)


def to_pointset(sp: SyntheticPoints):
    return from_synthetic(sp)
