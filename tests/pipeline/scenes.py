"""Parametric synthetic interchange catalogue: one fast scene per in-scope type (spec section 1, section 9).

A scene is a truth ``RoadModel`` (roads with alignment, profile and cross-section;
structures derived where a road stands ``DECK_ABOVE_M`` over the terrain; truth
crossings; links; optional terrain) plus what the pipeline sees of it:

* ``scene_points``: airborne lidar of the truth surfaces. Pavement returns on every
  road surface visible from above; nothing under a higher deck or pavement (a road
  under a deck gets no returns there, FR-OC-1); where two pavements overlap at one
  level only one physical surface is sampled. Deck tops are class 17, parapets
  class 1, terrain and fill slopes class 2, vehicles class 1, low noise class 7;
  two strips 2 cm apart, 3 cm noise, deterministic seed.
* ``scene_vectors``: OSM-like centerlines with a slowly varying lateral bias and
  vertex noise, pinned where ways share a node, split into bridge ways
  (``bridge=yes``, ``layer``) over each structure; tags highway, oneway, lanes,
  ref / name as OSM carries them.
* ``crossing_reference``: the hand-labelled AC-H2 list (D-044 entries
  ``{upper_name, lower_name, x, y}``) in the names the vector graph gives roads.

Scene builders (keyword parameters, defaults give the catalogue scene):

* ``simple_overpass(skew, rise, length)``: two-lane street on a crest bridge over twin carriageways.
* ``diamond(skew, spread, rise)``: street on a crest bridge over the freeway; four ramps meet it
  at grade on its approach embankments (terminals; exit and entry ways share the terminal node).
* ``spui(skew, deck_half, rise)``: street over the freeway on one wide short deck; the four
  ramps climb onto it and meet it at grade near the deck.
* ``parclo(skew, radius, spread)``: street over the freeway; diamond ramps on one side, two
  270-degree loops (entry from the street, exit to it) on the other.
* ``cloverleaf(skew, radius)``: freeway over freeway, four 270-degree loops.
* ``trumpet(radius)``: freeway stem ending at the mainline: a loop and a semi-directional ramp
  side by side over the mainline (the semi round the outside of the loop), two outer ramps: one
  sharing the semi's first metres, one merging onto the stem past the loop's end.
* ``directional(skew, radius)``: freeway over freeway, two left-turn flyovers a level higher.
* ``cd_braided(spacing, angle)``: collector-distributor road beside the mainline and a braided
  exit / entry pair crossing over each other.
* ``aux_lane(spacing, gap)``: an auxiliary lane in from one interchange, ``spacing`` m alongside
  the mainline ``gap`` m off its pavement (0: abutting), out to the next; no shared node.
* ``cd_parallel(gap)``: a C-D road beside the mainline for the whole scene, never joining it.
* ``viaduct(skew, n_streets, spacing)``: twin carriageways on one viaduct over 3+ streets in a
  row; the sheet never rejoins the ground.
* ``twin_different_elevations(dz, skew)``: twin carriageways with independent profiles and a
  median slope, crossed by a street bridge.
* ``stack(levels, skew)``: ``levels`` levels at one point (ground twins plus ``levels - 1``
  straight ramps on crest viaducts, profiles from ``tests.pipeline.stack_scene``).

Every ramp is built by ``ramp``: tangent legs on its parent / target surfaces (exact),
free lines solved so it lands on its target, its own profile between.

``SCENES`` maps type name to builder; ``PARAM_RANGES`` / ``fuzz_params`` give the fuzz ranges;
``run_scene`` takes a scene through the pipeline, solids and gates.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import shapely

from roader.conflate.polyline import Polyline
from roader.io import VectorEdge
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
from roader.model.schema import (
    Barrier,
    Crossing,
    CrossSection,
    DeckSpec,
    Frame,
    LaneGroup,
    LidarCollection,
    Road,
    RoadModel,
    Structure,
    Support,
    TerrainPatch,
    UnitRecord,
    VectorRef,
    VectorSourceRecord,
)
from roader.model.serialize import stable_id
from roader.model.synthetic import SyntheticPoints
from tests.fit.scenes import reverse

TRUTH = Provenance(Source.MEASURED, 1.0)
RULE = Provenance(Source.RULE, 0.5, rule_id="synthetic")

TERRAIN_Z = 99.0
GRADE_Z = 100.0
MEDIAN_HALF = 9.0  # twin reference lines this far either side of the freeway axis
FREEWAY_HALF = 5.5
STREET_HALF = 5.0
RAMP_HALF = 3.0
LEAVE_R = 400.0  # radius of a ramp's tangent leg where it leaves or joins its parent
SEP = 60.0  # length of that leg; the ramp's centre is SEP^2 / 2R = 4.5 m off the parent at its end
THETA = SEP / LEAVE_R
LOOP_LEG = (200.0, 30.0)  # (radius, length) of the tangent legs of a loop ramp
CURVE = 60.0  # vertical curves inside a ramp's own stretch
PEAK_CURVE = 40.0  # vertical curves of a ramp climbing to a flyover peak
MAX_GRADE = 0.08
DECK_ABOVE_M = 6.0  # the truth deck is where the road centre stands this far over the terrain
DECK_DEPTH = 1.3
OVERHANG = 0.9
PARAPET_M = 1.07
SAME_SURFACE_M = 1.0  # overlapping pavements within this height are one physical surface
FRAME = Frame(proj="+proj=tmerc +lat_0=42 +lon_0=-72 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", anchor_lon=-72.0, anchor_lat=42.0)


@dataclass
class Scene:
    name: str
    model: RoadModel
    params: dict = field(default_factory=dict)

    def road(self, fid: str) -> Road:
        return next(r for r in self.model.roads if r.vector_refs[0].feature_id == fid)


# ---------------------------------------------------------------- vectorised truth evaluation


def _pw(piece: Piecewise, s: np.ndarray) -> np.ndarray:
    return np.interp(s, [k.s for k in piece.knots], [k.value for k in piece.knots])


def _profile_eval(profile: Profile, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(z, grade) as ``Profile.z`` / ``Profile.grade``, vectorised."""
    s = np.asarray(s, dtype=np.float64)
    els = profile.elements
    ends = np.array([e.s0 + e.length for e in els])
    idx = np.minimum(np.searchsorted(ends, s, side="left"), len(els) - 1)
    z = np.empty(s.shape)
    g = np.empty(s.shape)
    for i, e in enumerate(els):
        m = idx == i
        if not m.any():
            continue
        t = s[m] - e.s0
        inside = np.clip(t, 0.0, e.length)
        gi = e.g0 + (e.g1 - e.g0) * inside / e.length
        z[m] = e.z0 + e.g0 * inside + 0.5 * (e.g1 - e.g0) * inside * inside / e.length + gi * (t - inside)
        g[m] = gi
    return z, g


def surface(road: Road, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    xs = road.cross_section
    z, _g = _profile_eval(road.profile, s)
    return z + np.where(u >= 0.0, _pw(xs.slope_left, s), _pw(xs.slope_right, s)) * u


class _Geo:
    """Plan geometry of one truth road for sampling: line, widths, deck ranges."""

    def __init__(self, road: Road, decks: list[tuple[float, float]]):
        self.road = road
        self.line = Polyline.from_alignment(road.alignment, step=1.0)
        self.decks = decks
        xs = road.cross_section
        self.reach = max(max(k.value for k in xs.width_left.knots), max(k.value for k in xs.width_right.knots)) + OVERHANG + 1.0

    def on_deck(self, s: np.ndarray) -> np.ndarray:
        out = np.zeros(len(s), dtype=bool)
        for a, b in self.decks:
            out |= (s >= a) & (s <= b)
        return out

    def footprint(self, xy: np.ndarray, *, decks: bool = True, inset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(inside, top z) of the pavement (plus the deck overhang over decks) at plan points."""
        s, u, ok = self.line.project(xy, max_dist=self.reach)
        inside = np.zeros(len(xy), dtype=bool)
        z = np.full(len(xy), -np.inf)
        if not ok.any():
            return inside, z
        i = np.flatnonzero(ok)
        si, ui = s[i], u[i]
        xs = self.road.cross_section
        extra = np.where(self.on_deck(si), OVERHANG, 0.0) if decks else 0.0
        wl = _pw(xs.width_left, si) + extra - inset
        wr = _pw(xs.width_right, si) + extra - inset
        m = (ui <= wl) & (ui >= -wr)
        inside[i[m]] = True
        z[i[m]] = surface(self.road, si[m], np.clip(ui[m], -_pw(xs.width_right, si[m]), _pw(xs.width_left, si[m])))
        return inside, z


# ---------------------------------------------------------------- road construction


@dataclass
class El:
    """A plan element in a ramp chain: a line or an arc of curvature k; ``free`` lines are solved."""

    kind: str
    length: float
    k: float = 0.0
    free: bool = False


def line(length: float = 0.0, free: bool = False) -> El:
    return El("line", length, 0.0, free)


def arc(turn: float, radius: float) -> El:
    """Arc turning by ``turn`` rad (positive left) on ``radius``."""
    return El("arc", abs(turn) * radius, math.copysign(1.0 / radius, turn))


def _chain(x: float, y: float, h: float, els: list[El], free: dict[int, float] | None = None) -> Alignment:
    out, s0 = [], 0.0
    for n, e in enumerate(els):
        length = (free or {}).get(n, e.length)
        if length <= 1e-9:
            continue
        el = AlignElement(e.kind, s0, length, x, y, h, e.k, e.k, TRUTH)
        out.append(el)
        x, y, h, _k = el.end_state()
        s0 += length
    return Alignment(out)


def _end(x: float, y: float, h: float, els: list[El], free: dict[int, float] | None = None) -> tuple[float, float, float]:
    al = _chain(x, y, h, els, free)
    if not al.elements:
        return x, y, h
    ex, ey, eh, _k = al.elements[-1].end_state()
    return ex, ey, eh


def _headings(h: float, els: list[El]) -> list[float]:
    out = []
    for e in els:
        out.append(h)
        h += e.k * e.length
    return out


def turn_to(h_from: float, h_to: float, sense: float) -> float:
    """Signed turn from h_from to h_to going in ``sense`` (+1 left, -1 right), in (0, 2 pi]."""
    d = (sense * (h_to - h_from)) % (2.0 * math.pi)
    return sense * (d if d > 1e-9 else 2.0 * math.pi)


def pvi_profile(pvis: list[tuple[float, float]], curves: list[float]) -> Profile:
    """Tangents through (station, z) PVIs with a symmetric parabolic curve of ``curves[i]`` at each interior PVI."""
    grades = [(z1 - z0) / (s1 - s0) for (s0, z0), (s1, z1) in itertools.pairwise(pvis)]
    els: list[VerticalElement] = []
    s, z, g = pvis[0][0], pvis[0][1], grades[0]

    def add(length: float, g0: float, g1: float) -> None:
        nonlocal s, z
        if length <= 1e-6:
            return
        el = VerticalElement(s, length, z, g0, g1, TRUTH)
        els.append(el)
        s, z = s + length, el.z(length)

    for i, c in enumerate(curves):
        sp = pvis[i + 1][0]
        add(sp - c / 2.0 - s, g, g)
        add(c, g, grades[i + 1])
        g = grades[i + 1]
    add(pvis[-1][0] - s, g, g)
    return Profile(els)


def crest_profile(length: float, centre: float, rise: float, half_span: float, crest: float, sag: float, z: float = GRADE_Z) -> Profile:
    """Flat, up at rise / half_span, crest at ``centre``, down, flat."""
    return pvi_profile([(0.0, z), (centre - half_span, z), (centre, z + rise), (centre + half_span, z), (length, z)], [sag, crest, sag])


def _section(half_l: float, half_r: float, lanes: int, sl=-0.02, sr=0.02) -> CrossSection:
    def pw(v):
        return v if isinstance(v, Piecewise) else constant(v, TRUTH)

    groups = [LaneGroup("right", 0.0, 1e9, lanes, 3.66, 1.2, 1.0, RULE)] if lanes == 1 else [
        LaneGroup("left", 0.0, 1e9, lanes // 2, 3.66, 1.2, 1.0, RULE),
        LaneGroup("right", 0.0, 1e9, lanes - lanes // 2, 3.66, 1.2, 1.0, RULE),
    ]
    return CrossSection(pw(half_l), pw(half_r), pw(sl), pw(sr), constant(0.3, RULE), lanes=groups)


def _road(fid: str, kind: str, cls: str, al: Alignment, pr: Profile, xs: CrossSection, *, oneway: bool, lanes: int, name: str = "", ref: str = "") -> Road:
    tags = {"highway": cls, "lanes": str(lanes)}
    if oneway:
        tags["oneway"] = "yes"
    if ref:
        tags["ref"] = ref
    if name:
        tags["name"] = name
    return Road(
        id=stable_id("rd", "scene", fid),
        name=name or ref or fid,
        road_class=cls,
        kind=kind,
        oneway=oneway,
        alignment=al,
        profile=pr,
        cross_section=xs,
        vector_refs=[VectorRef("synthetic", fid, tags)],
    )


def _straight(x0: float, y0: float, h: float, length: float) -> Alignment:
    return Alignment([AlignElement("line", 0.0, length, x0, y0, h, 0.0, 0.0, TRUTH)])


def twins(length: float, heading: float, ref: str, fids: tuple[str, str], profiles: tuple[Profile, Profile], centre=(0.0, 0.0), offset: float = MEDIAN_HALF) -> tuple[Road, Road]:
    """Twin one-way carriageways: the first along ``heading`` on the right of the axis, the second back."""
    d = np.array([math.cos(heading), math.sin(heading)])
    n = np.array([-d[1], d[0]])
    c = np.asarray(centre, dtype=float)
    a0 = c - offset * n - 0.5 * length * d
    b0 = c + offset * n + 0.5 * length * d
    out = []
    for fid, p0, h, pr in ((fids[0], a0, heading, profiles[0]), (fids[1], b0, heading + math.pi, profiles[1])):
        out.append(_road(fid, "mainline", "motorway", _straight(float(p0[0]), float(p0[1]), h, length), pr, _section(FREEWAY_HALF, FREEWAY_HALF, 2), oneway=True, lanes=2, ref=ref))
    return out[0], out[1]


def street(fid: str, name: str, length: float, heading: float, pr: Profile, centre=(0.0, 0.0), half: float | Piecewise = STREET_HALF) -> Road:
    d = np.array([math.cos(heading), math.sin(heading)])
    p0 = np.asarray(centre, dtype=float) - 0.5 * length * d
    return _road(fid, "crossing_street", "primary", _straight(float(p0[0]), float(p0[1]), heading, length), pr, _section(half, half, 2), oneway=False, lanes=2, name=name)


def _straight_line(road: Road) -> tuple[np.ndarray, np.ndarray]:
    x, y, h, _k = road.alignment.evaluate(0.0)
    if any(e.kind != "line" for e in road.alignment.elements):
        raise ValueError(f"{road.name}: solving against a curved road")
    return np.array([x, y]), np.array([math.cos(h), math.sin(h)])


def _near_slope(road: Road, s: float, side: float) -> float:
    xs = road.cross_section
    return xs.slope_left(s) if side > 0 else xs.slope_right(s)


def _diverge_leg(parent: Road, s0: float, side: float, leg: tuple[float, float]) -> list[tuple[float, float, float, float, float]]:
    """(t0, length, z0, g0, g1) pieces of a ramp leg leaving ``parent`` at ``s0`` tangent, on its surface (exact)."""
    radius, sep = leg
    c = _near_slope(parent, s0, side) * side / (2.0 * radius)
    cuts = sorted({0.0, sep, *(e.s0 - s0 for e in parent.profile.elements if 0.0 < e.s0 - s0 < sep)})
    out = []
    for t0, t1 in itertools.pairwise(cuts):
        z0 = parent.profile.z(s0 + t0) + c * t0 * t0
        out.append((t0, t1 - t0, z0, parent.profile.grade(s0 + t0) + 2 * c * t0, parent.profile.grade(s0 + t1) + 2 * c * t1))
    return out


def _merge_leg(target: Road, s_end: float, side: float, length: float, leg: tuple[float, float]) -> list[tuple[float, float, float, float, float]]:
    """The same for a leg ending tangent on ``target`` at ``s_end``; ramp stations end at ``length``."""
    radius, sep = leg
    c = _near_slope(target, s_end, side) * side / (2.0 * radius)
    cuts = sorted({0.0, sep, *(s_end - e.s0 for e in target.profile.elements if 0.0 < s_end - e.s0 < sep)}, reverse=True)
    out = []
    for tau0, tau1 in itertools.pairwise(cuts):  # tau decreasing along the ramp
        r0 = length - tau0
        z0 = target.profile.z(s_end - tau0) + c * tau0 * tau0
        out.append((r0, tau0 - tau1, z0, target.profile.grade(s_end - tau0) - 2 * c * tau0, target.profile.grade(s_end - tau1) - 2 * c * tau1))
    return out


def _middle(a: float, b: float, z1: float, g1: float, z2: float, g2: float, peak: float | None, grade: float, label: str, lead: float = 0.0) -> list[tuple[float, float, float]]:
    """(length, g0, g1) pieces from (a, z1, g1) to (b, z2, g2); with ``peak``, a flat top at that height after ``lead`` m level."""
    m = b - a
    if m <= 1e-6:
        return []
    if peak is not None and lead > 0.0:
        c = PEAK_CURVE
        head = [(c, g1, 0.0), (lead, 0.0, 0.0)]
        return head + _middle(a + c + lead, b, z1 + g1 * c / 2.0, 0.0, z2, g2, peak, grade, label)
    if peak is None:
        c = min(CURVE, m / 3.0)
        t = m - 2.0 * c
        gm = (z2 - z1 - g1 * c / 2.0 - g2 * c / 2.0) / (c + t)
        if abs(gm) > MAX_GRADE:
            raise ValueError(f"{label}: grade {gm:.3f} over {m:.0f} m")
        return [(c, g1, gm), (t, gm, gm), (c, gm, g2)]
    c, gg = PEAK_CURVE, grade
    t1 = (peak - z1 - (g1 + gg) * c / 2.0 - gg * c / 2.0) / gg
    t2 = (peak - gg * c / 2.0 + (g2 - gg) * c / 2.0 - z2) / gg
    flat = m - 4.0 * c - t1 - t2
    if min(t1, t2, flat) < 0.0:
        raise ValueError(f"{label}: peak {peak:.1f} not reachable in {m:.0f} m at grade {gg}")
    return [(c, g1, gg), (t1, gg, gg), (c, gg, 0.0), (flat, 0.0, 0.0), (c, 0.0, -gg), (t2, -gg, -gg), (c, -gg, g2)]


def ramp(
    fid: str,
    parent: Road,
    body: list[El],
    *,
    s: float | None = None,
    side: float = -1.0,
    start: str = "diverge",
    start_heading: float | None = None,
    target: Road | None = None,
    end: str = "merge",
    target_side: float = -1.0,
    point: tuple[float, float] | None = None,
    end_z: float = GRADE_Z,
    peak: float | None = None,
    grade: float = 0.05,
    lead: float = 0.0,
    half: float = RAMP_HALF,
    cls: str = "motorway_link",
    leg: tuple[float, float] = (LEAVE_R, SEP),
) -> Road:
    """A one-way ramp from ``parent`` to ``target`` (or to ``point`` / a free end).

    start: "diverge" (tangent leg leaving the parent's centre line to ``side``) or
    "terminal" (at the parent's centre line, heading ``start_heading`` or square to it).
    end: "merge" (tangent leg joining the target from ``target_side``), "terminal"
    (on the target's centre line at grade), "point" (ends at ``point``, z ``end_z``) or
    "free" (ends where the body ends, z ``end_z``). Unknowns: the start station when
    ``s`` is None (straight parent), and every ``free`` line, so the end lands on the
    straight target line or the point. The profile follows the parent and target
    surfaces exactly over the legs, with its own stretch between (crest to ``peak``).
    """
    leave_r, sep = leg
    els = ([El("arc", sep, side / leave_r)] if start == "diverge" else []) + list(body)
    if end == "merge":
        els.append(El("arc", sep, target_side / leave_r))
    unknowns: list[tuple[str, int]] = []
    s_ref = 0.5 * parent.length if s is None else s
    if s is None:
        unknowns.append(("start", -1))
    unknowns += [("line", n) for n, e in enumerate(els) if e.free]

    def pose(st: float) -> tuple[float, float, float]:
        x, y, h, _k = parent.alignment.evaluate(st)
        if start == "terminal":
            h = start_heading if start_heading is not None else h + side * math.pi / 2.0
        return x, y, h

    free: dict[int, float] = {n: 0.0 for kind, n in unknowns if kind == "line"}
    if unknowns:
        x0, y0, h0 = pose(s_ref)
        base = np.array(_end(x0, y0, h0, els, free)[:2])
        hs = _headings(h0, els)
        cols = []
        for kind, n in unknowns:
            if kind == "start":
                cols.append(_straight_line(parent)[1])
            else:
                cols.append(np.array([math.cos(hs[n]), math.sin(hs[n])]))
        a = np.column_stack(cols)
        if end == "point":
            rows, rhs = a, np.asarray(point, dtype=float) - base
        else:
            p0, d = _straight_line(target)
            nrm = np.array([-d[1], d[0]])
            rows, rhs = nrm[None, :] @ a, np.array([nrm @ (p0 - base)])
        sol, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        if not np.allclose(rows @ sol, rhs, atol=1e-6):
            raise ValueError(f"{fid}: no exact solution for its free lengths")
        for (kind, n), v in zip(unknowns, sol, strict=True):
            if kind == "start":
                s_ref += float(v)
            elif v < 0.0:
                raise ValueError(f"{fid}: free line {n} solved to {v:.1f} m")
            else:
                free[n] = float(v)
    if not (0.0 <= s_ref <= parent.length):
        raise ValueError(f"{fid}: start station {s_ref:.1f} off its parent")
    x0, y0, h0 = pose(s_ref)
    al = _chain(x0, y0, h0, els, free)
    length = al.length

    # Profile: legs on the parent / target surfaces, own stretch between.
    pieces: list[tuple[float, float, float, float, float]] = []
    if start == "diverge":
        pieces += _diverge_leg(parent, s_ref, side, leg)
        a_s = sep
        z1, g1 = _profile_eval(Profile([VerticalElement(*p, TRUTH) for p in pieces]), np.array([sep]))
        z1, g1 = float(z1[0]), float(g1[0])
    else:
        a_s, z1, g1 = 0.0, parent.profile.z(s_ref), 0.0
    tail: list[tuple[float, float, float, float, float]] = []
    ex, ey, _eh, _k = al.elements[-1].end_state()
    if end == "merge":
        s_end = target.alignment.project(ex, ey)[0]
        tail = _merge_leg(target, s_end, target_side, length, leg)
        b_s, z2, g2 = length - sep, tail[0][2], tail[0][3]
    elif end == "terminal":
        s_end = target.alignment.project(ex, ey)[0]
        b_s, z2, g2 = length, target.profile.z(s_end), 0.0
    else:
        b_s, z2, g2 = length, end_z, 0.0
    r, z = a_s, z1
    for seg, ga, gb in _middle(a_s, b_s, z1, g1, z2, g2, peak, grade, fid, lead):
        if seg <= 1e-6:
            continue
        pieces.append((r, seg, z, ga, gb))
        z = VerticalElement(r, seg, z, ga, gb, TRUTH).z(seg)
        r += seg
    pieces += tail
    pr = Profile([VerticalElement(round(p[0], 6), p[1], p[2], p[3], p[4], TRUTH) for p in pieces])

    # Cross slope: the parent's (target's) surface over the legs, a crown between.
    tx = min(math.sqrt(2.0 * leave_r * half), 0.7 * sep)
    kl: list[tuple[float, float]] = []
    kr: list[tuple[float, float]] = []
    if start == "diverge":
        near = _near_slope(parent, s_ref, side)
        xs = parent.cross_section
        kl += [(0.0, xs.slope_left(s_ref)), (tx, near), (sep, near), (sep + 40.0, -0.02)]
        kr += [(0.0, xs.slope_right(s_ref)), (tx, near), (sep, near), (sep + 40.0, 0.02)]
    if end == "merge":
        near = _near_slope(target, s_end, target_side)
        xs = target.cross_section
        kl += [(length - sep - 40.0, -0.02), (length - sep, near), (length - tx, near), (length, xs.slope_left(s_end))]
        kr += [(length - sep - 40.0, 0.02), (length - sep, near), (length - tx, near), (length, xs.slope_right(s_end))]
    for knots in (kl, kr):
        if any(b[0] - a[0] < 1.0 for a, b in itertools.pairwise(knots)):
            raise ValueError(f"{fid}: {length:.0f} m too short for its legs")
    sl = Piecewise([Knot(a, b, TRUTH) for a, b in kl]) if kl else -0.02
    sr = Piecewise([Knot(a, b, TRUTH) for a, b in kr]) if kr else 0.02
    for v in np.linspace(0.0, length, 200):
        if abs(pr.grade(float(v))) > MAX_GRADE + 1e-9:
            raise ValueError(f"{fid}: grade {pr.grade(float(v)):.3f} at s={v:.0f}")
    kind = "ramp" if cls.endswith("_link") else "mainline"
    return _road(fid, kind, cls, al, pr, _section(half, half, 1, sl, sr), oneway=True, lanes=1)


# ---------------------------------------------------------------- truth structures, crossings, model


def _ground(terrain: TerrainPatch | None) -> Callable[[np.ndarray], np.ndarray]:
    if terrain is None:
        return lambda xy: np.full(len(xy), TERRAIN_Z)
    z = np.array(terrain.z, dtype=float).reshape(terrain.ny, terrain.nx)

    def f(xy: np.ndarray) -> np.ndarray:
        fx = np.clip((xy[:, 0] - terrain.origin_x) / terrain.cell_m, 0.0, terrain.nx - 1.000001)
        fy = np.clip((xy[:, 1] - terrain.origin_y) / terrain.cell_m, 0.0, terrain.ny - 1.000001)
        ix, iy = fx.astype(int), fy.astype(int)
        ax, ay = fx - ix, fy - iy
        return (z[iy, ix] * (1 - ax) + z[iy, ix + 1] * ax) * (1 - ay) + (z[iy + 1, ix] * (1 - ax) + z[iy + 1, ix + 1] * ax) * ay

    return f


def _centre_xy(road: Road, s: np.ndarray) -> np.ndarray:
    return Polyline.from_alignment(road.alignment, step=1.0).point(s)


def _structures(roads: list[Road], ground) -> list[Structure]:
    out = []
    lines = {r.id: shapely.LineString(_centre_xy(r, np.linspace(0.0, r.length, max(2, int(r.length) + 1)))) for r in roads}
    for road in roads:
        s = np.arange(0.0, road.length, 0.5)
        z, _g = _profile_eval(road.profile, s)
        up = z - ground(_centre_xy(road, s)) >= DECK_ABOVE_M
        edges = np.flatnonzero(np.diff(np.concatenate([[0], up.astype(int), [0]])))
        for i0, i1 in zip(edges[::2], edges[1::2], strict=True):
            a, b = float(s[i0]), float(s[i1 - 1])
            if b - a < 10.0:
                continue
            sid = stable_id("st", road.id, round(a, 1))
            crossed = []
            for other in roads:
                if other is road:
                    continue
                for p in _points(lines[road.id].intersection(lines[other.id])):
                    su = road.alignment.project(p.x, p.y)[0]
                    so = other.alignment.project(p.x, p.y)[0]
                    if a <= su <= b and road.profile.z(su) - other.profile.z(so) > 3.0 and other.id not in crossed:
                        crossed.append(other.id)
            supports = [Support(stable_id("sp", sid, tag), st, "abutment", 0.0, 0, 0.0, 1.2, 1.5, TERRAIN_Z - 1.5, RULE) for tag, st in (("a1", a), ("a2", b))]
            out.append(Structure(sid, "bridge", road.id, a, b, "multi_girder", None, crossed, DeckSpec(DECK_DEPTH, 0.22, OVERHANG, RULE), supports, TRUTH))
            for side in ("left", "right"):
                piece = road.cross_section.width_left if side == "left" else road.cross_section.width_right
                off = max(piece(a), piece(b), piece(0.5 * (a + b)))
                road.cross_section.barriers.append(Barrier(stable_id("br", sid, side), side, a, b, off, PARAPET_M, 0.45, 0.25, "parapet", RULE))
    return out


def _points(geom) -> list:
    if geom.is_empty:
        return []
    if isinstance(geom, shapely.Point):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, shapely.Point)]


def _crossings(roads: list[Road], structures: list[Structure]) -> list[Crossing]:
    lines = {r.id: shapely.LineString(_centre_xy(r, np.linspace(0.0, r.length, max(2, int(r.length) + 1)))) for r in roads}
    out = []
    for i, a in enumerate(roads):
        for b in roads[i + 1:]:
            for p in _points(lines[a.id].intersection(lines[b.id])):
                sa, sb = a.alignment.project(p.x, p.y)[0], b.alignment.project(p.x, p.y)[0]
                dz = a.profile.z(sa) - b.profile.z(sb)
                if abs(dz) < SAME_SURFACE_M:
                    continue
                (up, su), (lo, sl) = ((a, sa), (b, sb)) if dz > 0 else ((b, sb), (a, sa))
                hu, hl = up.alignment.evaluate(su)[2], lo.alignment.evaluate(sl)[2]
                skew = abs(math.remainder(hu - hl, math.pi)) - math.pi / 2.0
                st = next((x for x in structures if x.carried_road == up.id and x.s_start <= su <= x.s_end), None)
                out.append(
                    Crossing(
                        stable_id("cx", up.id, lo.id, round(p.x, 1), round(p.y, 1)), up.id, su, lo.id, sl, p.x, p.y, abs(dz), abs(skew), TRUTH,
                        clearance_m=abs(dz) - DECK_DEPTH, structure_id=st.id if st else None,
                    )
                )
    return out


def finish(name: str, roads: list[Road], terrain: TerrainPatch | None = None, params: dict | None = None) -> Scene:
    structures = _structures(roads, _ground(terrain))
    unit = UnitRecord(
        unit_id=f"synthetic_{name}",
        aoi_lonlat=(-72.003, 41.997, -71.997, 42.003),
        software_version="test",
        lidar=[LidarCollection("synthetic", 2021, "QL1", 8.0, "unit-frame", "NAVD88")],
        vectors=[VectorSourceRecord("synthetic", "test-only", True)],
    )
    model = RoadModel(unit=unit, frame=FRAME, roads=roads, crossings=_crossings(roads, structures), structures=structures, terrain=terrain)
    return Scene(name, model, dict(params or {}))


# ---------------------------------------------------------------- names, crossing reference, truth for AC-A4 / AC-A5


def _bound(heading: float) -> str:
    deg = math.degrees(heading) % 360.0
    return "northbound" if 45.0 <= deg < 135.0 else "westbound" if 135.0 <= deg < 225.0 else "southbound" if 225.0 <= deg < 315.0 else "eastbound"


def model_name(road: Road) -> str:
    """The name the vector graph gives this road (FR-VC-1): route and direction for one-way mainlines."""
    tags = road.vector_refs[0].tags
    name = tags.get("name") or tags.get("ref") or f"{tags.get('highway', 'road')} {road.vector_refs[0].feature_id}"
    if road.oneway and road.kind == "mainline":
        ref = (tags.get("ref") or name).split(";")[0].strip()
        if ref.startswith("I ") and ref[2:].isdigit():
            ref = f"I-{ref[2:]}"
        (x0, y0), (x1, y1) = road.alignment.offset_point(0.0, 0.0), road.alignment.offset_point(road.length, 0.0)
        name = f"{ref} {_bound(math.atan2(y1 - y0, x1 - x0))}"
    return name


def crossing_reference(scene: Scene) -> list[dict]:
    """AC-H2 hand-labelled list (D-044 entries) for every truth grade separation."""
    m = scene.model
    return [
        {"upper_name": model_name(m.road(c.upper_road)), "lower_name": model_name(m.road(c.lower_road)), "x": round(c.x, 2), "y": round(c.y, 2)}
        for c in sorted(m.crossings, key=lambda c: c.id)
    ]


def truth_dz_at(scene: Scene, model: RoadModel, match_m: float = 15.0) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """AC-A4 reference: per model crossing, the truth surface dz at the matched truth crossing point (D-096)."""
    truth = scene.model
    measured, at = {}, {}
    for c in model.crossings:
        up, lo = model.road(c.upper_road).name, model.road(c.lower_road).name
        best = None
        for t in truth.crossings:
            names = {model_name(truth.road(t.upper_road)), model_name(truth.road(t.lower_road))}
            d = math.hypot(t.x - c.x, t.y - c.y)
            if names == {up, lo} and d <= match_m and (best is None or d < best[0]):
                best = (d, t)
        if best is None:
            continue
        t = best[1]
        tu, tl = truth.road(t.upper_road), truth.road(t.lower_road)
        hu = tu.surface_z(*tu.alignment.project(t.x, t.y))
        hl = tl.surface_z(*tl.alignment.project(t.x, t.y))
        sign = 1.0 if model_name(tu) == up else -1.0
        measured[c.id] = sign * (hu - hl)
        at[c.id] = (t.x, t.y)
    return measured, at


def truth_stations(scene: Scene, model: RoadModel, near_m: float = 4.0) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """AC-A5 reference: truth deck ends projected onto the model structure carried along the same line, and unmatched truth structures.

    Matched in plan, not by name: OSM-like ways of two ramps meeting end to end chain into one model road.
    """
    truth = scene.model
    out, missing = {}, []
    for ts in truth.structures:
        tr = truth.road(ts.carried_road)
        name = model_name(tr)
        probe = [tr.alignment.offset_point(float(s), 0.0) for s in np.linspace(ts.s_start, ts.s_end, 5)]
        hit = None
        for st in model.structures:
            road = model.road(st.carried_road)
            su = [road.alignment.project(x, y) for x, y in probe]
            if any(abs(u) > near_m or not 0.0 <= s <= road.length for s, u in su):
                continue
            ss = sorted((su[0][0], su[-1][0]))
            if ss[0] <= st.s_end and ss[1] >= st.s_start and st.id not in out:
                hit = (st.id, (ss[0], ss[1]))
                break
        if hit is None:
            missing.append(f"{name} s={ts.s_start:.0f}-{ts.s_end:.0f}")
        else:
            out[hit[0]] = hit[1]
    return out, missing


# ---------------------------------------------------------------- vectors


def scene_vectors(scene: Scene, seed: int = 3, bias: float = 1.0, noise: float = 0.2, step: float = 8.0, pin_m: float = 40.0) -> list[VectorEdge]:
    """OSM-like ways: pinned at shared nodes, split into bridge ways over every structure."""
    m = scene.model
    rng = np.random.default_rng(seed)
    lines = {r.id: Polyline.from_alignment(r.alignment, step=1.0) for r in m.roads}
    ends = np.array([p for r in m.roads for p in (lines[r.id].xy[0], lines[r.id].xy[-1])])
    layers = _layers(m)
    out = []
    for road in m.roads:
        ln = lines[road.id]
        ps, _pu, ok = ln.project(ends, max_dist=3.0)
        pins = sorted({0.0, road.length, *(float(v) for v in ps[ok])})
        decks = sorted((st.s_start, st.s_end, st.id) for st in m.structures if st.carried_road == road.id)
        cuts = [v for a, b, _ in decks for v in (a, b)]
        fixed = np.array(sorted({*pins, *cuts}))
        regular = np.linspace(0.0, road.length, max(2, math.ceil(road.length / step) + 1))
        # No vertex a few cm from a split or shared node: its noisy heading would break chaining (FR-VC-1).
        regular = regular[np.min(np.abs(regular[:, None] - fixed[None, :]), axis=1) >= 0.4 * step]
        s = np.unique(np.concatenate([regular, fixed]))
        off = bias * np.sin(s / 90.0 + rng.uniform(0.0, 2.0 * math.pi)) + rng.normal(0.0, noise, len(s))
        dist = np.min(np.abs(s[:, None] - np.array(pins)[None, :]), axis=1)
        off *= np.clip(dist / pin_m, 0.0, 1.0)
        xy = ln.offset_point(s, off)
        fid = road.vector_refs[0].feature_id
        tags = dict(road.vector_refs[0].tags)
        bounds = [0.0, *cuts, road.length]
        n = 0
        for a, b in itertools.pairwise(bounds):
            if b - a < 1e-6:
                continue
            k = (s >= a - 1e-9) & (s <= b + 1e-9)
            piece_tags = dict(tags)
            deck = next((d for d in decks if abs(d[0] - a) < 1e-9 and abs(d[1] - b) < 1e-9), None)
            if deck is not None:
                piece_tags.update({"bridge": "yes", "layer": str(layers[deck[2]])})
            out.append(VectorEdge("synthetic", fid if n == 0 else f"{fid}:{n}", xy[k], piece_tags))
            n += 1
    return out


def _layers(m: RoadModel) -> dict[str, int]:
    """OSM layer of each bridge way: one above the highest layer it crosses."""
    def mean_z(st):
        return m.road(st.carried_road).profile.z(0.5 * (st.s_start + st.s_end))

    layer: dict[str, int] = {}
    for st in sorted(m.structures, key=mean_z):
        below = 0
        for c in m.crossings:
            if c.structure_id != st.id:
                continue
            lower = next((x for x in m.structures if x.carried_road == c.lower_road and x.s_start <= c.lower_s <= x.s_end), None)
            below = max(below, layer.get(lower.id, 0) if lower else 0)
        layer[st.id] = below + 1
    return layer


# ---------------------------------------------------------------- lidar


def scene_points(
    scene: Scene,
    spacing: float = 0.6,
    seed: int = 7,
    terrain_spacing: float | None = None,
    margin: float = 40.0,
    oblique: float = 0.0,
    vehicles: int = 6,
) -> SyntheticPoints:
    """Airborne returns of the truth surfaces; see the module docstring. ``oblique``: returns kept this far inside a higher deck's edge."""
    m = scene.model
    rng = np.random.default_rng(seed)
    ground = _ground(m.terrain)
    geos = [_Geo(r, [(st.s_start, st.s_end) for st in m.structures if st.carried_road == r.id]) for r in m.roads]
    parts: list[tuple[np.ndarray, np.ndarray, int | np.ndarray, str | np.ndarray]] = []

    def covered(xy: np.ndarray, z: np.ndarray, skip: int | None = None) -> np.ndarray:
        out = np.zeros(len(xy), dtype=bool)
        for j, g in enumerate(geos):
            if j == skip:
                continue
            inside, zt = g.footprint(xy, inset=oblique)
            out |= inside & (zt > z + SAME_SURFACE_M)
        return out

    def same_surface(xy: np.ndarray, z: np.ndarray, others: list[int]) -> np.ndarray:
        out = np.zeros(len(xy), dtype=bool)
        for j in others:
            inside, zt = geos[j].footprint(xy, decks=False)
            out |= inside & (np.abs(zt - z) <= SAME_SURFACE_M)
        return out

    for i, g in enumerate(geos):
        road, xs = g.road, g.road.cross_section
        wmax = g.reach - OVERHANG - 1.0
        ss, uu = np.meshgrid(np.arange(0.0, road.length, spacing), np.arange(-wmax, wmax + 1e-9, spacing))
        ss = np.clip(ss.ravel() + rng.uniform(-0.4, 0.4, ss.size) * spacing, 0.0, road.length)
        uu = uu.ravel()
        keep = (uu <= _pw(xs.width_left, ss)) & (uu >= -_pw(xs.width_right, ss))
        ss, uu = ss[keep], uu[keep]
        xy = g.line.offset_point(ss, uu)
        z = surface(road, ss, uu)
        keep = ~covered(xy, z, skip=i) & ~same_surface(xy, z, list(range(i)))
        deck = g.on_deck(ss[keep])
        parts.append((xy[keep], z[keep], np.where(deck, 17, 2), np.where(deck, "deck", "pavement")))
        for b in xs.barriers:
            if b.kind != "parapet":
                continue
            sign = 1.0 if b.side == "left" else -1.0
            s = np.arange(b.s0, b.s1, spacing)
            u = np.full(s.size, sign * (b.offset_m + 0.1))
            bxy = g.line.offset_point(s, u)
            base = surface(road, s, u)
            free = ~same_surface(bxy, base, [j for j in range(len(geos)) if j != i])
            for dz in (0.35, 0.7, b.height_m):
                k = free & ~covered(bxy, base + dz, skip=i)
                parts.append((bxy[k], base[k] + dz, 1, "parapet"))

    # Terrain within ``margin`` of a centre line, fill slopes up to pavement edges off the decks.
    allxy = np.vstack([g.line.xy for g in geos])
    lo, hi = allxy.min(axis=0) - margin, allxy.max(axis=0) + margin
    step = terrain_spacing or spacing * 1.6
    tx, ty = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    txy = np.column_stack([tx.ravel(), ty.ravel()])
    txy = txy + rng.uniform(-0.3, 0.3, txy.shape) * step
    near = np.zeros(len(txy), dtype=bool)
    tz = ground(txy)
    for g in geos:
        s, u, ok = g.line.project(txy, max_dist=margin)
        near |= ok
        if not ok.any():
            continue
        i = np.flatnonzero(ok)
        si, ui = s[i], u[i]
        xs = g.road.cross_section
        half = np.where(ui >= 0.0, _pw(xs.width_left, si), _pw(xs.width_right, si))
        beyond = np.abs(ui) - half
        edge = surface(g.road, si, np.sign(ui) * half)
        edge = np.where(g.on_deck(si), -np.inf, edge)
        ratio = 4.0 if g.road.kind == "mainline" else 2.0
        slope = edge - np.maximum(beyond, 0.0) / ratio
        tz[i] = np.where(beyond > 0.0, np.maximum(tz[i], slope), tz[i])
    keep = near.copy()
    for g in geos:
        inside, _z = g.footprint(txy)
        keep &= ~inside
    parts.append((txy[keep], tz[keep], 2, "terrain"))

    # Vehicles where they can be seen.
    for _ in range(vehicles):
        i = int(rng.integers(len(geos)))
        g = geos[i]
        s0 = float(rng.uniform(0.1, 0.85) * g.road.length)
        s = rng.uniform(s0, s0 + 4.5, 120)
        u = rng.uniform(-2.6, -0.8, 120)
        vxy = g.line.offset_point(s, u)
        vz = surface(g.road, s, u) + rng.uniform(1.2, 1.5, 120)
        k = ~covered(vxy, vz - 1.4, skip=i)
        parts.append((vxy[k], vz[k], 1, "vehicle"))

    nxy = np.column_stack([rng.uniform(lo[0] + margin, hi[0] - margin, 40), rng.uniform(lo[1] + margin, hi[1] - margin, 40)])
    parts.append((nxy, ground(nxy) - 8.0, 7, "noise"))

    xy = np.vstack([p[0] for p in parts])
    z = np.concatenate([p[1] for p in parts])
    cls = np.concatenate([np.broadcast_to(np.asarray(p[2]), len(p[1])) for p in parts]).astype(np.uint8)
    truth = np.concatenate([np.broadcast_to(np.asarray(p[3]), len(p[1])) for p in parts]).astype(str)
    z = z + rng.normal(0.0, 0.03, z.size)
    strip = (xy[:, 0] > 0.0).astype(np.uint16) + 1
    z = z + np.where(strip == 2, 0.02, 0.0)
    intensity = np.where(np.isin(truth, ["pavement", "deck"]), 30.0, 80.0) + rng.normal(0.0, 5.0, z.size)
    return SyntheticPoints(
        xyz=np.column_stack([xy, z]),
        classification=cls,
        intensity=intensity,
        return_number=np.ones(z.size, dtype=np.uint8),
        point_source_id=strip,
        truth=truth,
    )


def pavement_xyz(points: SyntheticPoints) -> np.ndarray:
    """Truth-labelled pavement and deck returns (AC-A1 reference)."""
    return points.xyz[np.isin(points.truth, ["pavement", "deck"])]


# ---------------------------------------------------------------- scene builders


def _flat(length: float, z: float = GRADE_Z) -> Profile:
    return Profile([VerticalElement(0.0, length, z, 0.0, 0.0, TRUTH)])


def simple_overpass(*, skew: float = 0.17, rise: float = 8.5, length: float = 400.0) -> Scene:
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    st = street("st", "Route 1", length, math.pi / 2.0 + skew, crest_profile(length, length / 2.0, rise, length / 2.0 - 50.0, 160.0, 60.0))
    return finish("simple_overpass", [eb, wb, st], params={"skew": skew, "rise": rise, "length": length})


def _freeway_over(length: float, rise: float) -> Profile:
    return crest_profile(length, length / 2.0, rise, 150.0, 150.0, 100.0)


def _street_over(length: float, rise: float) -> Profile:
    return crest_profile(length, length / 2.0, rise, 120.0, 100.0, 60.0)


def _exit_body(spread: float, sign: float, alpha: float = 0.35, radius: float = 200.0) -> list[El]:
    """After a leg turning ``sign * THETA``: out to ``sign * alpha``, ``spread`` m further off, back parallel, free line to the end."""
    return [arc(sign * (alpha - THETA), radius), line(spread / math.sin(alpha)), arc(-sign * alpha, radius), line(free=True)]


def _diamond_ramps(eb: Road, wb: Road, st: Road, spread: float, s_exit: float = 20.0) -> list[Road]:
    """Exit and entry ramps on both carriageways, ending (starting) on the street centre line at grade."""
    ramps = []
    for parent, tag in ((eb, "eb"), (wb, "wb")):
        ex = ramp(f"exit_{tag}", parent, _exit_body(spread, -1.0), s=s_exit, side=-1.0, target=st, end="terminal")
        ramps.append(ex)
        # Entry: built as an exit from the reversed carriageway (on its left), then reversed, so it starts on the street.
        en = reverse(ramp(f"entry_{tag}", reverse(parent), _exit_body(spread, 1.0), s=s_exit, side=1.0, target=st, end="terminal"))
        en.cross_section.lanes = list(ex.cross_section.lanes)
        ramps.append(en)
    return ramps


def diamond(*, skew: float | None = None, spread: float | None = None, rise: float = 8.0, length: float = 600.0) -> Scene:
    skew = 0.0 if skew is None else float(skew)
    spread = 20.0 if spread is None else float(spread)
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    st = street("st", "Route 1", 400.0, math.pi / 2.0 + skew, _street_over(400.0, rise))
    ramps = _diamond_ramps(eb, wb, st, spread)
    return finish("diamond", [eb, wb, st, *ramps], params={"skew": skew, "spread": spread, "rise": rise})


def spui(*, skew: float | None = None, deck_half: float = 12.0, rise: float = 8.5, length: float = 600.0) -> Scene:
    skew = 0.0 if skew is None else float(skew)
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    ls = 400.0
    c = ls / 2.0
    half = Piecewise([Knot(c - 90.0, STREET_HALF, TRUTH), Knot(c - 50.0, deck_half, TRUTH), Knot(c + 50.0, deck_half, TRUTH), Knot(c + 90.0, STREET_HALF, TRUTH)])
    st = street("st", "Route 1", ls, math.pi / 2.0 + skew, crest_profile(ls, c, rise, 140.0, 120.0, 60.0), half=half)
    ramps = _diamond_ramps(eb, wb, st, 0.0, s_exit=60.0)
    return finish("spui", [eb, wb, st, *ramps], params={"skew": skew, "deck_half": deck_half, "rise": rise})


def _loop(fid: str, parent: Road, target: Road, radius: float, peak: float | None = None) -> Road:
    """Clockwise 270-degree loop: right leg off ``parent`` (start solved), merging ``target`` from its right."""
    _r, sep = LOOP_LEG
    th = sep / _r
    h0 = parent.alignment.evaluate(0.0)[2]
    h1 = target.alignment.evaluate(0.0)[2]
    main = turn_to(h0 - th, h1 + th, -1.0)
    return ramp(fid, parent, [arc(main, radius)], s=None, side=-1.0, target=target, end="merge", target_side=-1.0, leg=LOOP_LEG, peak=peak)


def parclo(*, skew: float | None = None, radius: float | None = None, spread: float | None = None, rise: float = 8.0) -> Scene:
    """Diamond ramps on the south side (EB), two loops on the north side (WB): entry from the street's NB half, exit to its SB half."""
    skew = 0.1 if skew is None else float(skew)
    radius = 65.0 if radius is None else float(radius)
    spread = 30.0 if spread is None else float(spread)
    length, ls = 700.0, 400.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    st = street("st", "Route 1", ls, math.pi / 2.0 + skew, _street_over(ls, rise))
    ex = ramp("exit_eb", eb, _exit_body(spread, -1.0), s=20.0, side=-1.0, target=st, end="terminal")
    en = reverse(ramp("entry_eb", reverse(eb), _exit_body(spread, 1.0), s=20.0, side=1.0, target=st, end="terminal"))
    en.cross_section.lanes = list(ex.cross_section.lanes)
    loops = [_loop("loop_nb_wb", st, wb, radius), _loop("loop_wb_sb", wb, reverse(st), radius)]
    return finish("parclo", [eb, wb, st, ex, en, *loops], params={"skew": skew, "radius": radius, "spread": spread})


def cloverleaf(*, skew: float | None = None, radius: float | None = None, rise: float = 8.5) -> Scene:
    skew = 0.1 if skew is None else float(skew)
    radius = 65.0 if radius is None else float(radius)
    length = 600.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    nb, sb = twins(length, math.pi / 2.0 + skew, "I 88", ("nb", "sb"), (_street_over(length, rise), _street_over(length, rise)))
    loops = [_loop("loop_sb_eb", sb, eb, radius), _loop("loop_nb_wb", nb, wb, radius), _loop("loop_eb_nb", eb, nb, radius), _loop("loop_wb_sb", wb, sb, radius)]
    return finish("cloverleaf", [eb, wb, nb, sb, *loops], params={"skew": skew, "radius": radius})


def trumpet(*, radius: float | None = None, rise: float = 8.0) -> Scene:
    radius = 70.0 if radius is None else float(radius)
    length = 700.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    stem_len, j, x_stem = 150.0, -260.0, 6.0
    xs2 = _section(FREEWAY_HALF, FREEWAY_HALF, 2)
    nb = _road("nb", "mainline", "motorway", _straight(x_stem, j - stem_len, math.pi / 2.0, stem_len), _flat(stem_len), xs2, oneway=True, lanes=2, ref="I 199")
    sb = _road("sb", "mainline", "motorway", _straight(-x_stem, j, -math.pi / 2.0, stem_len), _flat(stem_len), _section(FREEWAY_HALF, FREEWAY_HALF, 2), oneway=True, lanes=2, ref="I 199")
    top = GRADE_Z + rise
    _r, sep = LOOP_LEG
    th = sep / _r
    # Loop: off WB to the north, clockwise round to the south, over the mainline beside the semi-directional, to the stem.
    body = [arc(turn_to(math.pi - th, -math.pi / 2.0, -1.0), radius), line(free=True)]
    loop = ramp("loop_ws", wb, body, s=None, side=-1.0, end="point", point=(-x_stem, j), end_z=GRADE_Z, peak=top, grade=0.06, leg=LOOP_LEG)
    # Climb late, so the loop is at grade until it heads for the mainline (top reached 15 m before crossing WB).
    climb = 2.0 * PEAK_CURVE + (rise - 0.06 * PEAK_CURVE) / 0.06
    lead = loop.length - (MEDIAN_HALF - j) - 15.0 - climb - sep - CURVE
    loop = ramp("loop_ws", wb, body, s=None, side=-1.0, end="point", point=(-x_stem, j), end_z=GRADE_Z, peak=top, grade=0.06, leg=LOOP_LEG, lead=max(lead, 0.0))
    # Semi-directional: north over the mainline, round the outside of the loop (concentric), joining WB from the north.
    outer = radius + 2.0 * x_stem
    cy = _loop_centre(loop)[1]
    alpha = 0.7
    # Outer ramp to EB shares the semi-directional's first metres north of the stem; the semi stays level until they part.
    outer_ne = ramp("outer_ne", nb, [line(free=True), arc(-(math.pi / 2.0) + THETA, 200.0)], s=stem_len, start="terminal", start_heading=math.pi / 2.0, target=eb, end="merge", target_side=-1.0)
    shared = outer_ne.alignment.elements[0].length if outer_ne.alignment.elements[0].kind == "line" else 0.0
    semi = ramp(
        "semi_nw", nb, [line(cy - j), arc(math.pi / 2.0 + alpha, outer), line(free=True), arc(-(alpha - THETA), 250.0)],
        s=stem_len, start="terminal", start_heading=math.pi / 2.0, target=wb, end="merge", target_side=-1.0, peak=top, grade=0.06, lead=shared + 15.0,
    )
    # Outer ramp from EB merges onto the stem past the loop's end, so it comes alongside the loop only where both are at grade.
    outer_es = ramp("outer_es", eb, [arc(-(math.pi / 2.0) + 2.0 * THETA, 150.0), line(120.0)], s=None, side=-1.0, target=sb, end="merge", target_side=-1.0)
    return finish("trumpet", [eb, wb, nb, sb, loop, semi, outer_ne, outer_es], params={"radius": radius})


def _loop_centre(road: Road) -> tuple[float, float]:
    """Centre of curvature of a road's longest arc."""
    el = max((e for e in road.alignment.elements if e.kind == "arc"), key=lambda e: e.length)
    x, y = el.point(0.0)
    r = 1.0 / el.k0
    return x - r * math.sin(el.heading0), y + r * math.cos(el.heading0)


def directional(*, skew: float | None = None, radius: float | None = None, rise: float = 8.0) -> Scene:
    skew = 0.0 if skew is None else float(skew)
    radius = 220.0 if radius is None else float(radius)
    length = 900.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    nb, sb = twins(length, math.pi / 2.0 + skew, "I 88", ("nb", "sb"), (_street_over(length, rise), _street_over(length, rise)))
    top = GRADE_Z + 2.0 * rise - 1.0
    flys = []
    for fid, parent, target in (("fly_eb_nb", eb, nb), ("fly_wb_sb", wb, sb)):
        h0, h1 = parent.alignment.evaluate(0.0)[2], target.alignment.evaluate(0.0)[2]
        main = turn_to(h0 - THETA, h1 + THETA, 1.0)
        flys.append(ramp(fid, parent, [line(60.0), arc(main, radius), line(free=True)], s=None, side=-1.0, target=target, end="merge", target_side=-1.0, peak=top, grade=0.075))
    return finish("directional", [eb, wb, nb, sb, *flys], params={"skew": skew, "radius": radius})


def cd_braided(*, spacing: float | None = None, angle: float | None = None, rise: float = 7.5) -> Scene:
    spacing = 200.0 if spacing is None else float(spacing)
    angle = 0.4 if angle is None else float(angle)
    length = 1000.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    # C-D road: leaves EB, runs 18 m off it for 450 m, rejoins (symmetric, so it lands back on EB).
    cd = ramp("cd", eb, [line(60.0), arc(THETA, LEAVE_R), line(450.0), arc(THETA, LEAVE_R), line(60.0)], s=100.0, side=-1.0, target=eb, end="merge", target_side=-1.0, half=4.0)
    cd.kind = "connector"
    s_mid = 0.5 * cd.length
    leg = (300.0, 45.0)
    th = leg[1] / leg[0]
    exit_ = ramp("braid_exit", cd, [arc(-angle + th, 300.0), line(320.0)], s=s_mid - spacing, side=-1.0, end="free", leg=leg)
    back = reverse(cd)
    entry = ramp("braid_entry", back, [arc(angle - th, 300.0), line(320.0)], s=back.length - (s_mid + spacing), side=1.0, end="free", peak=GRADE_Z + rise, grade=0.07, leg=leg)
    entry = reverse(entry)
    entry.cross_section.lanes = list(exit_.cross_section.lanes)
    return finish("cd_braided", [eb, wb, cd, exit_, entry], params={"spacing": spacing, "angle": angle})


def _beside(main: Road, half: float, gap: float) -> tuple[float, CrossSection, float]:
    """Offset to the right of a crowned ``main``, section and height of a road ``gap`` m beside its pavement, on the plane of its right half."""
    offset = main.cross_section.width_right(0.0) + gap + half
    slope = main.cross_section.slope_right(0.0)
    return offset, _section(half, half, 1, slope, slope), main.profile.z(0.0) - slope * offset


def aux_lane(*, spacing: float | None = None, gap: float | None = None) -> Scene:
    """An auxiliary lane between two interchanges: in from one, ``spacing`` m alongside EB ``gap`` m off its pavement, out to the next."""
    spacing = 300.0 if spacing is None else float(spacing)
    gap = 0.0 if gap is None else float(gap)
    length = 900.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    offset, xs, z = _beside(eb, RAMP_HALF, gap)
    a, taper, r = 0.2, 80.0, LEAVE_R
    els = [line(taper), arc(-a, r), line(spacing), arc(-a, r), line(taper)]
    dx = taper * math.cos(a) + r * math.sin(a)
    dy = taper * math.sin(a) + r * (1.0 - math.cos(a))
    al = _chain(-0.5 * spacing - dx, -MEDIAN_HALF - offset - dy, a, els)
    aux = _road("aux", "ramp", "motorway_link", al, _flat(al.length, z), xs, oneway=True, lanes=1)
    return finish("aux_lane", [eb, wb, aux], params={"spacing": spacing, "gap": gap})


def cd_parallel(*, gap: float | None = None) -> Scene:
    """A collector-distributor road beside EB for the whole scene, ``gap`` m off its pavement and never joining it."""
    gap = 0.5 if gap is None else float(gap)
    length = 700.0
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (_flat(length), _flat(length)))
    offset, xs, z = _beside(eb, 4.0, gap)
    cd = _road("cd", "connector", "motorway_link", _straight(-0.5 * length, -MEDIAN_HALF - offset, 0.0, length), _flat(length, z), xs, oneway=True, lanes=1)
    return finish("cd_parallel", [eb, wb, cd], params={"gap": gap})


def viaduct(*, skew: float | None = None, n_streets: int = 3, spacing: float | None = None, rise: float = 8.5, dead_end: bool = False, street_length: float = 200.0, crest: float = 0.0) -> Scene:
    """Twin carriageways on one viaduct over ``n_streets`` streets.

    ``dead_end`` adds a street that passes under the eastbound deck and ends at grade under
    the westbound one, 1.5 m short of its centre line. ``crest``: each street rises this
    much to a crest under the eastbound deck (a cross street's crown), over 60 m.
    """
    skew = 0.2 if skew is None else float(skew)
    spacing = 60.0 if spacing is None else float(spacing)
    run = (n_streets - 1) * spacing
    length = run + 2 * 220.0
    top_a, top_b = 220.0 - 40.0, 220.0 + run + 40.0
    pr = pvi_profile([(0.0, GRADE_Z), (60.0, GRADE_Z), (top_a - 30.0, GRADE_Z + rise), (top_b + 30.0, GRADE_Z + rise), (length - 60.0, GRADE_Z), (length, GRADE_Z)], [40.0, 60.0, 60.0, 40.0])
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (pr, pvi_profile([(0.0, GRADE_Z), (60.0, GRADE_Z), (top_a - 30.0, GRADE_Z + rise), (top_b + 30.0, GRADE_Z + rise), (length - 60.0, GRADE_Z), (length, GRADE_Z)], [40.0, 60.0, 60.0, 40.0])))
    roads = [eb, wb]
    names = ["1st Street", "2nd Street", "3rd Street", "4th Street", "5th Street", "6th Street"]
    for k in range(n_streets):
        x = -run / 2.0 + k * spacing
        pr_st = _flat(street_length)
        if crest:
            c = 0.5 * street_length - MEDIAN_HALF / math.cos(skew)
            pr_st = pvi_profile([(0.0, GRADE_Z), (c - 30.0, GRADE_Z), (c, GRADE_Z + 2.0 * crest), (c + 30.0, GRADE_Z), (street_length, GRADE_Z)], [20.0, 30.0, 20.0])
        roads.append(street(f"st{k + 1}", names[k], street_length, math.pi / 2.0 + skew, pr_st, centre=(x, 0.0)))
    if dead_end:
        end_y = MEDIAN_HALF - 1.5
        roads.append(street("dead", "Dead End Street", 90.0, math.pi / 2.0, _flat(90.0), centre=(run / 2.0 + 45.0, end_y - 45.0)))
    return finish("viaduct", roads, params={"skew": skew, "n_streets": n_streets, "spacing": spacing, "dead_end": dead_end, "street_length": street_length, "crest": crest})


def twin_different_elevations(*, dz: float | None = None, skew: float | None = None, rise: float = 8.5) -> Scene:
    dz = 1.5 if dz is None else float(dz)
    skew = 0.25 if skew is None else float(skew)
    length = 500.0
    eb_pr = Profile([VerticalElement(0.0, length, GRADE_Z, 0.008, 0.008, TRUTH)])
    # Westbound runs dz higher at the crossing on its own grade: z(x) = 100 + dz + 0.004 x + 2.0 for x in [-250, 250].
    wb_pr = Profile([VerticalElement(0.0, length, GRADE_Z + dz + 0.008 * 250.0 + 0.004 * 250.0, -0.004, -0.004, TRUTH)])
    eb, wb = twins(length, 0.0, "I 99", ("eb", "wb"), (eb_pr, wb_pr))
    base = GRADE_Z + max(dz, 0.0) + 0.008 * 250.0
    ls = 400.0
    st = street("st", "Route 1", ls, math.pi / 2.0 + skew, crest_profile(ls, ls / 2.0, rise + base - GRADE_Z, ls / 2.0 - 50.0, 160.0, 60.0))
    return finish("twin_different_elevations", [eb, wb, st], params={"dz": dz, "skew": skew})


def stack(*, levels: int = 3, skew: float | None = None, level_dz: float = 8.0) -> Scene:
    from tests.pipeline.stack_scene import _level_profile

    skew = 0.0 if skew is None else float(skew)
    headings = [math.radians(96.0), math.radians(45.0), math.radians(135.0)]
    ramps_ = []
    longest = 0.0
    for k in range(1, levels):
        profile, length = _level_profile(level_dz * k)
        h = headings[k - 1] + skew
        half = length / 2.0
        r = _road(f"l{k}", "ramp", "motorway_link", _straight(-half * math.cos(h), -half * math.sin(h), h, length), profile, _section(4.5, 4.5, 1), oneway=True, lanes=1)
        ramps_.append(r)
        longest = max(longest, length)
    fl = max(500.0, longest * 0.75)
    eb, wb = twins(fl, 0.0, "I 99", ("eb", "wb"), (_flat(fl), _flat(fl)))
    return finish("stack", [eb, wb, *ramps_], params={"levels": levels, "skew": skew})


# Fuzz ranges per type: name -> {parameter: (low, high)}; integers where both bounds are int.
PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "simple_overpass": {"skew": (0.0, math.radians(60.0)), "rise": (7.5, 9.5)},
    "diamond": {"skew": (0.0, math.radians(25.0)), "spread": (15.0, 30.0)},
    "spui": {"skew": (0.0, math.radians(25.0)), "deck_half": (9.0, 14.0)},
    "parclo": {"skew": (0.0, math.radians(20.0)), "radius": (55.0, 80.0), "spread": (20.0, 45.0)},
    "cloverleaf": {"skew": (0.0, math.radians(20.0)), "radius": (55.0, 80.0)},
    "trumpet": {"radius": (60.0, 80.0)},
    "directional": {"skew": (0.0, math.radians(20.0)), "radius": (200.0, 240.0)},
    "cd_braided": {"spacing": (180.0, 230.0), "angle": (0.3, 0.5)},
    "aux_lane": {"spacing": (200.0, 400.0), "gap": (0.0, 0.5)},
    "cd_parallel": {"gap": (0.0, 2.0)},
    "viaduct": {"skew": (0.0, math.radians(60.0)), "n_streets": (3, 4), "spacing": (45.0, 80.0)},
    "twin_different_elevations": {"dz": (0.8, 2.5), "skew": (0.0, math.radians(60.0))},
    "stack": {"levels": (3, 4), "skew": (-0.3, 0.3)},
}


def fuzz_params(name: str, seed: int) -> dict:
    """Seeded parameters for ``SCENES[name]`` drawn from ``PARAM_RANGES``."""
    rng = np.random.default_rng([seed, sorted(SCENES).index(name)])
    out = {}
    for key, (lo, hi) in sorted(PARAM_RANGES[name].items()):
        out[key] = int(rng.integers(lo, hi + 1)) if isinstance(lo, int) and isinstance(hi, int) else round(float(rng.uniform(lo, hi)), 4)
    return out


SCENES: dict[str, Callable[..., Scene]] = {
    "simple_overpass": simple_overpass,
    "diamond": diamond,
    "spui": spui,
    "parclo": parclo,
    "cloverleaf": cloverleaf,
    "trumpet": trumpet,
    "directional": directional,
    "cd_braided": cd_braided,
    "aux_lane": aux_lane,
    "cd_parallel": cd_parallel,
    "viaduct": viaduct,
    "twin_different_elevations": twin_different_elevations,
    "stack": stack,
}


# ---------------------------------------------------------------- one scene through the pipeline, solids and gates


@dataclass
class Outcome:
    """A scene's pipeline run: the gate results by name, or the error that stopped it."""

    scene: Scene
    error: str | None = None
    stage: str = ""
    model: RoadModel | None = None
    gates: dict = field(default_factory=dict)
    missing_structures: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


def mesh_hashes(meshes) -> dict[str, str]:
    """sha256 per mesh of its id, element, part, vertices and faces: the AC-H5 comparison without writing files."""
    import hashlib

    out = {}
    for m in meshes:
        h = hashlib.sha256()
        h.update(f"{m.id}|{m.element_id}|{m.part}|{m.source}".encode())
        h.update(np.ascontiguousarray(m.vertices, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m.faces, dtype=np.int64).tobytes())
        out[m.id] = h.hexdigest()
    return dict(sorted(out.items()))


def run_scene(scene: Scene, *, spacing: float = 0.6, seed: int = 7, profiles: tuple[str, ...] = ("analytic",), gates: bool = True, val3dity_exe: str | None = None) -> Outcome:
    """Points and vectors of ``scene`` through ``run_pipeline`` (D-018 holdout), solids and every section 9 gate.

    Solids are rendered tolerantly (FR-EX-1, D-119): an element that cannot be built is
    replaced or dropped, flagged on the model and reported by the FR-EX-1 gate.
    Solids are rendered twice, from the model and from its canonical reload; AC-H5 compares
    the two renderings mesh by mesh (``mesh_hashes``) instead of writing exports. With
    ``gates=False`` only the pipeline and the solids run. Exceptions are caught and returned
    with the stage that raised, so a crash is a reported outcome.
    """
    import time

    from roader.export.run import with_failures
    from roader.io import from_synthetic
    from roader.model.serialize import dumps, loads
    from roader.pipeline import holdout_for, run_pipeline
    from roader.solids import generate
    from roader.validate import surface_accuracy
    from roader.validate.gates import compare_runs, run_gates

    out = Outcome(scene)
    t0 = time.perf_counter()
    sp = scene_points(scene, spacing=spacing, seed=seed)
    pts = from_synthetic(sp)
    vectors = scene_vectors(scene)
    pav = pavement_xyz(sp)
    hold = holdout_for(pts, pav)
    out.seconds["sample"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        res = run_pipeline(pts, vectors, scene.model.frame, scene.model.unit, holdout=hold)
    except Exception as exc:  # noqa: BLE001 - a crash is an outcome (robust track)
        out.error, out.stage = f"{type(exc).__name__}: {exc}", "pipeline"
        return out
    out.seconds["pipeline"] = time.perf_counter() - t0
    out.model = model = res.model
    dz, at = truth_dz_at(scene, model)
    stations, out.missing_structures = truth_stations(scene, model)
    measured = {"crossing_dz": dz, "crossing_points": at, "structure_stations": stations}
    ref = crossing_reference(scene)
    for profile in profiles:
        t0 = time.perf_counter()
        stage = f"solids ({profile})"
        try:
            failures: list = []
            meshes = generate(model, profile=profile, failures=failures)
            again = generate(loads(dumps(model)), profile=profile, failures=[])
            if failures:
                model = out.model = with_failures(model, failures)
            if not gates:
                out.seconds[profile] = time.perf_counter() - t0
                continue
            stage = f"gates ({profile})"
            report = run_gates(model, meshes, crossings_ref=ref, measured=measured, val3dity_exe=val3dity_exe, profile=profile)
        except Exception as exc:  # noqa: BLE001
            out.error, out.stage = f"{type(exc).__name__}: {exc}", stage
            return out
        out.seconds[profile] = time.perf_counter() - t0
        for r in report.results:
            if profile == "analytic" or r.gate.startswith("AC-H1") or r.gate == "FR-EX-1":
                out.gates[r.gate] = r
        if profile == "analytic":
            out.gates["AC-H5"] = compare_runs(mesh_hashes(meshes), mesh_hashes(again))
    if gates:
        out.gates["AC-A1"] = surface_accuracy(model, pav)
    return out
