"""Synthetic four-level stack for milestone 2 layer tests (FR-RS-2, FR-VC-5, FR-CR-3, FR-OC-1..3).

Level 0 is a twin-carriageway freeway on light fill along +x (carriageways at
y = -9 and +9, z = 100). Three one-way ramps cross it through the origin on
crest-curved viaducts: level 1 heading about 96 degrees peaking at z = 107,
level 2 at 45 degrees peaking at 114, level 3 at 135 degrees peaking at 121.
Each ramp climbs from grade at 6 %, so every level overlaps every lower level in
plan at the origin and the plan cell there holds four surfaces.

Airborne-lidar defects: nothing is sampled under a higher surface except a
1.2 m strip inside each deck edge (oblique returns, so cells there hold two to
four layers); vehicles on both levels; a tree canopy over the westbound
freeway; strips 2 cm apart. A paved ramp with no vector leaves the eastbound
carriageway at grade, joined to it by a 25 m paved taper (FR-VC-5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roader.io import VectorEdge
from roader.model.geometry import (
    AlignElement,
    Alignment,
    Profile,
    VerticalElement,
    constant,
)
from roader.model.provenance import Provenance, Source
from roader.model.schema import CrossSection, LaneGroup, Road, VectorRef
from roader.model.serialize import stable_id
from roader.model.synthetic import SyntheticPoints

TERRAIN_Z = 99.0
GRADE_Z = 100.0
LEVEL_DZ = 8.0
GRADE = 0.06
SAG_M = 60.0
CREST_M = 120.0
FLAT_M = 30.0
MEDIAN_HALF = 9.0
FREEWAY_HALF = 5.5
RAMP_HALF = 4.5
OVERHANG = 0.9
PARAPET_M = 1.07
OBLIQUE_M = 1.2  # returns kept this far inside a higher deck's edge
STRUCTURE_ABOVE_M = 5.5  # the truth deck is where the road is this far above the terrain
FILL_RATIO = 2.0
FREEWAY_LENGTH = 700.0
TAPER = (470.0, 495.0)  # eastbound stations of the paved taper to the unvectored ramp
LEVEL_HEADINGS = {1: math.radians(96.0), 2: math.radians(45.0), 3: math.radians(135.0)}

TRUTH = Provenance(Source.MEASURED, 1.0)
RULE = Provenance(Source.RULE, 0.5, rule_id="synthetic")


def _section(half: float, lanes: int) -> CrossSection:
    return CrossSection(
        width_left=constant(half, TRUTH),
        width_right=constant(half, TRUTH),
        slope_left=constant(-0.02, TRUTH),
        slope_right=constant(0.02, TRUTH),
        pavement_thickness=constant(0.3, RULE),
        lanes=[LaneGroup("left", 0.0, 1e9, max(1, lanes // 2), 3.66, 1.2, 3.0, RULE), LaneGroup("right", 0.0, 1e9, max(1, lanes - lanes // 2), 3.66, 1.2, 3.0, RULE)],
    )


def _level_profile(height: float) -> tuple[Profile, float]:
    """Flat, sag to +GRADE, tangent, crest to -GRADE, tangent, sag, flat; peak ``height`` above grade."""
    rise_curves = GRADE * SAG_M / 2.0 + GRADE * CREST_M / 4.0
    tangent = (height - rise_curves) / GRADE
    els = []
    s, z = 0.0, GRADE_Z
    for length, g0, g1 in (
        (FLAT_M, 0.0, 0.0),
        (SAG_M, 0.0, GRADE),
        (tangent, GRADE, GRADE),
        (CREST_M, GRADE, -GRADE),
        (tangent, -GRADE, -GRADE),
        (SAG_M, -GRADE, 0.0),
        (FLAT_M, 0.0, 0.0),
    ):
        els.append(VerticalElement(s, length, z, g0, g1, TRUTH))
        z += 0.5 * (g0 + g1) * length
        s += length
    return Profile(els), s


def _line(x0: float, y0: float, heading: float, length: float) -> Alignment:
    return Alignment([AlignElement("line", 0.0, length, x0, y0, heading, 0.0, 0.0, TRUTH)])


@dataclass
class StackScene:
    roads: list[Road]
    level: dict[str, int]  # feature id -> level
    decks: dict[str, tuple[float, float]]  # feature id -> truth structure station range
    unvectored: np.ndarray  # plan centerline of the ramp that has no vector


def stack_scene() -> StackScene:
    roads, level, decks = [], {}, {}
    for fid, y, heading in (("eb", -MEDIAN_HALF, 0.0), ("wb", MEDIAN_HALF, math.pi)):
        x0 = -FREEWAY_LENGTH / 2.0 if heading == 0.0 else FREEWAY_LENGTH / 2.0
        roads.append(
            Road(
                id=stable_id("rd", "stack", fid),
                name=f"Freeway {fid.upper()}",
                road_class="motorway",
                kind="mainline",
                oneway=True,
                alignment=_line(x0, y, heading, FREEWAY_LENGTH),
                profile=Profile([VerticalElement(0.0, FREEWAY_LENGTH, GRADE_Z, 0.0, 0.0, TRUTH)]),
                cross_section=_section(FREEWAY_HALF, 2),
                vector_refs=[VectorRef("synthetic", fid, {"highway": "motorway"})],
            )
        )
        level[fid] = 0
    for k, heading in LEVEL_HEADINGS.items():
        fid = f"l{k}"
        profile, length = _level_profile(LEVEL_DZ * k)
        half = length / 2.0
        road = Road(
            id=stable_id("rd", "stack", fid),
            name=f"Level {k} ramp",
            road_class="motorway_link",
            kind="ramp",
            oneway=True,
            alignment=_line(-half * math.cos(heading), -half * math.sin(heading), heading, length),
            profile=profile,
            cross_section=_section(RAMP_HALF, 1),
            vector_refs=[VectorRef("synthetic", fid, {"highway": "motorway_link", "ref": f"L{k}"})],
        )
        roads.append(road)
        level[fid] = k
        s = np.arange(0.0, length, 0.25)
        up = np.array([profile.z(float(v)) for v in s]) - TERRAIN_Z >= STRUCTURE_ABOVE_M
        decks[fid] = (float(s[up][0]), float(s[up][-1]))
    # The unvectored ramp: leaves the eastbound right edge at x = 120 heading -8 degrees.
    t = np.arange(0.0, 160.0, 1.0)
    h = math.radians(-8.0)
    start = np.array([120.0, -MEDIAN_HALF - FREEWAY_HALF - RAMP_HALF])
    unvectored = start + t[:, None] * np.array([math.cos(h), math.sin(h)])
    return StackScene(roads, level, decks, unvectored)


def stack_vectors(scene: StackScene, seed: int = 5, bias: float = 1.2, noise: float = 0.3, step: float = 10.0) -> list[VectorEdge]:
    """Centerlines with a slowly varying lateral bias and vertex noise; the unvectored ramp is left out."""
    rng = np.random.default_rng(seed)
    out = []
    for road in scene.roads:
        s = np.arange(0.0, road.length + 1e-9, step)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        h = road.alignment.evaluate(0.0)[2]
        normal = np.array([-math.sin(h), math.cos(h)])
        offset = bias * np.sin(s / 120.0) + rng.normal(0.0, noise, len(s))
        ref = road.vector_refs[0]
        tags = {**ref.tags, "oneway": "yes"}
        if road.kind == "mainline":
            tags["ref"] = "I 99"
        else:
            tags["name"] = road.name
        out.append(VectorEdge("synthetic", ref.feature_id, xy + offset[:, None] * normal, tags))
    return out


def _frame(road: Road, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, heading, _ = road.alignment.evaluate(0.0)
    dx, dy = x - x0, y - y0
    c, sn = math.cos(heading), math.sin(heading)
    return dx * c + dy * sn, -dx * sn + dy * c


def _xy(road: Road, s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, heading, _ = road.alignment.evaluate(0.0)
    c, sn = math.cos(heading), math.sin(heading)
    return x0 + s * c - u * sn, y0 + s * sn + u * c


def _surface(road: Road, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    z = np.array([road.profile.z(float(v)) for v in np.clip(s, 0.0, road.length)])
    return z + np.where(u >= 0.0, -0.02, 0.02) * u


def _covered(scene: StackScene, x: np.ndarray, y: np.ndarray, z: np.ndarray, margin: float) -> np.ndarray:
    """True where a road surface more than 1 m above z covers (x, y), ``margin`` inside its deck edge."""
    out = np.zeros(len(x), dtype=bool)
    for road in scene.roads:
        s, u = _frame(road, x, y)
        half = road.cross_section.width_left(0.0) + OVERHANG
        inside = (s >= 0.0) & (s <= road.length) & (np.abs(u) <= half - margin)
        if not inside.any():
            continue
        zt = np.full(len(x), -np.inf)
        zt[inside] = _surface(road, s[inside], u[inside])
        out |= inside & (zt > z + 1.0)
    return out


def stack_points(scene: StackScene | None = None, spacing: float = 0.35, seed: int = 11) -> tuple[SyntheticPoints, np.ndarray]:
    """The scene's returns and, per return, the feature id of the road surface it lies on ("" otherwise)."""
    scene = scene or stack_scene()
    rng = np.random.default_rng(seed)
    xs, ys, zs, cls, labels, owners = [], [], [], [], [], []

    def add(x, y, z, code, label, owner=""):
        xs.append(x)
        ys.append(y)
        zs.append(z)
        cls.append(np.full(len(x), code))
        labels.append(np.full(len(x), label))
        owners.append(np.full(len(x), owner, dtype=object))

    for road in scene.roads:
        half = road.cross_section.width_left(0.0)
        fid = road.vector_refs[0].feature_id
        ss, uu = np.meshgrid(np.arange(0.0, road.length, spacing), np.arange(-half, half, spacing))
        ss = ss.ravel() + rng.uniform(-0.4, 0.4, ss.size) * spacing
        uu = uu.ravel()
        x, y = _xy(road, ss, uu)
        z = _surface(road, ss, uu)
        keep = ~_covered(scene, x, y, z, OBLIQUE_M)
        deck = scene.decks.get(fid)
        on_deck = np.zeros(len(x), dtype=bool) if deck is None else (ss >= deck[0]) & (ss <= deck[1])
        add(x[keep], y[keep], z[keep], 2, "pavement", fid)
        cls[-1] = np.where(on_deck[keep], 17, 2)
        labels[-1] = np.where(on_deck[keep], "deck", "pavement")
        if deck is None:
            continue
        # Parapets 1.07 m tall at the deck pavement edges, sampled on top and inner face.
        s = np.arange(deck[0], deck[1], spacing)
        for sign in (1.0, -1.0):
            for dz in (0.35, 0.7, PARAPET_M):
                u = np.full(s.size, sign * (half + 0.1))
                x, y = _xy(road, s, u)
                z = _surface(road, s, u) + dz
                keep = ~_covered(scene, x, y, z, OBLIQUE_M)
                add(x[keep], y[keep], z[keep], 1, "parapet")

    # Unvectored ramp at grade, joined to the eastbound pavement over its first 25 m.
    line = scene.unvectored
    h = math.atan2(line[1, 1] - line[0, 1], line[1, 0] - line[0, 0])
    tt, uu = np.meshgrid(np.arange(0.0, len(line) - 1.0, spacing), np.arange(-RAMP_HALF, RAMP_HALF, spacing))
    tt, uu = tt.ravel(), uu.ravel()
    x = line[0, 0] + tt * math.cos(h) - uu * math.sin(h)
    y = line[0, 1] + tt * math.sin(h) + uu * math.cos(h)
    add(x, y, np.full(len(x), GRADE_Z) - 0.02 * np.abs(uu), 2, "pavement_unvectored")
    eb = scene.roads[0]
    s_t, u_t = np.meshgrid(np.arange(TAPER[0], TAPER[1], spacing), np.arange(-FREEWAY_HALF - 2 * RAMP_HALF, -FREEWAY_HALF, spacing))
    s_t, u_t = s_t.ravel(), u_t.ravel()
    gx, gy = _xy(eb, s_t, u_t)
    add(gx, gy, GRADE_Z - 0.02 * np.abs(u_t) + 0.0, 2, "pavement_unvectored")

    # Terrain within 35 m of every centerline: flat at TERRAIN_Z, fill slopes up to
    # pavement edges off the structures, nothing under any road or deck.
    tx, ty = np.meshgrid(np.arange(-360.0, 360.0, spacing * 1.7), np.arange(-360.0, 360.0, spacing * 1.7))
    tx, ty = tx.ravel(), ty.ravel()
    near = np.zeros(tx.size, dtype=bool)
    tz = np.full(tx.size, TERRAIN_Z)
    paved = np.zeros(tx.size, dtype=bool)
    for road in scene.roads:
        s, u = _frame(road, tx, ty)
        inside_s = (s >= 0.0) & (s <= road.length)
        near |= inside_s & (np.abs(u) <= 35.0)
        half = road.cross_section.width_left(0.0)
        fid = road.vector_refs[0].feature_id
        deck = scene.decks.get(fid)
        paved |= inside_s & (np.abs(u) <= half + (OVERHANG if deck else 0.0) - (OBLIQUE_M if deck else 0.0))
        edge = np.full(tx.size, -np.inf)
        edge[inside_s] = np.array([road.profile.z(float(v)) for v in s[inside_s]]) - 0.02 * half
        if deck is not None:
            edge[inside_s & (s >= deck[0]) & (s <= deck[1])] = -np.inf
        slope = edge - (np.abs(u) - half) / FILL_RATIO
        tz = np.where(inside_s & (np.abs(u) > half), np.maximum(tz, slope), tz)
    dx, dy = tx - line[0, 0], ty - line[0, 1]
    s_r, u_r = dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)
    paved |= (s_r >= 0.0) & (s_r <= 160.0) & (np.abs(u_r) <= RAMP_HALF)
    s_e, u_e = _frame(eb, tx, ty)
    paved |= (s_e >= TAPER[0]) & (s_e <= TAPER[1]) & (u_e >= -FREEWAY_HALF - 2 * RAMP_HALF) & (u_e <= -FREEWAY_HALF)
    keep = near & ~paved & ~_covered(scene, tx, ty, tz, OBLIQUE_M)
    add(tx[keep], ty[keep], tz[keep], 2, "terrain")

    # Tree canopy over the westbound carriageway at x = -200: rough returns 6 to 12 m up,
    # nothing on the pavement beneath it.
    wb = scene.roads[1]
    cx, cy = np.meshgrid(np.arange(-215.0, -185.0, 0.4), np.arange(MEDIAN_HALF - 8.0, MEDIAN_HALF + 8.0, 0.4))
    cx, cy = cx.ravel(), cy.ravel()
    add(cx, cy, GRADE_Z + rng.uniform(6.0, 12.0, cx.size), 5, "canopy")

    # Vehicles on the freeway and on the level 2 deck.
    for road, s_car, u0 in ((eb, 100.0, -3.0), (eb, 520.0, -3.0), (wb, 260.0, 1.2), (scene.roads[3], 0.0, -2.0)):
        if road is scene.roads[3]:
            s_car = 0.5 * road.length + 40.0
        s = rng.uniform(s_car, s_car + 4.5, 120)
        u = rng.uniform(u0, u0 + 1.8, 120)
        x, y = _xy(road, s, u)
        add(x, y, _surface(road, s, u) + rng.uniform(1.2, 1.5, 120), 1, "vehicle")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    truth = np.concatenate(labels)
    owner = np.concatenate(owners).astype(str)
    classification = np.concatenate(cls).astype(np.uint8)
    # Wooded cells under the canopy have no pavement returns.
    _s_w, u_w = _frame(wb, x, y)
    under_tree = (np.abs(x + 200.0) <= 15.0) & (np.abs(u_w) <= 8.0) & (truth == "pavement")
    keep = ~under_tree
    x, y, z, truth, classification, owner = x[keep], y[keep], z[keep], truth[keep], classification[keep], owner[keep]
    z = z + rng.normal(0.0, 0.03, z.size)
    strip = (x > 0.0).astype(np.uint16) + 1
    z = z + np.where(strip == 2, 0.02, 0.0)
    paved_label = np.isin(truth, ["pavement", "deck", "pavement_unvectored"])
    intensity = np.where(paved_label, 30.0, 80.0) + rng.normal(0.0, 5.0, z.size)
    pts = SyntheticPoints(
        xyz=np.column_stack([x, y, z]),
        classification=classification,
        intensity=intensity.astype(np.float64),
        return_number=np.ones(z.size, dtype=np.uint8),
        point_source_id=strip,
        truth=truth,
    )
    return pts, owner
