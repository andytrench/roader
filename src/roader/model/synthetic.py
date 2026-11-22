"""Deterministic synthetic overpass: a reference model plus the lidar it would produce.

Shared by every lane's tests so stages can be built and checked without real data.
Scene: a twin-carriageway freeway on light fill along +x, crossed by a two-lane road
on a skewed bridge with approach embankments and a pier in the freeway median.
Nothing under the deck is sampled (spec section 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roader.model.geometry import (
    AlignElement,
    Alignment,
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
    Interval,
    LaneGroup,
    LidarCollection,
    Road,
    RoadModel,
    SideSlope,
    Structure,
    Support,
    TerrainPatch,
    UnitRecord,
    VectorRef,
    VectorSourceRecord,
)
from roader.model.serialize import stable_id

TERRAIN_Z = 99.0
MAINLINE_Z = 100.0
MEDIAN_HALF = 9.0  # carriageway reference lines at y = -9 (eastbound) and +9 (westbound)
OVERPASS_SKEW = 0.17  # radians from perpendicular to the freeway
OVERPASS_LENGTH = 500.0
DECK_DEPTH = 1.3

TRUTH = Provenance(Source.MEASURED, 1.0)
RULE = Provenance(Source.RULE, 0.5, rule_id="synthetic")


def _line(x0: float, y0: float, heading: float, length: float) -> Alignment:
    return Alignment([AlignElement("line", 0.0, length, x0, y0, heading, 0.0, 0.0, TRUTH)])


def _section(half_left: float, half_right: float, lanes: int, thickness: float) -> CrossSection:
    return CrossSection(
        width_left=constant(half_left, TRUTH),
        width_right=constant(half_right, TRUTH),
        slope_left=constant(-0.02, TRUTH),
        slope_right=constant(0.02, TRUTH),
        pavement_thickness=constant(thickness, RULE),
        lanes=[
            LaneGroup("left", 0.0, 1e9, lanes // 2, 3.66, 1.2, 3.0, RULE),
            LaneGroup("right", 0.0, 1e9, lanes - lanes // 2, 3.66, 1.2, 3.0, RULE),
        ],
    )


def _overpass_profile() -> Profile:
    g = 0.03
    curve = 160.0
    tangent = (OVERPASS_LENGTH - curve) / 2.0
    z1 = MAINLINE_Z + g * tangent
    z2 = z1 + g * curve - 0.5 * (2 * g) * curve
    return Profile(
        [
            VerticalElement(0.0, tangent, MAINLINE_Z, g, g, TRUTH),
            VerticalElement(tangent, curve, z1, g, -g, TRUTH),
            VerticalElement(tangent + curve, tangent, z2, -g, -g, TRUTH),
        ]
    )


def overpass_model() -> RoadModel:
    eb = Road(
        id=stable_id("rd", "synthetic", "eb"),
        name="Freeway EB",
        road_class="motorway",
        kind="mainline",
        oneway=True,
        alignment=_line(-300.0, -MEDIAN_HALF, 0.0, 600.0),
        profile=Profile([VerticalElement(0.0, 600.0, MAINLINE_Z, 0.0, 0.0, TRUTH)]),
        cross_section=_section(5.5, 5.5, 2, 0.3),
        vector_refs=[VectorRef("synthetic", "eb", {"highway": "motorway"})],
    )
    wb = Road(
        id=stable_id("rd", "synthetic", "wb"),
        name="Freeway WB",
        road_class="motorway",
        kind="mainline",
        oneway=True,
        alignment=_line(300.0, MEDIAN_HALF, math.pi, 600.0),
        profile=Profile([VerticalElement(0.0, 600.0, MAINLINE_Z, 0.0, 0.0, TRUTH)]),
        cross_section=_section(5.5, 5.5, 2, 0.3),
        vector_refs=[VectorRef("synthetic", "wb", {"highway": "motorway"})],
    )
    heading = math.pi / 2.0 + OVERPASS_SKEW
    half = OVERPASS_LENGTH / 2.0
    over = Road(
        id=stable_id("rd", "synthetic", "over"),
        name="Route 1",
        road_class="primary",
        kind="crossing_street",
        oneway=False,
        alignment=_line(-half * math.cos(heading), -half * math.sin(heading), heading, OVERPASS_LENGTH),
        profile=_overpass_profile(),
        cross_section=_section(5.0, 5.0, 2, 0.25),
        vector_refs=[VectorRef("synthetic", "over", {"highway": "primary", "bridge": "yes"})],
    )

    # Abutments 26 m either side of the freeway centre, measured across the freeway.
    s_mid = half
    reach = 26.0 / math.cos(OVERPASS_SKEW)
    s_start, s_end = s_mid - reach, s_mid + reach
    over.cross_section.barriers = [
        Barrier(stable_id("br", over.id, side), side, s_start, s_end, 5.0, 1.07, 0.45, 0.25, "parapet", RULE)
        for side in ("left", "right")
    ]
    over.support = [
        Interval(0.0, s_start, "observed"),
        Interval(s_start, s_end, "observed"),
        Interval(s_end, OVERPASS_LENGTH, "observed"),
    ]
    for road in (eb, wb):
        a = road.alignment
        # Freeway stations under the deck are occluded (FR-OC-1).
        s_under = 300.0
        road.support = [
            Interval(0.0, s_under - 14.0, "observed"),
            Interval(s_under - 14.0, s_under + 14.0, "occluded", "under_deck"),
            Interval(s_under + 14.0, a.length, "observed"),
        ]

    sid = stable_id("st", over.id, round(s_start, 1))
    supports = [
        Support(stable_id("sp", sid, "a1"), s_start, "abutment", OVERPASS_SKEW, 0, 0.0, 1.2, 1.5, TERRAIN_Z - 1.5, RULE),
        Support(stable_id("sp", sid, "p1"), s_mid, "pier", OVERPASS_SKEW, 2, 1.0, 1.2, 1.4, TERRAIN_Z - 1.5, RULE),
        Support(stable_id("sp", sid, "a2"), s_end, "abutment", OVERPASS_SKEW, 0, 0.0, 1.2, 1.5, TERRAIN_Z - 1.5, RULE),
    ]
    structure = Structure(
        id=sid,
        kind="bridge",
        carried_road=over.id,
        s_start=s_start,
        s_end=s_end,
        family="multi_girder",
        year_built=None,
        crossed_roads=[eb.id, wb.id],
        deck=DeckSpec(DECK_DEPTH, 0.22, 0.9, RULE),
        supports=supports,
        prov=TRUTH,
    )

    crossings = []
    for lower in (eb, wb):
        y = lower.alignment.evaluate(0.0)[1]
        s_upper = s_mid + y / math.sin(heading)
        x = s_upper * math.cos(heading) + over.alignment.evaluate(0.0)[0]
        s_lower = (x - lower.alignment.evaluate(0.0)[0]) * math.cos(lower.alignment.evaluate(0.0)[2])
        dz = over.profile.z(s_upper) - lower.profile.z(s_lower)
        crossings.append(
            Crossing(
                id=stable_id("cx", over.id, lower.id),
                upper_road=over.id,
                upper_s=s_upper,
                lower_road=lower.id,
                lower_s=s_lower,
                x=x,
                y=y,
                dz_m=dz,
                skew_rad=OVERPASS_SKEW,
                prov=TRUTH,
                clearance_m=dz - DECK_DEPTH,
                structure_id=sid,
            )
        )

    slopes = [
        SideSlope(stable_id("ss", road.id, side), road.id, side, 0.0, road.length, "fill", 4.0, RULE)
        for road in (eb, wb)
        for side in ("left", "right")
    ]
    for side in ("left", "right"):
        slopes.append(SideSlope(stable_id("ss", over.id, side, "a"), over.id, side, 0.0, s_start, "fill", 2.0, RULE))
        slopes.append(SideSlope(stable_id("ss", over.id, side, "b"), over.id, side, s_end, OVERPASS_LENGTH, "fill", 2.0, RULE))

    cell = 2.0
    nx = ny = 301
    terrain = TerrainPatch(-300.0, -300.0, cell, nx, ny, [TERRAIN_Z] * (nx * ny), TRUTH)

    unit = UnitRecord(
        unit_id="synthetic_overpass",
        aoi_lonlat=(-72.003, 41.997, -71.997, 42.003),
        software_version="test",
        lidar=[LidarCollection("synthetic", 2021, "QL1", 8.0, "unit-frame", "NAVD88")],
        vectors=[VectorSourceRecord("synthetic", "test-only", True)],
    )
    frame = Frame(proj="+proj=tmerc +lat_0=42 +lon_0=-72 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", anchor_lon=-72.0, anchor_lat=42.0)
    return RoadModel(
        unit=unit,
        frame=frame,
        roads=[eb, wb, over],
        crossings=crossings,
        structures=[structure],
        side_slopes=slopes,
        terrain=terrain,
    )


# ---------------------------------------------------------------- synthetic lidar


@dataclass
class SyntheticPoints:
    """Arrays in the unit frame. class codes follow ASPRS (2 ground, 1 unclassified, 17 deck, 7 noise)."""

    xyz: np.ndarray
    classification: np.ndarray
    intensity: np.ndarray
    return_number: np.ndarray
    point_source_id: np.ndarray
    truth: np.ndarray  # label: "pavement" | "deck" | "terrain" | "vehicle" | "parapet" | "noise"


def _road_xy_frame(road: Road, s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, heading, _ = road.alignment.evaluate(0.0)
    c, sn = math.cos(heading), math.sin(heading)
    return x0 + s * c - u * sn, y0 + s * sn + u * c


def overpass_points(model: RoadModel | None = None, spacing: float = 0.35, seed: int = 7) -> SyntheticPoints:
    """About 8 points per m^2 on observed surfaces, 3 cm noise, two strips 2 cm apart."""
    model = model or overpass_model()
    rng = np.random.default_rng(seed)
    over = model.roads[2]
    st = model.structures[0]
    xs, ys, zs, cls, labels = [], [], [], [], []

    def deck_footprint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, y0, heading, _ = over.alignment.evaluate(0.0)
        dx, dy = x - x0, y - y0
        s = dx * math.cos(heading) + dy * math.sin(heading)
        u = -dx * math.sin(heading) + dy * math.cos(heading)
        return (s >= st.s_start) & (s <= st.s_end) & (np.abs(u) <= 5.0 + st.deck.overhang_m + 0.5)

    for road in model.roads:
        wl = road.cross_section.width_left(0.0)
        wr = road.cross_section.width_right(0.0)
        s_grid = np.arange(0.0, road.length, spacing)
        u_grid = np.arange(-wr, wl, spacing)
        ss, uu = np.meshgrid(s_grid, u_grid)
        ss, uu = ss.ravel(), uu.ravel()
        ss = ss + rng.uniform(-0.4, 0.4, ss.size) * spacing
        x, y = _road_xy_frame(road, ss, uu)
        z = np.array([road.surface_z(float(s), float(u)) for s, u in zip(ss, uu, strict=True)])
        on_deck = np.zeros(ss.size, dtype=bool)
        if road is over:
            on_deck = (ss >= st.s_start) & (ss <= st.s_end)
            keep = np.ones(ss.size, dtype=bool)
        else:
            keep = ~deck_footprint(x, y)
        xs.append(x[keep]); ys.append(y[keep]); zs.append(z[keep])
        cls.append(np.where(on_deck[keep], 17, 2))
        labels.append(np.where(on_deck[keep], "deck", "pavement"))

    # Parapets on the deck, 1.07 m tall, sampled on their tops and inner faces.
    for barrier in over.cross_section.barriers:
        sign = 1.0 if barrier.side == "left" else -1.0
        s = np.arange(barrier.s0, barrier.s1, spacing)
        for dz in (0.35, 0.7, barrier.height_m):
            u = np.full(s.size, sign * (barrier.offset_m + 0.1))
            x, y = _road_xy_frame(over, s, u)
            z = np.array([over.surface_z(float(a), float(b)) for a, b in zip(s, u, strict=True)]) + dz
            xs.append(x); ys.append(y); zs.append(z); cls.append(np.full(s.size, 1)); labels.append(np.full(s.size, "parapet"))

    # Terrain everywhere outside road footprints and outside the deck shadow, with fill
    # slopes rising to the pavement edges.
    tx, ty = np.meshgrid(np.arange(-300.0, 300.0, spacing * 1.6), np.arange(-300.0, 300.0, spacing * 1.6))
    tx, ty = tx.ravel(), ty.ravel()
    tz = np.full(tx.size, TERRAIN_Z)
    keep = ~deck_footprint(tx, ty)
    for road in model.roads:
        x0, y0, heading, _ = road.alignment.evaluate(0.0)
        dx, dy = tx - x0, ty - y0
        s = dx * math.cos(heading) + dy * math.sin(heading)
        u = -dx * math.sin(heading) + dy * math.cos(heading)
        inside_s = (s >= 0.0) & (s <= road.length)
        half = road.cross_section.width_left(0.0)
        on_pavement = inside_s & (np.abs(u) <= half)
        keep &= ~on_pavement
        ratio = 4.0 if road.kind == "mainline" else 2.0
        edge_z = np.array([road.profile.z(float(v)) for v in np.clip(s, 0.0, road.length)]) - 0.02 * half
        if road is over:
            in_span = (s >= st.s_start) & (s <= st.s_end)
            edge_z = np.where(in_span, TERRAIN_Z, edge_z)
        slope_z = edge_z - (np.abs(u) - half) / ratio
        tz = np.where(inside_s & (np.abs(u) > half), np.maximum(tz, slope_z), tz)
    xs.append(tx[keep]); ys.append(ty[keep]); zs.append(tz[keep]); cls.append(np.full(int(keep.sum()), 2)); labels.append(np.full(int(keep.sum()), "terrain"))

    # Vehicles on the eastbound carriageway: 4.5 x 1.8 x 1.5 m boxes, class 1.
    eb = model.roads[0]
    for s_car in (60.0, 180.0, 420.0, 520.0):
        s = rng.uniform(s_car, s_car + 4.5, 120)
        u = rng.uniform(-3.0, -1.2, 120)
        x, y = _road_xy_frame(eb, s, u)
        z = MAINLINE_Z + rng.uniform(1.2, 1.5, 120)
        xs.append(x); ys.append(y); zs.append(z); cls.append(np.full(120, 1)); labels.append(np.full(120, "vehicle"))

    # Low noise below ground.
    nx_ = rng.uniform(-250, 250, 40)
    ny_ = rng.uniform(-250, 250, 40)
    xs.append(nx_); ys.append(ny_); zs.append(np.full(40, TERRAIN_Z - 8.0)); cls.append(np.full(40, 7)); labels.append(np.full(40, "noise"))

    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    classification = np.concatenate(cls).astype(np.uint8)
    truth = np.concatenate(labels)
    z = z + rng.normal(0.0, 0.03, z.size)
    strip = (x > 0.0).astype(np.uint16) + 1
    z = z + np.where(strip == 2, 0.02, 0.0)
    intensity = np.where(np.isin(truth, ["pavement", "deck"]), 30.0, 80.0) + rng.normal(0.0, 5.0, z.size)
    return SyntheticPoints(
        xyz=np.column_stack([x, y, z]),
        classification=classification,
        intensity=intensity.astype(np.float64),
        return_number=np.ones(z.size, dtype=np.uint8),
        point_source_id=strip,
        truth=truth,
    )
