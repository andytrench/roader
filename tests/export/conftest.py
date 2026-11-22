"""Fixtures for the export lane: the synthetic overpass plus a curved ramp, and simple
closed prism meshes per model element standing in for the solids lane."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roader.model import (
    AlignElement,
    Alignment,
    CrossSection,
    Flag,
    Interval,
    JunctionArea,
    Knot,
    LaneGroup,
    Link,
    Mesh,
    Piecewise,
    Profile,
    Provenance,
    Road,
    RoadModel,
    Source,
    VerticalElement,
    constant,
    signed_volume,
    stable_id,
)
from roader.model.synthetic import overpass_model, overpass_points
from roader.validate.report import GateReport, GateResult, Violation

M = Provenance(Source.MEASURED, 0.9, rms_m=0.04, n_points=1000)
R = Provenance(Source.RULE, 0.5, rule_id="test")


def _pw(*pairs: tuple[float, float]) -> Piecewise:
    return Piecewise([Knot(s, v, M) for s, v in pairs])


def curved_road(x0: float, y0: float, heading: float) -> Road:
    """Line, spiral, arc, spiral, line; a crest-free sag profile; banking and a width taper."""
    k = 1.0 / 200.0
    specs = [("line", 50.0, 0.0, 0.0), ("spiral", 60.0, 0.0, k), ("arc", 80.0, k, k), ("spiral", 60.0, k, 0.0),
             ("line", 50.0, 0.0, 0.0)]
    elements = []
    s, x, y, h = 0.0, x0, y0, heading
    for kind, length, k0, k1 in specs:
        e = AlignElement(kind, s, length, x, y, h, k0, k1, M)
        elements.append(e)
        x, y, h, _k = e.end_state()
        s += length
    g1 = 0.04
    z1 = 100.0
    z2 = z1 + 0.5 * g1 * 100.0
    profile = Profile([
        VerticalElement(0.0, 100.0, z1, 0.0, 0.0, M),
        VerticalElement(100.0, 100.0, z1, 0.0, g1, M),
        VerticalElement(200.0, 100.0, z2, g1, g1, M),
    ])
    xs = CrossSection(
        width_left=_pw((0.0, 4.0), (100.0, 4.0), (150.0, 5.0)),
        width_right=constant(4.0, M),
        slope_left=_pw((0.0, -0.02), (50.0, -0.02), (110.0, -0.06), (190.0, -0.06), (250.0, -0.02)),
        slope_right=_pw((0.0, 0.02), (50.0, 0.02), (80.0, 0.0), (110.0, -0.06), (190.0, -0.06), (220.0, 0.0),
                        (250.0, 0.02)),
        pavement_thickness=constant(0.3, R),
        lanes=[LaneGroup("right", 0.0, 1e9, 1, 3.66, 0.0, 0.34, R)],
    )
    road = Road(
        id=stable_id("rd", "test", "ramp"), name="Ramp A", road_class="motorway_link", kind="ramp", oneway=True,
        alignment=Alignment(elements), profile=profile, cross_section=xs,
        support=[Interval(0.0, 120.0, "observed"), Interval(120.0, 160.0, "unsupported", "canopy"),
                 Interval(160.0, 300.0, "observed")],
    )
    return road


def build_model() -> RoadModel:
    model = overpass_model()
    eb = model.roads[0]
    x, y, h, _k = eb.alignment.evaluate(eb.length)
    ramp = curved_road(x, y, h)
    model.roads.append(ramp)
    model.links.append(Link(stable_id("lk", eb.id, ramp.id), eb.id, "end", ramp.id, "start", "diverge"))
    model.junctions.append(JunctionArea(stable_id("jn", eb.id, ramp.id), [eb.id, ramp.id],
                                        [(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5)], R))
    # Pretend the bridge matched an inventory record and one slope came from a vector hint,
    # so all four provenance sources appear (FR-RH-1).
    model.structures[0].prov = Provenance(Source.INVENTORY, 0.8)
    model.structures[0].inventory_id = "NBI-TEST-1"
    model.side_slopes[0].prov = Provenance(Source.VECTOR, 0.6)
    model.flags.append(Flag("unsupported_gap", ramp.id, "FR-VC-2", "no sheet under canopy 120-160 m"))
    return model


# ---------------------------------------------------------------- simple closed meshes


def _closed_tube(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed prism through K rings of 4 corners each, capped; outward by signed volume."""
    k = len(rings)
    verts = np.vstack(rings)
    faces = []
    for i in range(k - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            c, d = (i + 1) * 4 + (j + 1) % 4, (i + 1) * 4 + j
            faces += [(a, b, c), (a, c, d)]
    last = (k - 1) * 4
    faces += [(0, 2, 1), (0, 3, 2), (last, last + 1, last + 2), (last, last + 2, last + 3)]
    faces = np.array(faces, dtype=np.int64)
    probe = Mesh("probe", "probe", "L3", "probe", "probe", Source.RULE, verts, faces)
    if signed_volume(probe) < 0.0:
        faces = faces[:, ::-1].copy()
    return verts, faces


def _stations(road: Road, s0: float, s1: float, step: float = 20.0) -> list[float]:
    stations = {s0, s1}
    stations.update(e.s0 for e in road.alignment.elements if s0 < e.s0 < s1)
    n = max(1, math.ceil((s1 - s0) / step))
    stations.update(s0 + (s1 - s0) * i / n for i in range(n + 1))
    return sorted(stations)


def swept(road: Road, s0: float, s1: float, u_right: float, u_left: float, dz_top: float, depth: float
          ) -> tuple[np.ndarray, np.ndarray]:
    rings = []
    for s in _stations(road, s0, s1):
        corners = []
        for u, top in ((u_left, True), (u_right, True), (u_right, False), (u_left, False)):
            x, y = road.alignment.offset_point(s, u)
            z = road.surface_z(s, u) + dz_top - (0.0 if top else depth)
            corners.append((x, y, z))
        rings.append(np.array(corners))
    return _closed_tube(rings)


def box(x: float, y: float, heading: float, half_along: float, half_across: float, z0: float, z1: float
        ) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(heading), math.sin(heading)
    rings = []
    for a in (-half_along, half_along):
        ring = []
        for b, z in ((half_across, z1), (-half_across, z1), (-half_across, z0), (half_across, z0)):
            ring.append((x + a * c - b * s, y + a * s + b * c, z))
        rings.append(np.array(ring))
    return _closed_tube(rings)


def build_meshes(model: RoadModel) -> list[Mesh]:
    meshes: list[Mesh] = []

    def add(mid: str, element_id: str, layer: str, part: str, material: str, source: Source, vf, closed=True):
        meshes.append(Mesh(mid, element_id, layer, part, material, source, vf[0], vf[1], closed))

    for road in model.roads:
        xs = road.cross_section
        wl, wr = xs.width_left(0.0), xs.width_right(0.0)
        add(f"{road.id}:pavement", road.id, "L3", "pavement", "asphalt", road.alignment.elements[0].prov.source,
            swept(road, 0.0, road.length, -wr, wl, 0.0, xs.pavement_thickness(0.0)))
        for barrier in xs.barriers:
            sign = 1.0 if barrier.side == "left" else -1.0
            u_in, u_out = sign * barrier.offset_m, sign * (barrier.offset_m + barrier.base_width_m)
            add(f"{barrier.id}:parapet", barrier.id, "L5", "parapet", "structural_concrete", barrier.prov.source,
                swept(road, barrier.s0, barrier.s1, min(u_in, u_out), max(u_in, u_out), barrier.height_m,
                      barrier.height_m))
    for slope in model.side_slopes:
        road = model.road(slope.road_id)
        w = road.cross_section.width_left(0.0) if slope.side == "left" else road.cross_section.width_right(0.0)
        lo, hi = (w + 0.01, w + 6.0) if slope.side == "left" else (-(w + 6.0), -(w + 0.01))
        add(f"{slope.id}:embankment", slope.id, "L4", "embankment", "soil_slope", slope.prov.source,
            swept(road, slope.s0, slope.s1, lo, hi, -0.3, 1.0))
    for st in model.structures:
        road = model.road(st.carried_road)
        half = road.cross_section.width_left(0.0) + st.deck.overhang_m
        add(f"{st.id}:deck", st.id, "L5", "deck", "deck", st.prov.source,
            swept(road, st.s_start, st.s_end, -half, half, -0.26, st.deck.depth_m - 0.26))
        for sp in st.supports:
            x, y, h, _k = road.alignment.evaluate(sp.station)
            soffit = road.profile.z(sp.station) - st.deck.depth_m - 0.01
            if sp.kind == "abutment":
                add(f"{sp.id}:abutment", sp.id, "L5", "abutment", "structural_concrete", sp.prov.source,
                    box(x, y, h + sp.skew_rad, 0.75, half, sp.footing_z, soffit))
                continue
            cap_bottom = soffit - sp.cap_depth_m
            add(f"{sp.id}:pier_cap", sp.id, "L5", "pier_cap", "structural_concrete", sp.prov.source,
                box(x, y, h + sp.skew_rad, 0.5 * sp.cap_width_m, half - 0.5, cap_bottom, soffit))
            for i in range(sp.n_columns):
                u = (i - 0.5 * (sp.n_columns - 1)) * 4.0
                cx, cy = road.alignment.offset_point(sp.station, u)
                add(f"{sp.id}:pier_column:{i}", sp.id, "L5", "pier_column", "structural_concrete", sp.prov.source,
                    box(cx, cy, h + sp.skew_rad, 0.5 * sp.column_width_m, 0.5 * sp.column_width_m, sp.footing_z,
                        cap_bottom - 0.01))
    # Open L7 terrain surface: a coarse 4 x 4 grid.
    grid = np.linspace(-300.0, 300.0, 5)
    gx, gy = np.meshgrid(grid, grid)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 99.0)])
    faces = []
    for r in range(4):
        for c in range(4):
            a = r * 5 + c
            faces += [(a, a + 1, a + 6), (a, a + 6, a + 5)]
    add("terrain:surface", "terrain", "L7", "terrain", "terrain", Source.MEASURED, (verts, np.array(faces)),
        closed=False)
    return meshes


def gate_report(unit_id: str, *, passing: bool) -> GateReport:
    results = [
        GateResult("AC-H1", passing, True, {"solids": 12.0},
                   [] if passing else [Violation("FR-SV-1", "st_x", "open edge at deck end")]),
        GateResult("AC-H5", True, True),
        GateResult("AC-A1", True, False, {"rms_m": 0.041, "p95_m": 0.08}),
    ]
    return GateReport(unit_id, results)


@pytest.fixture(scope="session")
def model() -> RoadModel:
    return build_model()


@pytest.fixture(scope="session")
def meshes(model: RoadModel) -> list[Mesh]:
    return build_meshes(model)


@pytest.fixture(scope="session")
def points(model: RoadModel):
    return overpass_points(overpass_model(), spacing=1.5)
