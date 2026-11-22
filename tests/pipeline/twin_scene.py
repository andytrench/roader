"""Synthetic twin decks over a divided freeway, with a ramp diverging on one deck (FR-CR-3, FR-ST-3).

The freeway runs along +x (carriageways at y = -9 and +9, 5.5 m half widths, z = 100),
median pier line at y = 0. Two one-way carriageways cross it on skewed twin decks:
``west`` heads about 113 degrees, ``east`` the opposite way, their reference lines
13 m apart, so the pavements leave a 1 m median gap and the rule overhangs (0.9 m)
of the two decks overlap there. Both decks carry rule parapets on both sides. A ramp
leaves ``west`` 15 m before its deck end, 2 m right of its reference line, veering
right by 0.05 rad, so its first metres ride on the west deck.
"""

from __future__ import annotations

import math

import shapely

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
    LaneGroup,
    LidarCollection,
    Link,
    Road,
    RoadModel,
    Structure,
    Support,
    UnitRecord,
    VectorRef,
    VectorSourceRecord,
)
from roader.model.serialize import stable_id

SKEW = 0.4
HEADING = math.pi / 2.0 + SKEW
HALF = 6.0
CENTRE_GAP = 13.0  # between the twin reference lines
OVERHANG = 0.9
DEPTH = 1.3
LENGTH = 500.0
RAMP_SEAM_BEFORE_END = 15.0
RAMP_U0 = -2.0
RAMP_TURN = -0.05
RAMP_HALF = 2.5

TRUTH = Provenance(Source.MEASURED, 1.0)
RULE = Provenance(Source.RULE, 0.5, rule_id="synthetic")


def _section(half: float) -> CrossSection:
    return CrossSection(
        width_left=constant(half, TRUTH),
        width_right=constant(half, TRUTH),
        slope_left=constant(-0.02, TRUTH),
        slope_right=constant(0.02, TRUTH),
        pavement_thickness=constant(0.3, RULE),
        lanes=[LaneGroup("left", 0.0, 1e9, 1, 3.66, 1.2, 3.0, RULE), LaneGroup("right", 0.0, 1e9, 1, 3.66, 1.2, 3.0, RULE)],
    )


def _crest() -> Profile:
    g, curve = 0.03, 160.0
    tangent = (LENGTH - curve) / 2.0
    z1 = 100.0 + g * tangent  # a symmetric crest returns to z1
    return Profile([VerticalElement(0.0, tangent, 100.0, g, g, TRUTH), VerticalElement(tangent, curve, z1, g, -g, TRUTH), VerticalElement(tangent + curve, tangent, z1, -g, -g, TRUTH)])


def _road(fid: str, x0: float, y0: float, heading: float, length: float, profile: Profile, half: float, kind: str, cls: str) -> Road:
    return Road(
        id=stable_id("rd", "twin", fid),
        name=fid,
        road_class=cls,
        kind=kind,
        oneway=True,
        alignment=Alignment([AlignElement("line", 0.0, length, x0, y0, heading, 0.0, 0.0, TRUTH)]),
        profile=profile,
        cross_section=_section(half),
        vector_refs=[VectorRef("synthetic", fid, {"highway": cls})],
    )


def _structure(road: Road, crossed: list[Road]) -> Structure:
    stations = [road.alignment.project(0.0, y)[0] for y in (-9.0, 9.0)]
    mid = 0.5 * sum(stations)
    reach = 26.0 / math.cos(SKEW)
    a, b = mid - reach, mid + reach
    sid = stable_id("st", road.id, round(a, 1))
    supports = [
        Support(stable_id("sp", sid, "a1"), a, "abutment", SKEW, 0, 0.0, 1.2, 1.5, 98.5, RULE),
        Support(stable_id("sp", sid, "p1"), mid, "pier", SKEW, 2, 1.0, 1.2, 1.4, 98.5, RULE),
        Support(stable_id("sp", sid, "a2"), b, "abutment", SKEW, 0, 0.0, 1.2, 1.5, 98.5, RULE),
    ]
    road.cross_section.barriers = [
        Barrier(stable_id("br", road.id, side), side, a, b, HALF, 1.07, 0.45, 0.25, "parapet", RULE) for side in ("left", "right")
    ]
    return Structure(sid, "bridge", road.id, a, b, "multi_girder", None, [r.id for r in crossed], DeckSpec(DEPTH, 0.22, OVERHANG, RULE), supports, TRUTH)


def twin_model(ramp: bool = False) -> RoadModel:
    flat = Profile([VerticalElement(0.0, 600.0, 100.0, 0.0, 0.0, TRUTH)])
    eb = _road("eb", -300.0, -9.0, 0.0, 600.0, flat, 5.5, "mainline", "motorway")
    wb = _road("wb", 300.0, 9.0, math.pi, 600.0, Profile([VerticalElement(0.0, 600.0, 100.0, 0.0, 0.0, TRUTH)]), 5.5, "mainline", "motorway")
    d = (math.cos(HEADING), math.sin(HEADING))
    n = (-d[1], d[0])
    half_len, off = LENGTH / 2.0, CENTRE_GAP / 2.0
    west = _road("west", -half_len * d[0] - off * n[0], -half_len * d[1] - off * n[1], HEADING, LENGTH, _crest(), HALF, "mainline", "motorway")
    east = _road("east", half_len * d[0] + off * n[0], half_len * d[1] + off * n[1], HEADING + math.pi, LENGTH, _crest(), HALF, "mainline", "motorway")
    structures = [_structure(west, [eb, wb]), _structure(east, [eb, wb])]
    roads = [eb, wb, west, east]
    crossings = []
    for upper, st in ((west, structures[0]), (east, structures[1])):
        for lower in (eb, wb):
            p = shapely.LineString([upper.alignment.offset_point(0.0, 0.0), upper.alignment.offset_point(LENGTH, 0.0)]).intersection(
                shapely.LineString([lower.alignment.offset_point(0.0, 0.0), lower.alignment.offset_point(600.0, 0.0)])
            )
            su, sl = upper.alignment.project(p.x, p.y)[0], lower.alignment.project(p.x, p.y)[0]
            dz = upper.profile.z(su) - lower.profile.z(sl)
            crossings.append(
                Crossing(stable_id("cx", upper.id, lower.id), upper.id, su, lower.id, sl, p.x, p.y, dz, SKEW, TRUTH, clearance_m=dz - DEPTH, structure_id=st.id)
            )
    links = []
    if ramp:
        seam = structures[0].s_end - RAMP_SEAM_BEFORE_END
        x0, y0 = west.alignment.offset_point(seam, RAMP_U0)
        z0, g0 = west.profile.z(seam), west.profile.grade(seam)
        r = _road("ramp", x0, y0, HEADING + RAMP_TURN, 150.0, Profile([VerticalElement(0.0, 150.0, z0, g0, g0, TRUTH)]), RAMP_HALF, "ramp", "motorway_link")
        r.cross_section.barriers = [Barrier(stable_id("br", r.id, side), side, 0.0, 40.0, RAMP_HALF, 1.07, 0.45, 0.25, "parapet", RULE) for side in ("left", "right")]
        roads.append(r)
        links.append(Link(stable_id("lk", west.id, r.id), west.id, "end", r.id, "start", "diverge", from_s=seam, to_s=0.0))
    unit = UnitRecord(
        unit_id="synthetic_twin",
        aoi_lonlat=(-72.003, 41.997, -71.997, 42.003),
        software_version="test",
        lidar=[LidarCollection("synthetic", 2021, "QL1", 8.0, "unit-frame", "NAVD88")],
        vectors=[VectorSourceRecord("synthetic", "test-only", True)],
    )
    frame = Frame(proj="+proj=tmerc +lat_0=42 +lon_0=-72 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m", anchor_lon=-72.0, anchor_lat=42.0)
    return RoadModel(unit=unit, frame=frame, roads=roads, links=links, crossings=crossings, structures=structures)
