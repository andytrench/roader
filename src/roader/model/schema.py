"""Section 5 data model, layers L0 to L8.

The model is the product. Meshes are regenerated from it (FR-DM-4) and never
edited. All lengths meters, angles radians, coordinates in the unit's working
frame (D-007, FR-DM-5). Provenance is per element (D-010).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roader.model.geometry import Alignment, Piecewise, Profile
from roader.model.provenance import Provenance

# ---------------------------------------------------------------- unit and frame


@dataclass
class Frame:
    """Transverse Mercator centered on the unit; heights NAVD88 (D-007)."""

    proj: str
    anchor_lon: float
    anchor_lat: float
    vertical_datum: str = "NAVD88"
    geoid_model: str = "GEOID18"
    source_crs: str = ""


@dataclass
class LidarCollection:
    """FR-IN-2 record for one collection used in the unit."""

    name: str
    year: int | None
    quality_level: str | None
    density_pavement_pts_m2: float | None
    horizontal_crs: str
    vertical_crs: str
    strip_dz_p95_m: float | None = None  # FR-PN-2
    point_count: int = 0


@dataclass
class VectorSourceRecord:
    """One vector source and its switch (FR-IN-7)."""

    name: str  # "osm" | "tiger" | "overture" | "dot"
    license: str
    enabled: bool
    sha256: str = ""


@dataclass
class UnitRecord:
    unit_id: str
    aoi_lonlat: tuple[float, float, float, float]  # west, south, east, north
    software_version: str
    lidar: list[LidarCollection] = field(default_factory=list)
    vectors: list[VectorSourceRecord] = field(default_factory=list)
    imagery_enabled: bool = False


@dataclass
class Flag:
    """A condition for the review queue (NFR-3) or the unit report (FR-EX-4)."""

    code: str
    element_id: str
    req: str
    message: str


# ---------------------------------------------------------------- L0 network


@dataclass
class VectorRef:
    source: str
    feature_id: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Interval:
    """A station range on a road. kind: observed | occluded | unsupported (FR-VC-2, FR-OC-1)."""

    s0: float
    s1: float
    kind: str
    reason: str = ""


@dataclass
class Link:
    """Directed connection between road ends. Ends are "start" or "end"."""

    id: str
    from_road: str
    from_end: str
    to_road: str
    to_end: str
    kind: str  # continue | diverge | merge
    # Seam stations where the two roads meet (AC-H4, FR-AF-3); None: the named road end.
    from_s: float | None = None
    to_s: float | None = None
    prov: Provenance | None = None


@dataclass
class Crossing:
    """FR-CR-1: two roads crossing in plan, ordered by measurement (FR-CR-2)."""

    id: str
    upper_road: str
    upper_s: float
    lower_road: str
    lower_s: float
    x: float
    y: float
    dz_m: float  # upper surface minus lower surface
    skew_rad: float  # angle between the two alignments at the crossing, 0 = perpendicular
    prov: Provenance
    clearance_m: float | None = None  # lower surface to soffit, filled in stage 9
    tag_upper: str = ""  # vector layer/level hint, logged when it disagrees
    tag_lower: str = ""
    tags_agree: bool | None = None
    structure_id: str | None = None


# ---------------------------------------------------------------- L3 cross-section


@dataclass
class LaneGroup:
    """Lanes on one side of the reference line over [s0, s1] (FR-XS-3: usually inferred)."""

    side: str  # left | right
    s0: float
    s1: float
    count: int
    lane_width_m: float
    shoulder_inner_m: float
    shoulder_outer_m: float
    prov: Provenance


@dataclass
class Median:
    s0: float
    s1: float
    kind: str  # none | painted | raised | depressed | barrier
    width_m: float
    prov: Provenance


@dataclass
class Barrier:
    """Linear raised feature beside the pavement (FR-RS-4) or a parapet on a deck."""

    id: str
    side: str  # left | right
    s0: float
    s1: float
    offset_m: float  # inner face distance from the reference line, positive
    height_m: float
    base_width_m: float
    top_width_m: float
    kind: str  # jersey | f_shape | single_slope | parapet | guardrail | curb
    prov: Provenance


@dataclass
class CrossSection:
    """Pavement extents and slopes along the road.

    width_left / width_right: distance from the reference line to the pavement
    edge, positive meters. slope_left / slope_right: dz/du of the pavement on that
    side, with u positive to the left. A normal 2% crown is slope_left = -0.02,
    slope_right = +0.02; a section banked up to the left has both positive.
    """

    width_left: Piecewise
    width_right: Piecewise
    slope_left: Piecewise
    slope_right: Piecewise
    pavement_thickness: Piecewise
    lanes: list[LaneGroup] = field(default_factory=list)
    medians: list[Median] = field(default_factory=list)
    barriers: list[Barrier] = field(default_factory=list)


@dataclass
class Road:
    """One carriageway, ramp or connector with its L1 to L3 description."""

    id: str
    name: str
    road_class: str  # motorway | motorway_link | trunk | primary | ...
    kind: str  # mainline | ramp | connector | crossing_street
    oneway: bool
    alignment: Alignment
    profile: Profile
    cross_section: CrossSection
    vector_refs: list[VectorRef] = field(default_factory=list)
    support: list[Interval] = field(default_factory=list)
    sheet_ids: list[str] = field(default_factory=list)
    prov: Provenance | None = None  # whole-road source and fit stats (AC-H6); None: derive from parameters

    @property
    def length(self) -> float:
        return self.alignment.length

    def surface_z(self, s: float, u: float) -> float:
        """Top of pavement at station s, lateral offset u."""
        xs = self.cross_section
        slope = xs.slope_left(s) if u >= 0.0 else xs.slope_right(s)
        return self.profile.z(s) + slope * u


# ---------------------------------------------------------------- L4 earthwork


@dataclass
class SideSlope:
    """Ties a pavement edge to terrain (L4). ratio is horizontal run per unit rise."""

    id: str
    road_id: str
    side: str  # left | right
    s0: float
    s1: float
    kind: str  # fill | cut | wall | none
    ratio: float
    prov: Provenance


# ---------------------------------------------------------------- L5 structures


@dataclass
class DeckSpec:
    depth_m: float  # top of deck to soffit, superstructure included
    slab_thickness_m: float
    overhang_m: float
    prov: Provenance


@dataclass
class Support:
    id: str
    station: float  # on the carried road
    kind: str  # abutment | pier
    skew_rad: float  # FR-ST-4: rotation from perpendicular to the carried road
    n_columns: int
    column_width_m: float
    cap_depth_m: float
    cap_width_m: float
    footing_z: float  # bottom elevation of the support
    prov: Provenance
    thickness_m: float | None = None  # abutment thickness along the road (FR-ST-3); None: cap_width_m (D-026)


@dataclass
class Structure:
    id: str
    kind: str  # bridge | culvert | tunnel_portal
    carried_road: str
    s_start: float  # FR-VC-4: measured ends
    s_end: float
    family: str  # slab | multi_girder | box_girder | truss | arch | culvert
    year_built: int | None
    crossed_roads: list[str]
    deck: DeckSpec
    supports: list[Support]
    prov: Provenance
    inventory_id: str | None = None
    review: bool = False  # FR-ST-6


# ---------------------------------------------------------------- L6, L7


@dataclass
class JunctionSeam:
    """Where a junction area meets a road sweep: station s, offsets u0 < u1 on that road (FR-SG-3, AC-H4)."""

    road_id: str
    s: float
    u0: float
    u1: float


@dataclass
class JunctionArea:
    """L6 gore / merge / intersection, built from a planar partition of lane and edge lines (FR-SG-3).

    ``outline`` is the plan polygon (counter-clockwise, unit frame). Its surface is
    lifted from the adjoining roads, so elevation and slope match at every seam.
    """

    id: str
    road_ids: list[str]
    outline: list[tuple[float, float]]
    prov: Provenance
    kind: str = "gore"  # gore | merge | diverge | intersection
    seams: list[JunctionSeam] = field(default_factory=list)
    holes: list[list[tuple[float, float]]] = field(default_factory=list)


@dataclass
class TerrainPatch:
    """L7 bare-earth grid in the unit frame; None where unobserved.

    Solids re-triangulate it with road and structure footprints as constraint
    edges (FR-SG-6). Row-major, row 0 at origin_y, column 0 at origin_x.
    """

    origin_x: float
    origin_y: float
    cell_m: float
    nx: int
    ny: int
    z: list[float | None]
    prov: Provenance


# ---------------------------------------------------------------- the model


@dataclass
class RoadModel:
    unit: UnitRecord
    frame: Frame
    roads: list[Road] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    structures: list[Structure] = field(default_factory=list)
    side_slopes: list[SideSlope] = field(default_factory=list)
    junctions: list[JunctionArea] = field(default_factory=list)
    terrain: TerrainPatch | None = None
    flags: list[Flag] = field(default_factory=list)

    def road(self, road_id: str) -> Road:
        for road in self.roads:
            if road.id == road_id:
                return road
        raise KeyError(road_id)

    def structure(self, structure_id: str) -> Structure:
        for item in self.structures:
            if item.id == structure_id:
                return item
        raise KeyError(structure_id)
