"""Section 5 data model, layers L0 to L8.

The model is the product. Meshes are regenerated from it (FR-DM-4) and never
edited. All lengths meters, angles radians, coordinates in the unit's working
frame (D-007, FR-DM-5). Provenance is per element (D-010).
"""

from __future__ import annotations

import math
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


# An element replaced or dropped because building it raised (FR-EX-1, D-119): the unit is held.
ELEMENT_FAILED = "FR-EX-1:element_failed"


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
    # Per-side overhang where one side differs, e.g. the joint side of a twin deck (FR-CR-3); None: overhang_m.
    overhang_left_m: float | None = None
    overhang_right_m: float | None = None
    # Per-station overhang on one side, e.g. a twin joint following the gap (FR-CR-3, D-134); None: overhang(side).
    overhang_left_at: Piecewise | None = None
    overhang_right_at: Piecewise | None = None

    def overhang(self, side: str) -> float:
        """One value per side; with a per-station overhang, its largest (the deck's outer reach)."""
        profile = self.overhang_left_at if side == "left" else self.overhang_right_at
        if profile is not None:
            return max(k.value for k in profile.knots)
        value = self.overhang_left_m if side == "left" else self.overhang_right_m
        return self.overhang_m if value is None else value

    def overhang_at(self, side: str, s: float) -> float:
        profile = self.overhang_left_at if side == "left" else self.overhang_right_at
        return self.overhang(side) if profile is None else profile(s)


@dataclass
class SharedSpan:
    """FR-CR-3: a stretch of another road that rides on this structure's deck (a ramp diverging on it)."""

    road_id: str
    s_start: float
    s_end: float


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
    # Other roads carried over part of their length; the deck is their surface there (FR-CR-3).
    shared: list[SharedSpan] = field(default_factory=list)


# ---------------------------------------------------------------- L6, L7


@dataclass
class JunctionSeam:
    """Where a junction area meets a road sweep: station s, offsets u0 < u1 on that road (FR-SG-3, AC-H4).

    The remaining fields lift the road's surface at the seam into a local patch
    (D-046), so the area's surface is a function of the stored model alone: plan
    reference point and heading of the road at s, its curvature, elevation, grade
    and grade rate, cross slopes and their rates along s, pavement thickness, all
    taken on the junction side of s. ``z`` None: written before the patch existed.
    """

    road_id: str
    s: float
    u0: float
    u1: float
    side: int = 0  # +1: the area lies at stations above s, -1 below, 0 unknown
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    z: float | None = None
    grade: float = 0.0
    grade_rate: float = 0.0
    slope_left: float = 0.0
    slope_right: float = 0.0
    slope_left_rate: float = 0.0
    slope_right_rate: float = 0.0
    thickness: float = 0.0
    # tan of the seam row's skew: the row runs through station s + skew * u (a deck's skewed end, D-073).
    skew: float = 0.0

    def local(self, x: float, y: float) -> tuple[float, float]:
        """(station, offset) of a plan point relative to the seam, on the osculating circle of the road."""
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.heading), math.sin(self.heading)
        k = self.curvature
        if abs(k) < 1e-9:
            return dx * c + dy * s, -dx * s + dy * c
        # Centre of curvature at P + n / k; a point at (t, u) is C - (1/k - u) n(t).
        cx, cy = -s / k, c / k
        vx, vy = dx - cx, dy - cy
        r = math.hypot(vx, vy)
        sign = 1.0 if k > 0.0 else -1.0
        u = 1.0 / k - sign * r
        if r < 1e-12:
            return 0.0, u
        nx_, ny_ = -sign * vx / r, -sign * vy / r  # left normal at the point's station
        ang = math.atan2(-s * ny_ - c * nx_, -s * nx_ + c * ny_)  # rotation from the seam's normal (-s, c)
        return ang / k, u

    def z_at(self, x: float, y: float) -> float:
        """Top of pavement on the seam's patch."""
        t, u = self.local(x, y)
        assert self.z is not None
        slope = self.slope_left + self.slope_left_rate * t if u >= 0.0 else self.slope_right + self.slope_right_rate * t
        return self.z + self.grade * t + 0.5 * self.grade_rate * t * t + slope * u

    def distance(self, x: float, y: float) -> float:
        """Blend distance to the section line from u0 to u1: along the road, plus lateral overshoot
        weighted by JUNCTION_LATERAL_WEIGHT, so a patch speaks mainly for its own pavement (D-046)."""
        t, u = self.local(x, y)
        excess = max(self.u0 - u, 0.0, u - self.u1)
        return math.hypot(t, JUNCTION_LATERAL_WEIGHT * excess)


JUNCTION_IDW_POWER = 3.0  # D-046: weights 1/d^3 give each seam its own value and slope
JUNCTION_LATERAL_WEIGHT = 4.0
JUNCTION_EDGE_TOL_M = 0.01  # surface_z answers this far outside the outline


def _in_ring(ring: list[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    n = len(ring)
    for k in range(n):
        (x1, y1), (x2, y2) = ring[k], ring[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _ring_distance(ring: list[tuple[float, float]], x: float, y: float) -> float:
    best = math.inf
    n = len(ring)
    for k in range(n):
        (x1, y1), (x2, y2) = ring[k], ring[(k + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        den = ex * ex + ey * ey
        f = 0.0 if den <= 0.0 else min(max(((x - x1) * ex + (y - y1) * ey) / den, 0.0), 1.0)
        best = min(best, math.hypot(x - x1 - f * ex, y - y1 - f * ey))
    return best


@dataclass
class JunctionArea:
    """L6 gore / merge / intersection, built from a planar partition of lane and edge lines (FR-SG-3).

    ``outline`` is the plan polygon (counter-clockwise, unit frame). Its surface is
    lifted from the adjoining roads, so elevation and slope match at every seam:
    an inverse-distance blend of the seam patches and of lifted interior
    ``sections`` of the roads it covers (D-046).
    """

    id: str
    road_ids: list[str]
    outline: list[tuple[float, float]]
    prov: Provenance
    kind: str = "gore"  # gore | merge | diverge | intersection
    seams: list[JunctionSeam] = field(default_factory=list)
    holes: list[list[tuple[float, float]]] = field(default_factory=list)
    sections: list[JunctionSeam] = field(default_factory=list)  # lifted road sections inside the area (D-046)

    def contains(self, x: float, y: float, tol: float = JUNCTION_EDGE_TOL_M) -> bool:
        rings = [list(self.outline), *[list(h) for h in self.holes]]
        if len(rings[0]) < 3:
            return False
        inside = _in_ring(rings[0], x, y) and not any(_in_ring(h, x, y) for h in rings[1:])
        return inside or min(_ring_distance(r, x, y) for r in rings) <= tol or self._in_skew_wedge(x, y)

    def _in_skew_wedge(self, x: float, y: float) -> bool:
        """Between a skewed seam row and the straight section at its station: the seam lies on the skewed row,
        and the straight section of the road there is read from the area (D-073)."""
        for sm in self.seams:
            if sm.skew == 0.0 or sm.z is None:
                continue
            t, u = sm.local(x, y)
            if sm.u0 - JUNCTION_EDGE_TOL_M <= u <= sm.u1 + JUNCTION_EDGE_TOL_M:
                lo, hi = sorted((0.0, sm.skew * u))
                if lo - JUNCTION_EDGE_TOL_M <= t <= hi + JUNCTION_EDGE_TOL_M:
                    return True
        return False

    def _blend(self, x: float, y: float, value) -> float:
        patches = [*self.seams, *self.sections]
        if not patches or any(p.z is None for p in patches):
            raise ValueError(f"junction {self.id}: no lifted surface (L6 patches, D-046)")
        num = den = 0.0
        for p in patches:
            d = p.distance(x, y)
            if d < 1e-9:
                return value(p)
            w = d ** -JUNCTION_IDW_POWER
            num += w * value(p)
            den += w
        return num / den

    def lifted_z(self, x: float, y: float) -> float:
        """Top of pavement from the patches, without the outline test."""
        return self._blend(x, y, lambda p: p.z_at(x, y))

    def thickness_at(self, x: float, y: float) -> float:
        return self._blend(x, y, lambda p: p.thickness)

    def surface_z(self, x: float, y: float) -> float:
        """Top of pavement inside the area; NaN outside it (validate.seams contract, AC-H4)."""
        if not self.contains(x, y):
            return math.nan
        return self.lifted_z(x, y)


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
    # Per-cell origin of z, same order as z (FR-OC-3, AC-H6). Empty means all
    # measured. 0 measured, 1 gap interpolated, 2 under a structure, 3 under a
    # road or embankment footprint, 4 lowered below a non-ground solid.
    fill: list[int] = field(default_factory=list)
    id: str = "terrain"  # element id of the L7 meshes (FR-DM-3)


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
