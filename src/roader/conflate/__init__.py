"""Stages 4-5: vector conflation, registration, structure extents, crossings. Lane `pipeline`.

FR-VC-1..5, FR-CR-1..3.

Public API:
    select_roads(edges, extent) -> list[RoadSeed]          FR-VC-1, spec section 1 scope
    track_line(line, evidence) -> Track                    FR-VC-2 sheet assignment, observed/occluded/unsupported
    cross_profiles, register_line                          FR-VC-3 lateral registration, FR-XS-1 edges
    find_extents(fit, evidence, skews) -> [StructureExtent] FR-VC-4 structure ends from data
    find_crossings(fits) -> (crossings, measures, flags)   FR-CR-1, FR-CR-2
    Polyline                                               vectorized station/offset geometry
"""

from roader.conflate.graph import RoadSeed, data_extent, select_roads
from roader.conflate.polyline import Polyline
from roader.conflate.register import Profiles, cross_profiles, near_points, register_line
from roader.conflate.track import Track, track_line

__all__ = [
    "Polyline",
    "Profiles",
    "RoadSeed",
    "Track",
    "cross_profiles",
    "data_extent",
    "near_points",
    "register_line",
    "select_roads",
    "track_line",
]
