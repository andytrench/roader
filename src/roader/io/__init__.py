"""Stages 1-2 inputs: points, vectors, inventory, reference frames. Lane `data`.

Public API:
    PointSet, VectorEdge            input contracts (points.py, vectors.py)
    unit_frame(lon, lat) -> Frame   D-007 working frame; FrameTransformer converts to/from it
    read_points(...) -> (PointSet, LidarCollection)   FR-IN-1..3, FR-PN-1
    read_osm, read_vectors, in_scope                  FR-IN-5, FR-IN-7, spec section 1
    load_unit_config, build_unit_record               FR-IN-2, FR-PL-3, D-009
"""

from roader.io.frame import FrameTransformer, aoi_anchor, unit_frame
from roader.io.points import NOISE_CLASSES, PointSet, from_synthetic, read_points
from roader.io.unit import (
    LidarSourceConfig,
    UnitConfig,
    VectorSourceConfig,
    build_unit_record,
    load_unit_config,
    sha256_file,
    verify_inputs,
)
from roader.io.vectors import (
    VectorEdge,
    in_scope,
    is_limited_access,
    read_osm,
    read_vectors,
)

__all__ = [
    "NOISE_CLASSES",
    "FrameTransformer",
    "LidarSourceConfig",
    "PointSet",
    "UnitConfig",
    "VectorEdge",
    "VectorSourceConfig",
    "aoi_anchor",
    "build_unit_record",
    "from_synthetic",
    "in_scope",
    "is_limited_access",
    "load_unit_config",
    "read_osm",
    "read_points",
    "read_vectors",
    "sha256_file",
    "unit_frame",
    "verify_inputs",
]
