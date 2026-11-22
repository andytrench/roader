"""Solid checks, accuracy metrics, gate reports (stage 12). Lane `validate`."""

from roader.validate.accuracy import (
    crossing_dz_accuracy,
    holdout_mask,
    horizontal_accuracy,
    model_crossing_dz,
    structure_station_accuracy,
    surface_accuracy,
)
from roader.validate.assets import check_budgets, check_transitions, check_uv_stretch
from roader.validate.gates import (
    Seam,
    check_continuity,
    check_crossing_order,
    check_provenance,
    check_solids,
    check_voids,
    compare_runs,
    hash_exports,
    run_gates,
)
from roader.validate.lod import LOD_CHORD_TOL, check_lods
from roader.validate.profile import (
    check_pair,
    check_profile,
    check_seam,
    seam_plane_from_line,
)
from roader.validate.realtime import check_realtime
from roader.validate.report import GateReport, GateResult, Violation
from roader.validate.solid import check_solid
from roader.validate.val3dity import find_val3dity, run_val3dity

__all__ = [
    "LOD_CHORD_TOL",
    "GateReport",
    "GateResult",
    "Seam",
    "Violation",
    "check_budgets",
    "check_continuity",
    "check_crossing_order",
    "check_lods",
    "check_pair",
    "check_profile",
    "check_provenance",
    "check_realtime",
    "check_seam",
    "check_solid",
    "check_solids",
    "check_transitions",
    "check_uv_stretch",
    "check_voids",
    "compare_runs",
    "crossing_dz_accuracy",
    "find_val3dity",
    "hash_exports",
    "holdout_mask",
    "horizontal_accuracy",
    "model_crossing_dz",
    "run_gates",
    "run_val3dity",
    "seam_plane_from_line",
    "structure_station_accuracy",
    "surface_accuracy",
]
