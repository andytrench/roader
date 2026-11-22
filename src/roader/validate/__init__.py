"""Solid checks, accuracy metrics, gate reports (stage 12). Lane `validate`."""

from roader.validate.accuracy import (
    crossing_dz_accuracy,
    holdout_mask,
    horizontal_accuracy,
    model_crossing_dz,
    structure_station_accuracy,
    surface_accuracy,
)
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
from roader.validate.profile import (
    check_pair,
    check_profile,
    check_seam,
    seam_plane_from_line,
)
from roader.validate.report import GateReport, GateResult, Violation
from roader.validate.solid import check_solid
from roader.validate.val3dity import find_val3dity, run_val3dity

__all__ = [
    "GateReport",
    "GateResult",
    "Seam",
    "Violation",
    "check_continuity",
    "check_crossing_order",
    "check_pair",
    "check_profile",
    "check_provenance",
    "check_seam",
    "check_solid",
    "check_solids",
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
