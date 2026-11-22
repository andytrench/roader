"""Stage 2 normalization metrics and stage 3 road-surface evidence. Lane `pipeline`.

FR-PN-2, FR-PN-3, FR-RS-1..4: density grid, strip disagreement, pavement
probability, surface sheets, transient removal, raised linear features.

Public API:
    build_evidence(points, lines) -> Evidence
    Evidence: per point (pavement probability, transient flag, sheet), per layer-cell
    (Layers), raised features, sheets, strip and density reports.
"""

from roader.evidence.build import (
    PAVEMENT_P,
    DensityReport,
    Evidence,
    SheetInfo,
    build_evidence,
    corridor_mask,
)
from roader.evidence.features import RaisedFeature
from roader.evidence.layers import Layers, build_layers, build_sheets
from roader.evidence.strips import StripReport, correct_strips, strip_disagreement

__all__ = [
    "PAVEMENT_P",
    "DensityReport",
    "Evidence",
    "Layers",
    "RaisedFeature",
    "SheetInfo",
    "StripReport",
    "build_evidence",
    "build_layers",
    "build_sheets",
    "correct_strips",
    "corridor_mask",
    "strip_disagreement",
]
