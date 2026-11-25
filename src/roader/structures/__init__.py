"""Stage 9: structure inference. Lane `pipeline`.

FR-ST-2..6 (FR-ST-1 inventory matching waits for the NBI reader): family from
span, rule-sized deck, abutments at measured ends, observed piers then piers by
the layout rule, supports aligned with the crossing skew, review queue; decks
shared between roads (FR-CR-3).

Public API:
    infer_structure(extent, upper_fit, crossings, fits, evidence, rules, drop, barriers) -> StructureResult
    parapet_fill(road, structure, existing_barriers, rules) -> list[Barrier]
    twin_joints(structures, roads, crossings, links) -> list[Flag]
    attach_shared(structures, shares, roads, barriers, crossings) -> list[Flag]
    open_ends(structures, roads, links) -> list[Flag]
"""

from roader.structures.infer import (
    StructureResult,
    infer_structure,
    open_ends,
    parapet_fill,
)
from roader.structures.shared import attach_shared, twin_joints

__all__ = ["StructureResult", "attach_shared", "infer_structure", "open_ends", "parapet_fill", "twin_joints"]
