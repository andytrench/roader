"""Stage 9: structure inference. Lane `pipeline`.

FR-ST-2..6 (FR-ST-1 inventory matching waits for the NBI reader): family from
span, rule-sized deck, abutments at measured ends, piers by the layout rule,
supports aligned with the crossing skew, review queue.

Public API:
    infer_structure(extent, upper_fit, crossings, fits, evidence, rules, drop, barriers) -> StructureResult
    parapet_fill(road, structure, existing_barriers, rules) -> list[Barrier]
"""

from roader.structures.infer import StructureResult, infer_structure, parapet_fill

__all__ = ["StructureResult", "infer_structure", "parapet_fill"]
