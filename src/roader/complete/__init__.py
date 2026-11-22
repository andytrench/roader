"""Stage 8: completing what was not observed. Lane `pipeline`.

FR-OC-1..3: carry alignment, profile and cross-section through occluded gaps;
clearance check against the structure above; engineered ground under bridges
(side slopes, terrain with deck footprints removed).

Public API:
    label_under_deck, split_at_structures, check_clearances, clearance_profile
    side_slopes, terrain_patch
"""

from roader.complete.earthwork import side_slopes, terrain_patch
from roader.complete.occlusion import (
    check_clearances,
    clearance_profile,
    label_under_deck,
    split_at_structures,
)

__all__ = [
    "check_clearances",
    "clearance_profile",
    "label_under_deck",
    "side_slopes",
    "split_at_structures",
    "terrain_patch",
]
