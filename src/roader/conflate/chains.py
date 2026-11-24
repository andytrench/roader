"""Chained roads split where they would cross themselves (FR-VC-1, FR-CR-1, D-162).

A trumpet's semi-direct ramp chains its approach, the flyover and the loop that passes
back under the flyover. As one road it crosses itself: the extent clip cuts it at the
crossing and keeps only the longest piece, and the crossing cannot be recorded, since a
crossing is between two roads. Splitting the chain at a way boundary keeps every OSM way
whole in one road, so references by way id still resolve to a single road.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely


def split_self_crossings(chain: list[int], merge: Callable[[list[int]], np.ndarray]) -> list[list[int]]:
    """``chain`` (edges in order of travel) cut into runs whose merged centerlines are simple.

    ``merge`` returns the merged coordinates of a run. A run grows one edge at a time; the
    edge that would make it cross itself starts the next run. A single edge that crosses
    itself stays as it is.
    """
    runs: list[list[int]] = []
    run: list[int] = []
    for edge in chain:
        if run and not shapely.LineString(merge([*run, edge])).is_simple:
            runs.append(run)
            run = []
        run.append(edge)
    runs.append(run)
    return runs
