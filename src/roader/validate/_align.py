"""Many-point station/offset projection, the batch form of ``Alignment.project``.

Delegates to the alignment's cached projector (D-125): one KD-tree over the 1 m
samples seeds the same Newton refinement as ``Alignment.project``.
"""

from __future__ import annotations

import numpy as np

from roader.model.geometry import Alignment

_CACHE: dict[int, Projector] = {}
CACHE_MAX = 1024


def projector_for(alignment: Alignment) -> Projector:
    """One shared ``Projector`` per alignment object, for callers projecting a few points many times."""
    p = _CACHE.get(id(alignment))
    if p is None or p.alignment is not alignment:
        if len(_CACHE) >= CACHE_MAX:
            _CACHE.clear()
        p = _CACHE[id(alignment)] = Projector(alignment)
    return p


class Projector:
    def __init__(self, alignment: Alignment, step: float = 1.0) -> None:
        self.alignment = alignment
        self.length = alignment.length
        self._step = step
        alignment._samples(step)

    def near(self, xy: np.ndarray, max_dist: float) -> np.ndarray:
        """Points whose nearest sample is within max_dist + one sample spacing."""
        return self.alignment.near(xy, max_dist, self._step)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Station and signed offset (u positive left) for each point."""
        return self.alignment.project_many(xy, self._step)
