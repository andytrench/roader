"""Many-point station/offset projection, the batch form of ``Alignment.project``.

Delegates to the alignment's cached projector (D-125): one KD-tree over the 1 m
samples seeds the same Newton refinement as ``Alignment.project``.
"""

from __future__ import annotations

import numpy as np

from roader.model.geometry import Alignment


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
