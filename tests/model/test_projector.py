"""One cached projector on the model alignment (D-125): same stations and offsets as the full scan."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roader.model import dumps
from roader.model.geometry import AlignElement, Alignment
from roader.model.synthetic import overpass_model
from tests.fit import scenes


def _scan(al: Alignment, x: float, y: float, step: float = 1.0) -> tuple[float, float]:
    """The original whole-road scan plus Newton refinement (reference)."""
    length = al.length
    count = max(2, math.ceil(length / step) + 1)
    best_s, best_d = 0.0, math.inf
    for i in range(count):
        s = length * i / (count - 1)
        px, py, _h, _k = al.evaluate(s)
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_s, best_d = s, d
    s = best_s
    for _ in range(8):
        px, py, heading, k = al.evaluate(s)
        tx, ty = math.cos(heading), math.sin(heading)
        dx, dy = x - px, y - py
        along, lateral = dx * tx + dy * ty, -dx * ty + dy * tx
        denom = 1.0 - k * lateral
        if abs(denom) < 1e-6:
            break
        ds = along / denom
        s = min(max(s + ds, 0.0), length)
        if abs(ds) < 1e-9:
            break
    px, py, heading, _k = al.evaluate(s)
    return s, -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)


def _alignments():
    m = overpass_model()
    return [m.roads[0].alignment, m.roads[2].alignment, scenes.loop_ramp().alignment, scenes.diverge_merge_scene()[1].alignment]


def _points(al: Alignment, n: int = 60, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.uniform(-20.0, al.length + 20.0, n)
    u = rng.uniform(-25.0, 25.0, n)
    return np.array([al.offset_point(float(a), float(b)) for a, b in zip(s, u, strict=True)])


@pytest.mark.parametrize("k", range(4))
def test_cached_project_matches_the_scan(k):
    al = _alignments()[k]
    for x, y in _points(al):
        s, u = al.project(float(x), float(y))
        rs, ru = _scan(al, float(x), float(y))
        assert abs(s - rs) < 1e-9 and abs(u - ru) < 1e-9


@pytest.mark.parametrize("k", range(4))
def test_project_many_is_project_per_point(k):
    al = _alignments()[k]
    xy = _points(al, seed=2)
    s, u = al.project_many(xy)
    for i, (x, y) in enumerate(xy):
        assert (s[i], u[i]) == al.project(float(x), float(y))


def test_cache_follows_edits_and_stays_out_of_the_model():
    m = overpass_model()
    before = dumps(m)
    al = m.roads[0].alignment
    s0, _u = al.project(10.0, 0.0)
    assert dumps(m) == before
    e = al.elements[0]
    al.elements[0] = AlignElement(e.kind, e.s0, e.length, e.x0 - 50.0, e.y0, e.heading0, e.k0, e.k1, e.prov)
    s1, _u = al.project(10.0, 0.0)
    assert abs((s1 - s0) - 50.0) < 1e-9
