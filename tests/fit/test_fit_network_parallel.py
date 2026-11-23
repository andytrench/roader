"""FR-AF-1, FR-AF-2: roads of one depth fitted in worker processes equal the serial fit (Dallas stack speed)."""

from __future__ import annotations

import numpy as np

from roader.conflate.graph import data_extent, select_roads
from roader.evidence import build_evidence
from roader.fit import links
from roader.fit.median import split_divided
from roader.model.serialize import dumps
from roader.rules import RuleBook
from tests.fit import scenes


def _inputs():
    truth = scenes.diverge_merge_scene()
    pts = scenes.to_pointset(scenes.points(truth))
    seeds = select_roads(scenes.vectors(truth), data_extent(pts.xyz[:, :2]))
    ev = build_evidence(pts, [s.line.xy for s in seeds])
    seeds, _ = split_divided(seeds, ev)
    return seeds, ev


def test_fr_af_1_parallel_fit_network_equals_serial(monkeypatch):
    seeds, ev = _inputs()
    rules = RuleBook.load()
    serial, _js = links.fit_network(seeds, ev, rules, workers=1)
    monkeypatch.setattr(links, "PARALLEL_MIN", 1)
    parallel, _js = links.fit_network(seeds, ev, rules, workers=2)
    for a, b in zip(serial, parallel, strict=True):
        assert a.road.id == b.road.id
        assert dumps(a.road) == dumps(b.road)
        assert np.array_equal(a.vert.pvi, b.vert.pvi)
