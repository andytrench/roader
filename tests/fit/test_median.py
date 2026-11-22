"""FR-XS-4: median classification and twin carriageways with independent profiles.

Divided highways are mapped either as two one-way OSM ways (each carriageway its
own road) or as one two-way way down the median. Both must end as two roads with
their own L2 profiles, and both carry a Median element of the measured kind.
"""

from __future__ import annotations

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.io import VectorEdge
from roader.model.provenance import Source
from roader.model.synthetic import overpass_model
from roader.pipeline import run_pipeline
from tests.fit import scenes

REF = overpass_model()


def _run(roads, vectors=None, **kw):
    sp = scenes.points(roads, **kw)
    vec = vectors if vectors is not None else scenes.vectors(roads)
    return run_pipeline(scenes.to_pointset(sp), vec, REF.frame, REF.unit).model


def _road(model, fid):
    return next(r for r in model.roads if any(ref.feature_id == fid for ref in r.vector_refs))


def _medians_cover(road, kind, lo=60.0, hi=None):
    """The road's medians, all of ``kind``, cover [lo, length - lo]."""
    hi = road.length - lo if hi is None else hi
    meds = road.cross_section.medians
    assert meds, road.name
    assert {m.kind for m in meds} == {kind}, [(m.kind, m.s0, m.s1) for m in meds]
    covered = sum(min(m.s1, hi) - max(m.s0, lo) for m in meds if m.s1 > lo and m.s0 < hi)
    assert covered >= 0.95 * (hi - lo)
    return meds


@pytest.fixture(scope="module")
def depressed():
    roads = scenes.twin_carriageways(median=14.0, dz=1.2)
    return roads, _run(roads)


def test_fr_xs_4_depressed_median_between_two_ways(depressed):
    _roads, model = depressed
    for fid in ("eb", "wb"):
        road = _road(model, fid)
        meds = _medians_cover(road, "depressed")
        for m in meds:
            assert abs(m.width_m - 14.0) < 0.6
            assert m.prov.source == Source.MEASURED and m.prov.n_points > 0
            assert "FR-XS-4" in m.prov.flags


def test_fr_xs_4_twin_profiles_independent(depressed):
    """The westbound carriageway runs 1.2 m higher; each profile follows its own returns."""
    roads, model = depressed
    for t in roads:
        road = _road(model, t.vector_refs[0].feature_id)
        tl = Polyline.from_alignment(t.alignment)
        s = np.arange(20.0, road.length - 20.0, 10.0)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        ts, _u, ok = tl.project(xy)
        dz = [road.profile.z(float(a)) - t.profile.z(float(b)) for a, b in zip(s[ok], ts[ok], strict=True)]
        assert np.max(np.abs(dz)) < 0.05


def test_fr_xs_4_barrier_median():
    roads = scenes.twin_carriageways(median=0.6, dz=0.0)
    extra, mask = scenes.median_band(*roads, "barrier")
    model = _run(roads, extra=extra, no_terrain=mask)
    for fid in ("eb", "wb"):
        _medians_cover(_road(model, fid), "barrier")


def test_fr_xs_4_painted_median():
    roads = scenes.twin_carriageways(median=3.0, dz=0.0)
    extra, mask = scenes.median_band(*roads, "painted")
    model = _run(roads, extra=extra, no_terrain=mask)
    for fid in ("eb", "wb"):
        meds = _medians_cover(_road(model, fid), "painted")
        assert all(abs(m.width_m - 3.0) < 0.6 for m in meds)


def test_fr_xs_4_one_way_divided_highway_is_split_into_twins():
    """One two-way way down the median of a divided expressway: two roads, independent profiles, logged (L0)."""
    roads = scenes.twin_carriageways(median=14.0, dz=1.2)
    line = np.column_stack([np.linspace(-300.0, 300.0, 61), np.zeros(61)])
    vec = [VectorEdge("synthetic", "div", line, {"highway": "trunk", "expressway": "yes", "oneway": "no", "ref": "US 9"})]
    model = _run(roads, vectors=vec)
    twins = [r for r in model.roads if any(ref.feature_id == "div" for ref in r.vector_refs)]
    assert len(twins) == 2
    assert all(r.oneway and r.kind == "mainline" for r in twins)
    assert any(f.code == "FR-XS-4:split" for f in model.flags)
    for t in roads:
        tl = Polyline.from_alignment(t.alignment)
        road = min(twins, key=lambda r: abs(tl.project(np.array([r.alignment.offset_point(r.length / 2, 0.0)]))[1][0]))
        h_t = t.alignment.evaluate(0.0)[2]
        h_r = road.alignment.evaluate(0.0)[2]
        assert np.cos(h_t - h_r) > 0.99  # each twin runs in its own direction of travel
        s = np.arange(20.0, road.length - 20.0, 10.0)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        ts, u, ok = tl.project(xy)
        assert np.max(np.abs(u[ok])) < 0.3
        dz = [road.profile.z(float(a)) - t.profile.z(float(b)) for a, b in zip(s[ok], ts[ok], strict=True)]
        assert np.max(np.abs(dz)) < 0.05
        _medians_cover(road, "depressed")


def test_fr_xs_4_m1_scene_median():
    """The shared synthetic overpass: twin carriageways 7 m apart on fill, so the median is depressed."""
    from roader.io import from_synthetic
    from roader.model.synthetic import overpass_points
    from tests.pipeline.conftest import by_feature, synthetic_vectors

    truth = overpass_model()
    res = run_pipeline(from_synthetic(overpass_points(truth)), synthetic_vectors(truth), truth.frame, truth.unit)
    for fid in ("eb", "wb"):
        meds = by_feature(res.model, fid).cross_section.medians
        assert meds and {m.kind for m in meds} == {"depressed"}
        assert all(abs(m.width_m - 7.0) < 0.6 for m in meds)
    assert not by_feature(res.model, "over").cross_section.medians
