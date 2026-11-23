"""FR-XS-1, FR-XS-4, D-063: twin carriageways never overlap across their median.

Two one-way carriageways 3 m apart: with a median barrier each inner edge stops at
the barrier face on its side; with a flush (painted) median, where the measured
pavement runs on across the median, each stops at the line midway between them.
"""

from __future__ import annotations

import numpy as np
import pytest

from roader.conflate.polyline import Polyline
from roader.model.synthetic import overpass_model
from roader.pipeline import run_pipeline
from tests.fit import scenes

REF = overpass_model()
MEDIAN_M = 3.0


def _run(kind):
    roads = scenes.twin_carriageways(median=MEDIAN_M, dz=0.0)
    extra, no_terrain = scenes.median_band(*roads, kind)
    sp = scenes.points(roads, extra=extra, no_terrain=no_terrain)
    model = run_pipeline(scenes.to_pointset(sp), scenes.vectors(roads), REF.frame, REF.unit).model
    eb = next(r for r in model.roads if r.vector_refs[0].feature_id == "eb")
    wb = next(r for r in model.roads if r.vector_refs[0].feature_id == "wb")
    return roads, model, eb, wb


def _facing(eb, wb):
    """(station, eb inner edge, wb inner edge) as y coordinates along the straight twins."""
    leb, lwb = Polyline.from_alignment(eb.alignment), Polyline.from_alignment(wb.alignment)
    out = []
    for s in np.arange(60.0, eb.length - 60.0, 10.0):
        y_eb = leb.offset_point(np.array([s]), np.array([eb.cross_section.width_left(float(s))]))[0, 1]
        x = leb.point(np.array([s]))[0, 0]
        sw, _u, _ok = lwb.project(np.array([[x, 0.0]]))
        y_wb = lwb.offset_point(sw, np.array([wb.cross_section.width_left(float(sw[0]))]))[0, 1]
        out.append((s, y_eb, y_wb))
    return np.array(out)


@pytest.fixture(scope="module", params=["barrier", "painted"])
def twin(request):
    return request.param, _run(request.param)


def test_fr_xs_4_twin_inner_edges_do_not_overlap(twin):
    _kind, (_roads, _model, eb, wb) = twin
    rows = _facing(eb, wb)
    assert np.all(rows[:, 1] <= rows[:, 2] + 1e-6), rows[rows[:, 1] > rows[:, 2]]


def test_fr_xs_1_inner_edge_stops_at_the_barrier_face(twin):
    kind, (roads, _model, eb, wb) = twin
    if kind != "barrier":
        pytest.skip("flush median")
    del roads
    rows = _facing(eb, wb)
    # The barrier runs midway between the twins with its faces at -/+ 0.28 m.
    assert np.all(rows[:, 1] <= -0.28 + 0.05) and np.all(rows[:, 2] >= 0.28 - 0.05)


def test_fr_xs_4_flush_median_split_at_the_midline(twin):
    kind, (_roads, _model, eb, wb) = twin
    if kind != "painted":
        pytest.skip("barrier median")
    rows = _facing(eb, wb)
    # Each edge on its own side of the midline, and not further back than the truth edge.
    assert np.all(rows[:, 1] <= 0.05) and np.all(rows[:, 1] >= -1.5 - 0.4)
    assert np.all(rows[:, 2] >= -0.05) and np.all(rows[:, 2] <= 1.5 + 0.4)


# ---------------------------------------------------------------- the split itself


def _overlapping(width):
    """The twins with each inner width ``width`` (more than the 7.0 m to the midline when overlapping)."""
    from types import SimpleNamespace

    roads = scenes.twin_carriageways(median=MEDIAN_M, dz=0.0)
    fits = []
    for r in roads:
        r.cross_section.width_left = scenes.piecewise([(0.0, width)])
        fits.append(SimpleNamespace(road=r, line=Polyline.from_alignment(r.alignment), flags=[]))
    return fits


def _inner_y(fits):
    eb, wb = fits[0].road, fits[1].road
    return eb.alignment.offset_point(300.0, eb.cross_section.width_left(300.0))[1], wb.alignment.offset_point(300.0, wb.cross_section.width_left(300.0))[1]


def test_fr_xs_4_overlapping_twins_split_at_the_midline():
    from roader.fit.median import split_twin_medians

    fits = _overlapping(9.0)
    flags = split_twin_medians(fits, np.zeros((0, 2)))
    y_eb, y_wb = _inner_y(fits)
    assert abs(y_eb) < 0.05 and abs(y_wb) < 0.05
    assert any(f.code == "FR-XS-4:median_split" for f in flags)
    # Outer edges untouched.
    assert fits[0].road.cross_section.width_right(300.0) == pytest.approx(5.5)


def test_fr_xs_1_twins_stop_at_the_median_barrier_face():
    from roader.fit.median import split_twin_medians

    fits = _overlapping(9.0)
    x = np.arange(-295.0, 295.0, 0.3)
    barrier = np.vstack([np.column_stack([x, np.full(x.size, dy)]) for dy in (-0.28, -0.1, 0.1, 0.28)])
    split_twin_medians(fits, barrier)
    y_eb, y_wb = _inner_y(fits)
    assert abs(y_eb + 0.28) < 0.05 and abs(y_wb - 0.28) < 0.05


def test_fr_xs_4_twins_short_of_the_median_are_left_alone():
    from roader.fit.median import split_twin_medians

    fits = _overlapping(6.0)
    assert split_twin_medians(fits, np.zeros((0, 2))) == []
    assert fits[0].road.cross_section.width_left(300.0) == pytest.approx(6.0)


def test_fr_xs_4_twins_at_different_levels_are_not_split():
    """Over 1 m apart in height (D-075) the facing pavements are not one median."""
    from roader.fit.median import split_twin_medians

    fits = _overlapping(9.0)
    for k in fits[1].road.profile.elements:
        k.z0 += 3.0
    assert split_twin_medians(fits, np.zeros((0, 2))) == []
