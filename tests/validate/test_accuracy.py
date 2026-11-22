"""AC-A1, AC-A3, AC-A4, AC-A5 and the spatial block holdout."""

from __future__ import annotations

import numpy as np
import pytest

from roader.model.synthetic import overpass_model, overpass_points
from roader.validate.accuracy import (
    crossing_dz_accuracy,
    holdout_mask,
    horizontal_accuracy,
    model_crossing_dz,
    structure_station_accuracy,
    surface_accuracy,
)


@pytest.fixture(scope="module")
def pavement_xyz():
    pts = overpass_points()
    return pts.xyz[np.isin(pts.truth, ["pavement", "deck"])]


def test_holdout_is_deterministic_and_about_a_fifth_of_area(pavement_xyz):
    xy = pavement_xyz[:, :2]
    mask = holdout_mask(xy)
    assert np.array_equal(mask, holdout_mask(xy.copy()))
    # Area share: occupied 1 m cells in held-out blocks over all occupied cells.
    cells = np.unique(np.floor(xy).astype(np.int64), axis=0)
    held_cells = np.unique(np.floor(xy[mask]).astype(np.int64), axis=0)
    share = len(held_cells) / len(cells)
    assert 0.17 <= share <= 0.30
    # Whole blocks: no 25 m block is split.
    blocks = np.floor(xy / 25.0).astype(np.int64)
    key = blocks[:, 0] * 100003 + blocks[:, 1]
    for k in np.unique(key[mask])[:20]:
        assert mask[key == k].all()


def test_holdout_split_does_not_depend_on_point_order(pavement_xyz):
    xy = pavement_xyz[:, :2]
    perm = np.random.default_rng(1).permutation(len(xy))
    assert np.array_equal(holdout_mask(xy)[perm], holdout_mask(xy[perm]))


def test_ac_a1_truth_model_passes(pavement_xyz):
    result = surface_accuracy(overpass_model(), pavement_xyz)
    assert result.passed, result.metrics
    assert 0.025 <= result.metrics["rms_m"] <= 0.045
    assert result.metrics["n_points"] > 1000
    assert not result.hard


def test_ac_a1_shifted_profile_fails(pavement_xyz):
    model = overpass_model()
    for road in model.roads:
        for el in road.profile.elements:
            el.z0 += 0.3
    result = surface_accuracy(model, pavement_xyz)
    assert not result.passed
    assert result.metrics["rms_m"] > 0.25


def test_ac_a1_no_points_is_skipped():
    result = surface_accuracy(overpass_model(), np.empty((0, 3)))
    assert not result.passed and result.skipped_reason


def _edge_points(model, offset: float = 0.0):
    out = []
    for road in model.roads:
        for s in np.arange(20.0, road.length - 20.0, 10.0):
            for side, sign in (("left", 1.0), ("right", -1.0)):
                w = road.cross_section.width_left(s) if side == "left" else road.cross_section.width_right(s)
                x, y = road.alignment.offset_point(float(s), sign * (w + offset))
                out.append({"road_id": road.id, "side": side, "x": x, "y": y})
    return out


def test_ac_a3_edges_on_model_pass_and_offset_fails():
    model = overpass_model()
    good = horizontal_accuracy(model, _edge_points(model, 0.1))
    assert good.passed and good.metrics["rms_m"] == pytest.approx(0.1, abs=1e-6)
    bad = horizontal_accuracy(model, _edge_points(model, 0.8))
    assert not bad.passed


def test_ac_a3_deck_outline_uses_overhang():
    model = overpass_model()
    st = model.structures[0]
    road = model.road(st.carried_road)
    pts = []
    for s in np.linspace(st.s_start + 2.0, st.s_end - 2.0, 10):
        w = road.cross_section.width_left(s) + st.deck.overhang_m
        x, y = road.alignment.offset_point(float(s), w)
        pts.append({"structure_id": st.id, "side": "left", "x": x, "y": y})
    assert horizontal_accuracy(model, pts).metrics["rms_m"] < 1e-6


def test_ac_a4_crossing_dz():
    model = overpass_model()
    truth = model_crossing_dz(model)
    assert truth[model.crossings[0].id] == pytest.approx(model.crossings[0].dz_m, abs=1e-6)
    assert crossing_dz_accuracy(model, {k: v + 0.1 for k, v in truth.items()}).passed
    bad = crossing_dz_accuracy(model, {k: v + 0.2 for k, v in truth.items()})
    assert not bad.passed and len(bad.violations) == 2


def test_ac_a5_structure_stations():
    model = overpass_model()
    st = model.structures[0]
    assert structure_station_accuracy(model, {st.id: (st.s_start + 1.5, st.s_end - 1.0)}).passed
    bad = structure_station_accuracy(model, {st.id: (st.s_start - 2.5, st.s_end)})
    assert not bad.passed
    assert "start" in bad.violations[0].message
