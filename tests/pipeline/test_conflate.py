"""Stages 4 and 5: FR-VC-1..5, FR-CR-1..3 on the synthetic overpass."""

from __future__ import annotations

import numpy as np

import roader
from roader.conflate.graph import data_extent, select_roads
from roader.conflate.polyline import Polyline
from roader.io import VectorEdge
from roader.pipeline import run_pipeline
from tests.pipeline.conftest import by_feature


def test_import_path_is_this_worktree():
    assert "roader-lanes" in roader.__file__ or "src/roader" in roader.__file__


def test_fr_vc_1_scope_and_chaining(points, vectors):
    over = next(v for v in vectors if v.feature_id == "over")
    half = len(over.coords) // 2
    split = [v for v in vectors if v.feature_id != "over"] + [
        VectorEdge("synthetic", "over_a", over.coords[: half + 1], dict(over.tags)),
        VectorEdge("synthetic", "over_b", over.coords[half:], dict(over.tags)),
        VectorEdge("synthetic", "far", np.array([[-280.0, 250.0], [-200.0, 250.0]]), {"highway": "residential"}),
    ]
    seeds = select_roads(split, data_extent(points.xyz[:, :2]))
    assert len(seeds) == 3, [s.name for s in seeds]
    street = next(s for s in seeds if s.kind == "crossing_street")
    assert sorted(sp.feature_id for sp in street.spans) == ["over_a", "over_b"]
    assert {s.kind for s in seeds} == {"mainline", "crossing_street"}
    assert all(s.oneway for s in seeds if s.kind == "mainline")


def test_fr_vc_1_two_way_street_drawn_in_mixed_directions_chains_without_doubling_back():
    tags = {"highway": "secondary", "name": "Main Street"}
    pieces = [np.array([[a, 0.0], [a + 50.0, 0.5], [a + 100.0, 0.0]]) for a in np.arange(0.0, 500.0, 100.0)]
    drawn = [p if k % 2 else p[::-1] for k, p in enumerate(pieces)]  # every other way drawn east to west
    edges = [VectorEdge("osm", str(k), c, dict(tags)) for k, c in enumerate(drawn)]
    edges.append(VectorEdge("osm", "fwy", np.array([[250.0, -100.0], [250.0, 100.0]]), {"highway": "motorway", "ref": "I 1", "oneway": "yes"}))
    street = next(s for s in select_roads(edges) if s.kind == "crossing_street")
    assert len(street.vector_refs) == 5
    assert abs(street.line.length - 500.0) < 0.1
    x = street.line.xy[:, 0]
    assert np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)


def _terminal_edges(through_street: bool) -> list[VectorEdge]:
    """An exit way ending on a street node and an entry way leaving it straight on (a diamond terminal)."""
    link = {"highway": "motorway_link", "oneway": "yes"}
    edges = [
        VectorEdge("osm", "fwy", np.array([[-300.0, -40.0], [300.0, -40.0]]), {"highway": "motorway", "ref": "I 1", "oneway": "yes"}),
        VectorEdge("osm", "exit", np.array([[-200.0, -30.0], [-100.0, -5.0], [0.0, 0.0]]), dict(link)),
        VectorEdge("osm", "entry", np.array([[0.0, 0.0], [100.0, -5.0], [200.0, -30.0]]), dict(link)),
    ]
    street = np.array([[0.0, -200.0], [0.0, 0.0], [0.0, 200.0]]) if through_street else np.array([[30.0, -200.0], [30.0, 200.0]])
    edges.append(VectorEdge("osm", "street", street, {"highway": "primary", "name": "Route 1"}))
    return edges


def test_fr_vc_1_ramp_ways_break_at_a_street_node_d123():
    """Ramp ways meeting at a node of a crossing street are two ramps ending at a terminal, not one road through it (D-123)."""
    ramps = sorted((s for s in select_roads(_terminal_edges(True)) if s.kind == "ramp"), key=lambda s: s.vector_refs[0].feature_id)
    assert [[r.feature_id for r in s.vector_refs] for s in ramps] == [["entry"], ["exit"]]
    # Away from a street node the same two ways are still one ramp.
    ramps = [s for s in select_roads(_terminal_edges(False)) if s.kind == "ramp"]
    assert len(ramps) == 1 and len(ramps[0].vector_refs) == 2


def test_d_160_fr_in_5_only_paths_on_their_own_bridges_are_selected():
    fwy = VectorEdge("osm", "fwy", np.array([[-300.0, 0.0], [300.0, 0.0]]), {"highway": "motorway", "ref": "I 1", "oneway": "yes"})
    paths = [
        VectorEdge("osm", "under", np.array([[0.0, -100.0], [0.0, 100.0]]), {"highway": "footway", "tunnel": "yes"}),
        VectorEdge("osm", "bridge", np.array([[60.0, -100.0], [60.0, 100.0]]), {"highway": "footway", "bridge": "yes", "layer": "1"}),
        VectorEdge("osm", "sidewalk", np.array([[120.0, -100.0], [120.0, 100.0]]),
                   {"highway": "footway", "footway": "sidewalk", "bridge": "yes", "layer": "1"}),
    ]
    picked = sorted(r.feature_id for s in select_roads([fwy, *paths]) for r in s.vector_refs)
    assert picked == ["bridge", "fwy"]


def test_fr_af_3_a_ramp_ending_on_a_street_node_is_its_child_d124():
    """A street is the parent of a ramp end on it, ahead of another ramp ending at the same node (D-124, revises D-051)."""
    from roader.fit.links import junctions

    seeds = select_roads(_terminal_edges(True))
    names = {s.id: s.vector_refs[0].feature_id for s in seeds}
    got = sorted((names[j.child], j.end, names[j.parent]) for j in junctions(seeds))
    assert ("entry", "start", "street") in got and ("exit", "end", "street") in got
    assert not any(names[j.parent] in ("exit", "entry") for j in junctions(seeds))


def test_fr_vc_2_lower_roads_are_occluded_under_the_deck(model):
    for fid in ("eb", "wb"):
        road = by_feature(model, fid)
        occluded = [iv for iv in road.support if iv.kind == "occluded"]
        assert len(occluded) == 1
        iv = occluded[0]
        assert iv.reason == "under_deck"
        x, _y = road.alignment.offset_point(0.5 * (iv.s0 + iv.s1), 0.0)
        assert abs(x) < 10.0
    over = by_feature(model, "over")
    assert all(iv.kind == "observed" for iv in over.support)


def test_fr_vc_3_registration_moves_centerlines_onto_the_pavement(model, truth, vectors):
    for fid in ("eb", "wb", "over"):
        t = Polyline.from_alignment(by_feature(truth, fid).alignment)
        fitted = Polyline.from_alignment(by_feature(model, fid).alignment)
        vec = next(v for v in vectors if v.feature_id == fid)
        _s, u_in, ok_in = t.project(vec.coords)
        _s, u_out, ok_out = t.project(fitted.xy[20:-20:10])
        assert np.sqrt(np.mean(u_in[ok_in] ** 2)) > 0.9  # the input is several decimeters off
        assert np.max(np.abs(u_out[ok_out])) < 0.10  # FR-VC-3 target is 0.5 m


def test_fr_vc_4_structure_ends_from_data(model, truth):
    st = model.structures[0]
    ts = truth.structures[0]
    assert len(model.structures) == 1
    assert abs(st.s_start - ts.s_start) < 1.0
    assert abs(st.s_end - ts.s_end) < 1.0
    assert any(f.startswith("FR-VC-4:") for f in st.prov.flags)


def test_fr_cr_1_crossings_order_dz_and_skew(model, truth):
    over = by_feature(model, "over")
    assert len(model.crossings) == 2
    truth_dz = {c.lower_road: c.dz_m for c in truth.crossings}
    for c in model.crossings:
        assert c.upper_road == over.id
        lower = model.road(c.lower_road)
        t_lower = by_feature(truth, lower.vector_refs[0].feature_id)
        assert abs(c.dz_m - truth_dz[t_lower.id]) < 0.1
        assert abs(abs(c.skew_rad) - 0.17) < 0.01
        assert c.tags_agree is True


def test_fr_cr_2_tag_disagreement_is_logged_with_both_values(points, vectors, truth):
    flipped = [VectorEdge(v.source, v.feature_id, v.coords, {**v.tags, "layer": "2"} if v.feature_id == "eb" else v.tags) for v in vectors]
    res = run_pipeline(points, flipped, truth.frame, truth.unit)
    eb = by_feature(res.model, "eb")
    c = next(c for c in res.model.crossings if c.lower_road == eb.id)
    assert c.upper_road != eb.id  # order is measured, not taken from tags
    assert c.tags_agree is False
    assert "layer=2" in c.tag_lower
    flags = [f for f in res.model.flags if f.code == "FR-CR-2:tags_disagree"]
    assert len(flags) == 1 and "layer=2" in flags[0].message


def test_fr_cr_3_chained_crossings_share_one_structure(model):
    st = model.structures[0]
    assert {c.structure_id for c in model.crossings} == {st.id}
    assert sorted(st.crossed_roads) == sorted(c.lower_road for c in model.crossings)


def test_fr_vc_2_track_crosses_a_gap_on_a_crest_curve_d129():
    """A 40 m hole in the returns on a crest curve (a D-018 holdout block, parclo street) does not end the track (D-129)."""
    from types import SimpleNamespace

    from roader.conflate.track import track_line

    s = np.arange(0.0, 300.0, 0.5)
    g0, g1, a, b = 0.065, -0.065, 100.0, 175.0  # crest curve between a and b
    z = np.where(s < a, 100.0 + g0 * s, np.where(s > b, 100.0 + g0 * a + 0.5 * (g0 + g1) * (b - a) + g1 * (s - b),
                                                  100.0 + g0 * a + g0 * (s - a) + 0.5 * (g1 - g0) / (b - a) * (s - a) ** 2))
    keep = (s < 130.0) | (s > 170.0)
    xs, zs = np.repeat(s[keep], 3), np.repeat(z[keep], 3)
    ys = np.tile([-1.0, 0.0, 1.0], keep.sum())
    n = len(xs)
    layers = SimpleNamespace(
        smooth=np.ones(n, bool), sheet=np.zeros(n, np.int64), z=zs,
        center_xy=lambda idx=None: np.column_stack([xs, ys]) if idx is None else np.column_stack([xs[idx], ys[idx]]),
    )
    ev = SimpleNamespace(layers=layers, layer_label=np.zeros(n, np.int64), layer_prob=np.ones(n),
                         cell_top=lambda xy, radius=0: np.full(len(xy), np.nan))
    tr = track_line(Polyline(np.array([[0.0, 0.0], [300.0, 0.0]])), ev)
    beyond = tr.s > 175.0
    assert np.all(tr.kind[beyond] == "observed")
    assert np.nanmax(np.abs(tr.z[beyond] - np.interp(tr.s[beyond], s, z))) < 0.05


def _crossing_fit(fid: str, xy: np.ndarray, z: float):
    from types import SimpleNamespace

    from roader.conflate.graph import RoadSeed
    from tests.pipeline import scenes as S

    xy = np.asarray(xy, dtype=float)
    line = Polyline(xy)
    h = float(np.arctan2(*(xy[-1] - xy[0])[::-1]))
    road = S._road(fid, "ramp", "motorway_link", S._straight(float(xy[0, 0]), float(xy[0, 1]), h, line.length), S._flat(line.length, z),
                   S._section(3.0, 3.0, 1), oneway=True, lanes=1)
    seed = RoadSeed(fid, fid, "motorway_link", "ramp", True, line, [], nodes=xy)
    empty = np.empty(0)
    return SimpleNamespace(road=road, line=line, seed=seed, points=SimpleNamespace(s=empty, u=empty, z=empty))


def test_fr_cr_1_lines_weaving_on_one_level_share_pavement_and_do_not_cross_d135():
    """Two branches sharing their first metres weave across each other within the noise: no crossing (D-135).

    The same weave with one road 7 m higher, or a square meeting at one level, is still a crossing.
    """
    from roader.conflate.crossings import find_crossings

    a = [[0.0, 0.0], [200.0, 0.0]]
    weave = [[0.0, 0.4], [100.0, -0.4], [200.0, 0.4]]
    crossings, _m, flags = find_crossings([_crossing_fit("a", a, 100.0), _crossing_fit("b", weave, 100.0)])
    assert crossings == []
    assert sorted(f.code for f in flags) == ["FR-CR-1:shared_pavement"] * 2

    crossings, _m, _f = find_crossings([_crossing_fit("a", a, 100.0), _crossing_fit("b", weave, 107.0)])
    assert len(crossings) == 2 and all(abs(c.dz_m - 7.0) < 1e-6 for c in crossings)

    crossings, _m, _f = find_crossings([_crossing_fit("a", a, 100.0), _crossing_fit("c", [[100.0, -50.0], [100.0, 50.0]], 100.0)])
    assert len(crossings) == 1
