"""Section 5 schema contract tests."""

import math

import numpy as np
import pytest

from roader.model import (
    AlignElement,
    Alignment,
    Knot,
    Piecewise,
    Provenance,
    Source,
    dumps,
    loads,
    model_hash,
    stable_id,
)
from roader.model.synthetic import overpass_model, overpass_points

P = Provenance(Source.MEASURED, 1.0)


def test_fr_dm_serialization_round_trips_byte_identical() -> None:
    model = overpass_model()
    data = dumps(model)
    again = dumps(loads(data))
    assert data == again


def test_ac_h5_hash_is_stable_across_builds() -> None:
    assert model_hash(overpass_model()) == model_hash(overpass_model())


def test_stable_id_depends_on_content_not_order() -> None:
    assert stable_id("rd", "osm", 1, 2) == stable_id("rd", "osm", 1, 2)
    assert stable_id("rd", "osm", 1, 2) != stable_id("rd", "osm", 2, 1)


def test_fr_af_1_spiral_is_heading_and_position_continuous() -> None:
    line = AlignElement("line", 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, P)
    x, y, h, _k = line.end_state()
    spiral = AlignElement("spiral", 50.0, 60.0, x, y, h, 0.0, 1 / 300.0, P)
    x2, y2, h2, k2 = spiral.end_state()
    arc = AlignElement("arc", 110.0, 80.0, x2, y2, h2, k2, k2, P)
    alignment = Alignment([line, spiral, arc])
    for s in (50.0, 110.0):
        before = alignment.evaluate(s - 1e-6)
        after = alignment.evaluate(s + 1e-6)
        assert math.hypot(before[0] - after[0], before[1] - after[1]) < 1e-4
        assert abs(before[2] - after[2]) < 1e-6
    # Spiral end heading = k1 * L / 2.
    assert abs(h2 - 60.0 / 600.0) < 1e-12


def test_project_inverts_offset_point_on_an_arc() -> None:
    arc = Alignment([AlignElement("arc", 0.0, 200.0, 0.0, 0.0, 0.3, 1 / 150.0, 1 / 150.0, P)])
    for s, u in ((10.0, 3.0), (120.0, -7.5), (199.0, 0.0)):
        x, y = arc.offset_point(s, u)
        s2, u2 = arc.project(x, y)
        assert abs(s2 - s) < 1e-6 and abs(u2 - u) < 1e-6


def test_ac_p2_piecewise_rejects_steps() -> None:
    with pytest.raises(ValueError):
        Piecewise([Knot(10.0, 3.6, P), Knot(10.0, 3.0, P)])
    taper = Piecewise([Knot(0.0, 3.6, P), Knot(50.0, 5.4, P)])
    assert taper(25.0) == pytest.approx(4.5)


def test_overpass_profile_is_continuous_and_clears_the_freeway() -> None:
    model = overpass_model()
    over = model.roads[2]
    s = np.linspace(0.0, over.length, 2001)
    z = np.array([over.profile.z(v) for v in s])
    assert np.max(np.abs(np.diff(z))) < 0.1
    for crossing in model.crossings:
        assert crossing.clearance_m > 4.9


def test_synthetic_points_leave_the_space_under_the_deck_unsampled() -> None:
    pts = overpass_points()
    assert len(pts.xyz) > 100_000
    under = (np.abs(pts.xyz[:, 0]) < 3.0) & (np.abs(pts.xyz[:, 1]) < 3.0) & (pts.xyz[:, 2] < 103.0)
    assert not np.any(under)
    assert set(np.unique(pts.truth)) >= {"pavement", "deck", "terrain", "vehicle", "parapet", "noise"}
