"""Kernels behind FR-SV-4 / FR-SV-7: batched and scalar triangle tests agree."""

from __future__ import annotations

import numpy as np

from roader.validate._geom import (
    points_strictly_inside,
    tri_tri_intersect_batch,
    tri_tri_touch,
)

from .meshes import box_arrays


def test_fr_sv_4_batched_triangle_test_matches_scalar():
    rng = np.random.default_rng(3)
    ta = rng.uniform(0.0, 1.0, (3000, 3, 3)) + [5123.0, -8871.0, 100.0]
    tb = ta[:, ::-1] + rng.normal(0.0, 0.15, (3000, 1, 3)) + rng.normal(0.0, 0.3, (3000, 3, 3))
    batch = tri_tri_intersect_batch(ta, tb)
    scalar = np.array([tri_tri_touch(a, b, []) for a, b in zip(ta, tb, strict=True)])
    assert 0.1 < batch.mean() < 0.9
    assert np.array_equal(batch, scalar)


def test_fr_sv_4_touching_counts_as_intersecting():
    a = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    b = np.array([[[0.25, 0.25, 0.0], [0.25, 0.25, 1.0], [1.0, 1.0, 1.0]]])  # vertex on a's face
    assert tri_tri_intersect_batch(a, b)[0]
    assert not tri_tri_intersect_batch(a, b + [0, 0, 1e-3])[0]


def test_points_strictly_inside_box():
    v, f = box_arrays()
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.999, 0.999, 0.001]])
    assert points_strictly_inside(v, f, pts).tolist() == [True, False, False, False, True]


def test_fr_sv_4_near_coplanar_faces_sharing_a_vertex_do_not_intersect():
    """Two terrain faces meeting at one vertex, coplanar to 1e-7, with disjoint sectors (M1 site case)."""
    from roader.validate._geom import vertex_overlap

    a = np.array([[82.0, 44.0, 172.5688], [81.64384972, 41.2607818, 172.87670152], [82.32274146, 40.98091639, 172.91206825]])
    b = np.array([[82.0, 44.0, 172.5688], [80.21572529, 41.84626491, 172.80267083], [80.92790427, 41.55495385, 172.83951405]])
    assert not vertex_overlap(a, b, 0, 0)
    # Rotating B into A's sector makes them overlap.
    c = np.array([a[0], a[0] + 0.5 * (a[1] - a[0]) + 0.5 * (a[2] - a[0]), a[1] + 1e-3 * (a[2] - a[1])])
    assert vertex_overlap(a, c, 0, 0)
