"""The lane's own checkers must catch what they claim to catch."""

from __future__ import annotations

import numpy as np

from roader.model import Mesh

from . import meshcheck as mc

_CUBE_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ]
)


def _box(lo, hi, name="b") -> Mesh:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return Mesh(name, name, "L5", "deck", "x", "rule", np.array(v, float), _CUBE_FACES)


def test_valid_box_passes():
    assert mc.solid_problems(_box((0, 0, 0), (1, 2, 3))) == []


def test_flipped_face_and_hole_are_caught():
    box = _box((0, 0, 0), (1, 1, 1))
    flipped = Mesh("f", "f", "L5", "deck", "x", "rule", box.vertices, box.faces.copy())
    flipped.faces[0] = flipped.faces[0][[0, 2, 1]]
    assert mc.closed_and_oriented(flipped)
    holed = Mesh("h", "h", "L5", "deck", "x", "rule", box.vertices, box.faces[1:])
    assert mc.closed_and_oriented(holed)


def test_overlap_crossing_and_t_junction_are_caught():
    a = _box((0, 0, 0), (2, 2, 2), "a")
    b = _box((1, 1, 1), (3, 3, 3), "b")
    assert mc.crossing_pairs([a, b])
    assert mc.inside(mc.interior_samples(b), a).any()
    c = _box((2, 0.5, 0), (3, 1.5, 1), "c")  # touches a's face with its own vertices inside it
    assert mc.t_junctions(c, a) > 0
    d = _box((2, 0, 0), (3, 2, 2), "d")  # same touching face, other diagonal: not conforming
    assert mc.t_junctions(d, a) == 0 and mc.shared_faces(d, a) == 0
    assert mc.crossing_pairs([a, d])
    e = Mesh("e", "e", "L5", "deck", "x", "rule", d.vertices, d.faces.copy())
    e.faces[10:12] = [[3, 0, 7], [0, 4, 7]]  # diagonal matching a's face: conforming
    assert mc.solid_problems(e) == []
    assert mc.shared_faces(e, a) == 2 and not mc.crossing_pairs([a, e])
