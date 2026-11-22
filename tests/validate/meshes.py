"""Hand-built meshes for the FR-SV tests."""

from __future__ import annotations

import numpy as np

from roader.model import Mesh

BOX_FACES = np.array(
    [
        [0, 3, 2], [0, 2, 1],  # bottom, z = z0
        [4, 5, 6], [4, 6, 7],  # top, z = z1
        [0, 1, 5], [0, 5, 4],  # y = y0
        [1, 2, 6], [1, 6, 5],  # x = x1
        [2, 3, 7], [2, 7, 6],  # y = y1
        [3, 0, 4], [3, 4, 7],  # x = x0
    ],
    dtype=np.int64,
)


def box_arrays(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    return v, BOX_FACES.copy()


def mesh(mesh_id: str, vertices, faces, *, closed: bool = True, element_id: str = "el") -> Mesh:
    return Mesh(
        id=mesh_id,
        element_id=element_id,
        layer="L5",
        part="deck",
        material="concrete",
        source="measured",
        vertices=vertices,
        faces=faces,
        closed=closed,
    )


def box(mesh_id: str = "box", lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0), **kw) -> Mesh:
    v, f = box_arrays(lo, hi)
    return mesh(mesh_id, v, f, **kw)


def merged(mesh_id: str, *parts: tuple[np.ndarray, np.ndarray]) -> Mesh:
    vs, fs, offset = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + offset)
        offset += len(v)
    return mesh(mesh_id, np.vstack(vs), np.vstack(fs))


def tetra(apex_up: bool = True, at=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedron with one vertex at ``at``; the rest above (or below) it."""
    o = np.array(at, dtype=np.float64)
    s = 1.0 if apex_up else -1.0
    v = np.array([o, o + [1, 0, s], o + [0, 1, s], o + [-1, -1, s]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]], dtype=np.int64)
    m = mesh("t", v, f)
    from roader.model import signed_volume

    if signed_volume(m) < 0:
        f = f[:, ::-1].copy()
    return v, f
