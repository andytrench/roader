"""Test-only swept slab: a crowned pavement section along an arc with a crest curve.

Stands in for generated solids so the checkers are exercised on long thin sweep
triangles at unit-frame coordinates (no false positives allowed).
"""

from __future__ import annotations

import math

import numpy as np

from roader.model import Mesh, signed_volume

# Closed convex section (u, dz): bottom, then crowned top.
SECTION = np.array([[-5.0, -0.4], [5.0, -0.4], [5.0, -0.1], [0.0, 0.0], [-5.0, -0.1]])


def centre(s: float, x0=5123.0, y0=-8871.0, h0=0.3, radius=300.0) -> tuple[float, float, float, float]:
    h = h0 + s / radius
    x = x0 + radius * (math.sin(h) - math.sin(h0))
    y = y0 - radius * (math.cos(h) - math.cos(h0))
    z = 100.0 + 0.03 * s - 0.5 * 0.06 * s * s / 400.0
    return x, y, z, h


def ring(s: float) -> np.ndarray:
    x, y, z, h = centre(s)
    nx, ny = -math.sin(h), math.cos(h)
    return np.array([[x + u * nx, y + u * ny, z + dz] for u, dz in SECTION])


def slab(stations: np.ndarray, mesh_id: str = "slab", element_id: str = "el") -> Mesh:
    n = len(SECTION)
    verts = np.vstack([ring(float(s)) for s in stations])
    faces = []
    for k in range(len(stations) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    last = (len(stations) - 1) * n
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
        faces.append([last, last + i, last + i + 1])
    m = Mesh(mesh_id, element_id, "L3", "pavement", "asphalt", "measured", verts, np.array(faces))
    if signed_volume(m) < 0:
        m.faces = m.faces[:, ::-1].copy()
    return m
