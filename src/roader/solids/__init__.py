"""Stage 10: solid generation from the model (section 7). Lane `solids`.

Public API: generate(model: RoadModel, chord_tol_m: float = 0.01) -> list[Mesh].
Solids are constructed, never recovered or repaired.
"""
