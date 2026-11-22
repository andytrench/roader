# Working on Roader

Roader is a **standalone** R&D project: reconstruct road objects (general, then
detailed) from LiDAR and NAIP imagery for RF / microwave modelling.

## Self-contained

- Do not import, symlink, or vendor code from outside this repository.
- If an idea was learned elsewhere, re-implement it here under Roader’s
  license and document the approach in `docs/` — do not copy files across.
- Never commit secrets, credentials, or absolute local paths.

## Goal

Build the Roadway Infrastructure Model (RIM) specified in
[ROADWAY_MODEL_SPEC.md](ROADWAY_MODEL_SPEC.md): public airborne lidar plus 2D
road vectors in, a parametric road model out, and closed solid meshes rendered
deterministically from that model. Scope is US limited-access highways, ramps,
interchanges, their grade separations, and crossing streets within 150 m.

The spec is the authority. Work through its milestones in order, write the
section 9 tests before the code, cite requirement IDs (for example FR-SG-3) in
tests, commits and comments, and record open choices in
[DECISIONS.md](DECISIONS.md). See `docs/MODEL.md` and `docs/RND_PILOT.md`.

## Stack policy

Prefer **commercial-use-friendly** open libraries (MIT, BSD, Apache-2.0).
Record every third-party dependency and its license in `docs/THIRD_PARTY.md`
before adding it. Avoid GPL/AGPL dependencies in the core library unless the
maintainer explicitly approves a quarantine CLI.

OpenStreetMap / TIGER / NAIP / 3DEP are **data** sources: keep attribution and
respect their terms (e.g. ODbL share-alike for OSM-derived *data* products).

## Layout

- `src/roader/` — library and CLI, one sub-package per spec area (`io`,
  `evidence`, `conflate`, `fit`, `complete`, `structures`, `rules`, `model`,
  `solids`, `assets`, `validate`, `export`); `build` chains them for one unit
- `configs/` — `*.example` config templates (local copies are not committed)
- `data/` — local inputs/outputs (not committed)
- `docs/` — model, pilot, licenses
- `tests/` — unit tests

## Git

Branch from `main`.
