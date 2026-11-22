# Milestones and test sites

The milestones are defined in [ROADWAY_MODEL_SPEC.md](../ROADWAY_MODEL_SPEC.md),
section 10. Each ends with a passing test run and a unit report. A milestone does
not start until the previous one passes its gates.

| Milestone | Scope | Status |
|---|---|---|
| 1. One overpass, end to end | Stages 1 to 10 and 12, analytic profile, finest LOD, flat color. All hard gates; AC-A1, A3, A4, A5. | Done 2026-11-22 |
| 2. One full interchange | Layer separation, joint fit, junction areas, occlusion, both profiles, LODs, UVs | In progress |
| 3. Structure and surface realism | Inventory, rule library, bridge parts, materials, markings | Not started |
| 4. 100 km corridor, unattended | Work planning, seams, confidence roll-up, review queue | Not started |
| 5. One state network | Throughput, cost per km, incremental update | Not started |

Test sites are recorded in [tests/sites.yaml](../tests/sites.yaml), one or more per
class in spec section 9. Lidar and vector extracts for those sites live in the
fixtures cache (`data/fixtures/`, not in git) and are fetched by
`scripts/fetch_fixtures.py`.

## Milestone 1 result

Site `wboylston_laurel_i190`: Laurel Street on one 108 m bridge over the I-190
twin carriageways, West Boylston MA, lidar MA_CentralEastern_1_2021 (QL1, 8.1 M
returns). Reproduce with:

```bash
roader build --unit configs/units/wboylston_laurel_i190.toml --out data/out
```

The unit publishes with every hard gate passing (val3dity included):

| Gate | Result |
|---|---|
| AC-H1 solids | 27 meshes, 0 native and 0 val3dity violations |
| AC-H2 crossings | 2 of 2 reference crossings, no extras |
| AC-H3 voids | 739 rays clear, minimum opening 4.64 m |
| AC-H4 continuity | 25 joints within D-020 tolerances |
| AC-H5 determinism | 12 files byte-identical across two renderings |
| AC-H6 provenance | 171 records, every rule fill named |
| AC-A1 surface | RMS 0.058 m, p95 0.065 m on 78 k held-out returns |
| AC-A3 edges | RMS 0.24 m |
| AC-A4 crossing dz | max error 0.024 m |
| AC-A5 structure ends | max error 0.17 m against draft class-17 ends (consistency check only, D-044) |

The reference crossing list and structure ends in
`tests/sites/wboylston_laurel_i190/crossings.yaml` were reviewed and confirmed on
2026-11-22. Remaining flags:

- Both under-deck clearances (4.67 m and 4.81 m) are below the 4.88 m class
  minimum. The surfaces are measured, but the deck depth is sized by rule because
  no soffit returns were found (D-033), so the flag stays.
- Four paved sheets with no vector are reported as FR-VC-5 candidates.

The unit report's `figures/` holds plan and profile plots. Driver, low-aerial and
under-bridge views (AC-P5) are captured from `tools/viewer/index.html` with and
without the lidar overlay, and in provenance color.

## Milestone 2 sites

The cloverleaf is `woburn_i93_i95` (I-93 over I-95, D-036). The stack candidate is
the I-635/US 75 High Five in Dallas.
