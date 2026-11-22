# Milestones and test sites

The milestones are defined in [ROADWAY_MODEL_SPEC.md](../ROADWAY_MODEL_SPEC.md),
section 10. Each ends with a passing test run and a unit report. A milestone does
not start until the previous one passes its gates.

| Milestone | Scope | Status |
|---|---|---|
| 1. One overpass, end to end | Stages 1 to 10 and 12, analytic profile, finest LOD, flat color. All hard gates; AC-A1, A3, A4, A5. | In progress |
| 2. One full interchange | Layer separation, joint fit, junction areas, occlusion, both profiles, LODs, UVs | Not started |
| 3. Structure and surface realism | Inventory, rule library, bridge parts, materials, markings | Not started |
| 4. 100 km corridor, unattended | Work planning, seams, confidence roll-up, review queue | Not started |
| 5. One state network | Throughput, cost per km, incremental update | Not started |

Test sites are recorded in [tests/sites.yaml](../tests/sites.yaml), one or more per
class in spec section 9. Lidar and vector extracts for those sites live in the
fixtures cache (`data/fixtures/`, not in git) and are fetched by
`scripts/fetch_fixtures.py`.

Data already on disk from the pre-spec pilot (Woburn I-93/I-95 cloverleaf) fits the
milestone 2 cloverleaf class.
