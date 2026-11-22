# Third-party code and data

Update this file before adding a dependency.

## Python / C++ libraries (intended stack)

| Package | License | Role |
|---|---|---|
| numpy | BSD-3 | Arrays |
| scipy | BSD-3 | Geometry helpers, nearest-point search |
| shapely | BSD-3 (links GEOS, LGPL-2.1, dynamically) | Vector geometry |
| pyproj | MIT (bundles PROJ, MIT) | CRS, unit working frame |
| geopandas | BSD-3 | GeoDataFrames / GeoPackage I/O |
| rasterio | BSD-3 | NAIP / DEM rasters |
| laspy (+ lazrs) | BSD-2 (lazrs MIT/Apache-2.0) | LAS/LAZ/COPC read |
| PDAL CLI (+ python-pdal) | BSD-3 | Clip and crop in `scripts/fetch_fixtures.py` (external command) |
| osmnx | MIT | OSM network + features download |
| networkx | BSD-3 | Graph ops (via osmnx) |
| click | BSD-3 | CLI |
| tomli / tomllib | Apache-2.0 / PSF | Config |
| boto3 | Apache-2.0 | NAIP S3 listing / fetch (scripts) |
| PyYAML | MIT | `tests/sites.yaml`, site registry |
| requests | Apache-2.0 | HTTP for vector and fixture downloads |
| matplotlib | Matplotlib License (PSF-based, BSD-compatible) | Unit report figures |
| pytest | MIT | Tests |

Optional later (not wired yet):

| Package | License | Role |
|---|---|---|
| libOpenDRIVE | Apache-2.0 | OpenDRIVE mesh / parse |
| r:trån | Apache-2.0 | OpenDRIVE → CityGML (Java CLI) |
| opencv-python-headless | Apache-2.0 | NAIP edge / pavement cues |

External tools (run as separate processes, never imported or distributed):

| Tool | License | Role |
|---|---|---|
| val3dity 2.7.0 | GPL-3.0 | AC-H1 independent validity oracle, tests only (D-003). Built outside the repo by `scripts/install_val3dity.sh`; located via `VAL3DITY`. Do not bundle in shipped artifacts or images. |

Avoid in core unless quarantined: GPL/AGPL packages (e.g. some CGAL bindings,
GPL CLIs).

## Data

| Source | Terms | Role |
|---|---|---|
| OpenStreetMap | ODbL; attribute "© OpenStreetMap contributors" | Centerlines, tags (pinned extracts in `data/fixtures/`; the M1 site's ways are a MassGIS import) |
| USGS 3DEP LiDAR | Public domain; credit "USGS 3D Elevation Program" | Point evidence (M1: MA_CentralEastern_1_2021) |
| NAIP | USDA — check current redistribution rules | Imagery cues |
| TIGER/Line | Public domain | Optional US centerline prior |

OSM-derived **data** products must keep ODbL attribution / share-alike as
required. That does not force Roader **code** to be ODbL.
