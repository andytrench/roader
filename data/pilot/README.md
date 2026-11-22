# Pilot data (local, not in git)

AOI: **I-93 / I-95 interchange**, Woburn–Reading, MA  
`bbox = [-71.150, 42.503, -71.125, 42.517]` (WGS84)

| File | Source | Notes |
|---|---|---|
| `lidar.laz` | USGS 3DEP EPT `MA_CentralEastern_1_2021` via PDAL (`fetch_lidar.json`), reprojected to EPSG:26986, ~1 m resolution | Includes ASPRS class **17** bridge decks |
| `naip.tif` | USDA NAIP 2025 COG chip (`scripts/fetch_naip.py`), EPSG:26986 | RGB |
| `out_roads.gpkg` | Created by `roader pilot` | Not staged yet |

Refresh:

```bash
pdal pipeline data/pilot/fetch_lidar.json
python scripts/fetch_naip.py
```
