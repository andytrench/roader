"""D-007 unit frame: FR-DM-5, FR-IN-2."""

import numpy as np
from pyproj import Geod, Proj

from roader.io import FrameTransformer, aoi_anchor, unit_frame

LON, LAT = -71.8161, 42.38906
GEOD = Geod(ellps="GRS80")


def test_fr_dm_5_anchor_is_origin():
    frame = unit_frame(LON, LAT)
    x, y = FrameTransformer(frame).lonlat_to_unit(LON, LAT)
    assert abs(float(x)) < 1e-9 and abs(float(y)) < 1e-9
    assert frame.vertical_datum == "NAVD88"
    assert "+proj=tmerc" in frame.proj and "+ellps=GRS80" in frame.proj and "+k=1" in frame.proj


def test_fr_dm_5_round_trip_under_1mm():
    frame = unit_frame(LON, LAT)
    t = FrameTransformer(frame)
    rng = np.random.default_rng(1)
    lon = LON + rng.uniform(-0.07, 0.07, 500)  # about +-5.8 km
    lat = LAT + rng.uniform(-0.05, 0.05, 500)
    x, y = t.lonlat_to_unit(lon, lat)
    lon2, lat2 = t.unit_to_lonlat(x, y)
    _, _, err = GEOD.inv(lon, lat, lon2, lat2)
    assert np.max(np.abs(err)) < 1e-3


def test_fr_dm_5_source_crs_round_trip():
    frame = unit_frame(LON, LAT)
    t = FrameTransformer(frame, "EPSG:3857")
    mx, my = -7994531.7, 5219438.1
    x, y = t.to_unit(mx, my)
    bx, by = t.from_unit(x, y)
    assert abs(float(bx) - mx) < 1e-4 and abs(float(by) - my) < 1e-4
    assert np.hypot(x, y) < 50.0


def test_fr_dm_5_distortion_under_1e5_at_5km():
    frame = unit_frame(LON, LAT)
    t = FrameTransformer(frame)
    proj = Proj(frame.proj)
    for bearing in (0.0, 45.0, 90.0, 135.0, 180.0, 270.0):
        lon, lat, _ = GEOD.fwd(LON, LAT, bearing, 5000.0)
        factors = proj.get_factors(lon, lat)
        assert abs(factors.meridional_scale - 1.0) < 1e-5
        assert abs(factors.parallel_scale - 1.0) < 1e-5
        # A 100 m segment at 5 km keeps its length to 1e-5.
        lon2, lat2, _ = GEOD.fwd(lon, lat, bearing + 30.0, 100.0)
        x, y = t.lonlat_to_unit([lon, lon2], [lat, lat2])
        assert abs(np.hypot(x[1] - x[0], y[1] - y[0]) / 100.0 - 1.0) < 1e-5


def test_aoi_anchor_is_center():
    assert aoi_anchor([-71.81975, 42.38636, -71.81245, 42.39176]) == (-71.8161, 42.38906)
