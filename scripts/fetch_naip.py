#!/usr/bin/env python3
"""Download a NAIP chip for the pilot bbox into data/pilot/naip.tif.

Uses USDA NAIP on s3://naip-analytic (requester-pays). Needs your own AWS
credentials (``aws configure`` or ``AWS_*`` environment variables); transfer
costs are billed to that account.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import boto3
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import from_bounds as window_from_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pilot" / "naip.tif"
BUCKET = "naip-analytic"
QUAD_DEG = 0.125
NAIP_NAME = re.compile(r"m_(\d{7})_(ne|nw|se|sw)_\d+_\d+_(\d{8})")

# Pilot AOI center (WGS84)
WEST, SOUTH, EAST, NORTH = -71.150, 42.503, -71.125, 42.517
LON = (WEST + EAST) / 2
LAT = (SOUTH + NORTH) / 2


def quarter_quad(lon: float, lat: float) -> tuple[str, str, str]:
    blat, blon = math.floor(lat), math.floor(-lon)
    row = min(7, int((blat + 1 - lat) / QUAD_DEG))
    col = min(7, int((lon + blon + 1) / QUAD_DEG))
    top = blat + 1 - row * QUAD_DEG
    west = -(blon + 1) + col * QUAD_DEG
    ns = "n" if lat >= top - QUAD_DEG / 2 else "s"
    ew = "e" if lon >= west + QUAD_DEG / 2 else "w"
    block = f"{blat:02d}{blon:03d}"
    return block, f"{block}{row * 8 + col + 1:02d}", ns + ew


def list_prefix(client, prefix: str) -> list[str]:
    out, token = [], None
    while True:
        kw = dict(
            Bucket=BUCKET,
            Prefix=prefix,
            Delimiter="/",
            RequestPayer="requester",
        )
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
        out += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


def find_key(client, state: str, block: str, quad: str, quarter: str) -> str | None:
    years = sorted(
        (p for p in list_prefix(client, f"{state}/") if re.search(r"/\d{4}/$", p)),
        reverse=True,
    )
    for year in years:
        for res in list_prefix(client, year):
            for kind in ("rgbir_cog/", "rgbir/"):
                keys = list_prefix(client, f"{res}{kind}{block}/")
                for key in keys:
                    m = NAIP_NAME.search(key)
                    if not m or not key.endswith((".tif", ".mrf")):
                        continue
                    if m.group(1) == quad and m.group(2) == quarter:
                        return key
                if keys:
                    break
    return None


def main() -> int:
    client = boto3.client("s3", region_name="us-west-2")
    block, quad, quarter = quarter_quad(LON, LAT)
    print(f"looking up NAIP ma/{block} {quad} {quarter} at {LON:.5f},{LAT:.5f}")
    key = find_key(client, "ma", block, quad, quarter)
    if not key:
        print("no NAIP key found", file=sys.stderr)
        return 1
    print(f"key s3://{BUCKET}/{key}")

    env = rasterio.Env(
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        AWS_REGION="us-west-2",
    )
    with env:
        with rasterio.open(f"/vsis3/{BUCKET}/{key}") as src:
            # Read AOI in source CRS
            to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x0, y0 = to_src.transform(WEST, SOUTH)
            x1, y1 = to_src.transform(EAST, NORTH)
            bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            win = window_from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            # NAIP rgbir: band 1 red, band 4 near infrared. NDVI needs both.
            indexes = [1, 2, 3, 4] if src.count >= 4 else [1, 2, 3]
            data = src.read(indexes=indexes, window=win, boundless=True, fill_value=0)
            transform = src.window_transform(win)
            # Reproject chip to EPSG:26986 to match lidar.laz
            dst_crs = "EPSG:26986"
            to_dst = Transformer.from_crs(src.crs, dst_crs, always_xy=True)
            corners = [
                (bounds[0], bounds[1]),
                (bounds[0], bounds[3]),
                (bounds[2], bounds[1]),
                (bounds[2], bounds[3]),
            ]
            xs, ys = zip(*(to_dst.transform(x, y) for x, y in corners))
            dst_bounds = (min(xs), min(ys), max(xs), max(ys))
            # ~0.6 m NAIP
            res = 0.6
            width = max(1, int(math.ceil((dst_bounds[2] - dst_bounds[0]) / res)))
            height = max(1, int(math.ceil((dst_bounds[3] - dst_bounds[1]) / res)))
            dst_transform = from_bounds(*dst_bounds, width=width, height=height)
            dest = np.zeros((data.shape[0], height, width), dtype=data.dtype)
            reproject(
                source=data,
                destination=dest,
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": int(dest.shape[0]),
                "dtype": dest.dtype,
                "crs": dst_crs,
                "transform": dst_transform,
                "compress": "deflate",
                "tiled": True,
            }
            with rasterio.open(OUT, "w", **profile) as dst:
                dst.write(dest)
    print(f"wrote {OUT} {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
