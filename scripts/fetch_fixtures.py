"""Download pinned test-site inputs into data/fixtures/<site_id>/ (FR-PL-3, D-008).

For each site in tests/sites.yaml (or one given by --site):

- lidar.laz: the site AOI cropped from the 3DEP EPT at full resolution, all
  dimensions forwarded, via the PDAL CLI.
- osm.json: an Overpass extract of every highway way touching the AOI, with all
  of its nodes, written as canonical JSON (elements sorted by type and id).

The sha256 of each file goes to data/fixtures/<site_id>/manifest.json and back
into tests/sites.yaml (lidar_sha256, vector_sha256). Existing files are kept
unless --force.

    python scripts/fetch_fixtures.py [--site ID] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "tests" / "sites.yaml"
FIXTURES = ROOT / "data" / "fixtures"
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "roader/0.1 (test fixture fetch)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ept_bounds(aoi_lonlat: list[float], ept_url: str) -> tuple[str, str]:
    """AOI as a PDAL bounds string in the EPT's native CRS, plus that CRS."""
    meta = requests.get(ept_url, timeout=60, headers={"User-Agent": USER_AGENT})
    meta.raise_for_status()
    srs = meta.json()["srs"]
    crs = f"{srs['authority']}:{srs['horizontal']}"
    west, south, east, north = aoi_lonlat
    to_native = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    minx, miny, maxx, maxy = to_native.transform_bounds(west, south, east, north, densify_pts=21)
    return f"([{minx!r},{maxx!r}],[{miny!r},{maxy!r}])", crs


def fetch_lidar(site: dict, dest: Path) -> dict:
    pdal = shutil.which("pdal") or "/opt/homebrew/bin/pdal"
    ept = site["lidar"]["ept"]
    bounds, crs = ept_bounds(site["aoi_lonlat"], ept)
    pipeline = {
        "pipeline": [
            {"type": "readers.ept", "filename": ept, "bounds": bounds},
            {
                "type": "writers.las",
                "filename": str(dest.with_suffix(".part.laz")),
                "compression": "laszip",
                "minor_version": 4,
                "forward": "all",
                "extra_dims": "all",
            },
        ]
    }
    print(f"  lidar: {ept} bounds {bounds} ({crs})", flush=True)
    result = subprocess.run([pdal, "pipeline", "--stdin"], input=json.dumps(pipeline), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pdal failed: {(result.stderr or result.stdout)[-2000:]}")
    dest.with_suffix(".part.laz").replace(dest)
    return {"source": ept, "bounds": bounds, "crs": crs, "pdal": pdal}


def overpass_query(aoi_lonlat: list[float]) -> str:
    west, south, east, north = aoi_lonlat
    return f'[out:json][timeout:180];way["highway"]({south},{west},{north},{east});(._;>;);out body;'


def fetch_osm(site: dict, dest: Path) -> dict:
    query = overpass_query(site["aoi_lonlat"])
    last: Exception | None = None
    for attempt, url in enumerate(OVERPASS_URLS):
        if attempt:
            time.sleep(5)
        try:
            print(f"  osm: {url}", flush=True)
            response = requests.post(url, data={"data": query}, timeout=240, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            doc = response.json()
            if not any(e.get("type") == "way" for e in doc.get("elements", [])):
                raise RuntimeError("Overpass returned no highway ways")
            break
        except Exception as exc:  # noqa: BLE001 - the next mirror may answer
            print(f"    failed: {exc}", flush=True)
            last = exc
    else:
        raise RuntimeError(f"no Overpass mirror answered: {last}")
    doc["elements"] = sorted(doc["elements"], key=lambda e: (e["type"], e["id"]))
    dest.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    return {"source": url, "query": query, "osm_base": doc.get("osm3s", {}).get("timestamp_osm_base", "")}


def pin_hashes(site_id: str, lidar_sha: str, vector_sha: str) -> None:
    """Rewrite the two hash lines inside the site's block of sites.yaml, keeping comments."""
    lines = SITES.read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if re.match(rf"\s*- id: {re.escape(site_id)}\s*$", line))
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"\s*- id: ", lines[i])), len(lines))
    for i in range(start, end):
        for key, value in (("lidar_sha256", lidar_sha), ("vector_sha256", vector_sha)):
            match = re.match(rf"(\s*){key}:", lines[i])
            if match:
                lines[i] = f'{match.group(1)}{key}: "{value}"\n'
    SITES.write_text("".join(lines))


def describe(site: dict, name: str) -> str:
    """Source of a file fetched by an earlier run that left no manifest entry."""
    if name == "lidar.laz":
        return site["lidar"]["ept"]
    return overpass_query(site["aoi_lonlat"])


def fetch_site(site: dict, force: bool) -> None:
    out = FIXTURES / site["id"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"site_id": site["id"], "files": {}}
    print(f"{site['id']}:", flush=True)
    for name, fetch in (("lidar.laz", fetch_lidar), ("osm.json", fetch_osm)):
        path = out / name
        if path.exists() and not force:
            print(f"  {name}: exists, kept", flush=True)
            entry = manifest["files"].get(name, {})
            entry.setdefault("source", describe(site, name))
            entry.setdefault("fetched_utc", "unknown")
        else:
            entry = fetch(site, path)
            entry["fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry["sha256"] = sha256_file(path)
        entry["bytes"] = path.stat().st_size
        manifest["files"][name] = entry
        manifest["aoi_lonlat"] = site["aoi_lonlat"]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"  {name}: {entry['bytes']} bytes sha256 {entry['sha256']}", flush=True)
    pin_hashes(site["id"], manifest["files"]["lidar.laz"]["sha256"], manifest["files"]["osm.json"]["sha256"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", help="only this site id")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    sites = yaml.safe_load(SITES.read_text())["sites"]
    if args.site:
        sites = [s for s in sites if s["id"] == args.site]
        if not sites:
            parser.error(f"unknown site {args.site}")
    for site in sites:
        fetch_site(site, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
