"""Build every catalogue site in both export profiles and pool the gates (D-120).

For each site in tests/sites.yaml with pinned inputs (or those given by --site):

- runs ``roader build`` once per profile into <out>/<site>/<profile>/ (log in
  build.log next to it), with the LOD chain so FR-AS-1 is gated;
- reads each report.json and writes <out>/catalogue.json and <out>/catalogue.md:
  a site x gate table, the readiness verdict per site (published, every hard
  gate and FR-AS-1 passing) and AC-A1, A3, A4, A5 pooled over the sites;
- renders a driver, a low-aerial and an under-bridge view of each site's
  largest-clearance crossing with tools/viewer and headless Chrome
  (<out>/<site>/renders/).

    python scripts/run_catalogue.py --out data/catalogue [--site ID ...]
        [--profile analytic --profile realtime] [--skip-build] [--no-renders]

Chrome comes from --chrome or $CHROME; renders are skipped when neither is set
and no default install is found. The viewer loads three.js from a CDN.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "tests" / "sites.yaml"
VIEWER = ROOT / "tools" / "viewer" / "index.html"
PROFILES = ("analytic", "realtime")
HARD = ("AC-H1", "AC-H2", "AC-H3", "AC-H4", "AC-H5", "AC-H6", "FR-AS-1")
SOFT = ("AC-A1", "AC-A3", "AC-A4", "AC-A5", "AC-P2", "AC-P3", "AC-P4")
# Spec section 9 accuracy targets, pooled over the catalogue.
A1_RMS, A1_P95, A3_RMS, A4_MAX, A5_MAX = 0.10, 0.20, 0.5, 0.15, 2.0
CHROME_DEFAULTS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium")


def pinned_sites(only: list[str] | None) -> list[dict]:
    sites = yaml.safe_load(SITES.read_text())["sites"]
    if only:
        unknown = set(only) - {s["id"] for s in sites}
        if unknown:
            raise SystemExit(f"unknown site(s): {', '.join(sorted(unknown))}")
        return [s for s in sites if s["id"] in only]
    return [s for s in sites if s.get("lidar_sha256") and s.get("vector_sha256")]


def build(site: dict, profile: str, out: Path, timeout: float) -> dict:
    run_dir = out / site["id"] / profile
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    roader = Path(sys.executable).with_name("roader")
    cmd = [str(roader), "build", "--unit", f"configs/units/{site['id']}.toml", "--crossings", site["crossings"],
           "--out", str(run_dir), "--profile", profile, "--no-points"]
    t0 = time.monotonic()
    with open(run_dir / "build.log", "w") as log:
        try:
            code = subprocess.run(cmd, check=False, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return {"exit": code, "seconds": round(time.monotonic() - t0)}


def unit_dir(site_id: str, profile: str, out: Path) -> tuple[str, Path] | None:
    for status in ("published", "held"):
        d = out / site_id / profile / status / site_id
        if (d / "report.json").exists():
            return status, d
    return None


def site_row(site_id: str, profile: str, out: Path, run: dict | None) -> dict:
    row = {"site": site_id, "profile": profile, "run": run, "status": "no report", "gates": {}, "metrics": {}}
    found = unit_dir(site_id, profile, out)
    if found is None:
        row["ready"] = False
        return row
    row["status"], d = found
    report = json.loads((d / "report.json").read_text())
    for g in report["gates"]["results"]:
        name = g["gate"].split(":")[0]  # the realtime profile reports e.g. "AC-H1:realtime"
        row["gates"][name] = "skip" if g["skipped_reason"] else ("pass" if g["passed"] else f"FAIL({g['violation_count']})")
        row["metrics"][name] = g["metrics"]
    hard_failed = any(g["hard"] and not g["passed"] and not g["skipped_reason"] for g in report["gates"]["results"])
    row["ready"] = row["status"] == "published" and not hard_failed and all(row["gates"].get(g) == "pass" for g in HARD)
    return row


def pooled(rows: list[dict]) -> dict:
    """AC-A1/A3 RMS and bias pooled by point count; p95, A4 and A5 as the worst site (an upper bound)."""

    def rms(gate: str) -> dict:
        ms = [r["metrics"][gate] for r in rows if gate in r["metrics"] and r["metrics"][gate].get("n_points")]
        n = sum(m["n_points"] for m in ms)
        if not n:
            return {"n_points": 0}
        out = {"n_points": n, "rms_m": math.sqrt(sum(m["n_points"] * m["rms_m"] ** 2 for m in ms) / n),
               "bias_m": sum(m["n_points"] * m["bias_m"] for m in ms) / n}
        if gate == "AC-A1":
            out["p95_max_m"] = max(m["p95_m"] for m in ms)
        return out

    def worst(gate: str, key: str) -> float | None:
        vals = [r["metrics"][gate][key] for r in rows if key in r["metrics"].get(gate, {})]
        return max(vals) if vals else None

    a1, a3 = rms("AC-A1"), rms("AC-A3")
    a4, a5 = worst("AC-A4", "max_abs_err_m"), worst("AC-A5", "max_abs_err_m")
    return {
        "AC-A1": {**a1, "passed": bool(a1["n_points"]) and a1["rms_m"] <= A1_RMS and a1["p95_max_m"] <= A1_P95},
        "AC-A3": {**a3, "passed": bool(a3["n_points"]) and a3["rms_m"] <= A3_RMS},
        "AC-A4": {"max_abs_err_m": a4, "passed": a4 is not None and a4 <= A4_MAX},
        "AC-A5": {"max_abs_err_m": a5, "passed": a5 is not None and a5 <= A5_MAX},
    }


def views(model_path: Path) -> list[tuple[str, tuple, tuple]]:
    """Driver, low-aerial and under-bridge cameras at the crossing with the largest dz, unit frame."""
    from roader.model.serialize import loads

    model = loads(model_path.read_bytes())
    if not model.crossings:
        return []
    roads = {r.id: r for r in model.roads}
    c = max(model.crossings, key=lambda c: (roads[c.upper_road].kind == roads[c.lower_road].kind == "mainline", c.dz_m))
    up, lo = roads[c.upper_road], roads[c.lower_road]

    def on(road, s: float, dz: float) -> tuple:
        x, y, _h, _ = road.alignment.evaluate(s)
        return (x, y, road.surface_z(s, 0.0) + dz)

    def back(road, s: float, dist: float) -> float:
        """Station ``dist`` before ``s`` on the side with more road."""
        return max(s - dist, 0.0) if s >= road.length - s else min(s + dist, road.length)

    zc = up.surface_z(c.upper_s, 0.0)
    heading = up.alignment.evaluate(c.upper_s)[2] + math.radians(135.0)
    aerial = (c.x + 130.0 * math.cos(heading), c.y + 130.0 * math.sin(heading), zc + 55.0)
    return [
        ("driver", on(up, back(up, c.upper_s, 90.0), 1.6), on(up, c.upper_s, 1.6)),
        ("aerial", aerial, (c.x, c.y, zc - c.dz_m)),
        ("under", on(lo, back(lo, c.lower_s, 80.0), 1.6), on(lo, c.lower_s, 2.5)),
    ]


def find_chrome(arg: str | None) -> str | None:
    for cand in (arg, os.environ.get("CHROME"), *CHROME_DEFAULTS):
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    return None


def render(site_id: str, out: Path, chrome: str, port: int) -> list[str]:
    found = unit_dir(site_id, "analytic", out)
    if found is None:
        return []
    status, d = found
    shots = out / site_id / "renders"
    shots.mkdir(parents=True, exist_ok=True)
    rel = d.relative_to(out).as_posix()
    written = []
    for name, cam, at in views(d / "model.json"):
        query = urlencode({"f": f"{rel}/meshes.glb", "frame": "unit", "label": f"{site_id} {name} ({status})",
                           "cam": ",".join(f"{v:.2f}" for v in cam), "at": ",".join(f"{v:.2f}" for v in at)})
        png = shots / f"{name}.png"
        subprocess.run([chrome, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                        "--window-size=1600,900", "--virtual-time-budget=40000", "--hide-scrollbars",
                        f"--screenshot={png}", f"http://127.0.0.1:{port}/index.html?{query}"],
                       check=False, capture_output=True, timeout=300)
        if png.exists():
            written.append(png.relative_to(out).as_posix())
    return written


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def serve(out: Path) -> tuple[ThreadingHTTPServer, int]:
    shutil.copyfile(VIEWER, out / "index.html")
    handler = functools.partial(QuietHandler, directory=str(out))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def markdown(rows: list[dict], pools: dict, renders: dict) -> str:
    gates = HARD + SOFT
    lines = ["# Interchange catalogue (D-120)", "",
             "| site | profile | status | ready | " + " | ".join(gates) + " |",
             "|---" * (4 + len(gates)) + "|"]
    for r in rows:
        cells = [r["site"], r["profile"], r["status"], "yes" if r["ready"] else "no"]
        lines.append("| " + " | ".join(cells + [r["gates"].get(g, "-") for g in gates]) + " |")
    for profile, pool in pools.items():
        lines += ["", f"## Pooled accuracy, {profile}", "", "| gate | value | passed |", "|---|---|---|"]
        for gate, m in pool.items():
            vals = ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items() if k != "passed")
            lines.append(f"| {gate} | {vals} | {'yes' if m['passed'] else 'no'} |")
    if renders:
        lines += ["", "## Renders", ""]
        lines += [f"- {site}: " + ", ".join(f"[{Path(p).stem}]({p})" for p in paths) for site, paths in renders.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output root (not committed)")
    parser.add_argument("--site", action="append", help="only this site id (repeatable)")
    parser.add_argument("--profile", action="append", choices=PROFILES, help="default: both")
    parser.add_argument("--skip-build", action="store_true", help="only re-read reports and re-render")
    parser.add_argument("--no-renders", action="store_true")
    parser.add_argument("--chrome", help="Chrome/Chromium executable for renders")
    parser.add_argument("--timeout", type=float, default=4 * 3600, help="per build, seconds")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profiles = args.profile or list(PROFILES)
    sites = pinned_sites(args.site)

    rows = []
    for site in sites:
        for profile in profiles:
            run = None if args.skip_build else build(site, profile, out, args.timeout)
            rows.append(site_row(site["id"], profile, out, run))
            print(f"{site['id']} {profile}: {rows[-1]['status']} ready={rows[-1]['ready']} {run or ''}", flush=True)
    pools = {p: pooled([r for r in rows if r["profile"] == p]) for p in profiles}

    renders: dict[str, list[str]] = {}
    chrome = None if args.no_renders else find_chrome(args.chrome)
    if chrome:
        server, port = serve(out)
        try:
            for site in sites:
                renders[site["id"]] = render(site["id"], out, chrome, port)
        finally:
            server.shutdown()
    elif not args.no_renders:
        print("no Chrome found; renders skipped", file=sys.stderr)

    summary = {"sites": rows, "pooled": pools, "renders": renders,
               "ready": all(r["ready"] for r in rows) and all(m["passed"] for p in pools.values() for m in p.values())}
    (out / "catalogue.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "catalogue.md").write_text(markdown(rows, pools, renders))
    print(f"catalogue ready: {summary['ready']}  ({out / 'catalogue.md'})")
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
