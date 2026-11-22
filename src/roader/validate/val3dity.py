"""val3dity as an independent solid-validity oracle (AC-H1, D-003).

val3dity is GPL-3.0 and runs only as an external command; it is never imported
or vendored. The executable is found through the ``VAL3DITY`` environment
variable, then ``PATH``. When it is missing the result is a failed hard gate with
``skipped_reason`` set: a skipped oracle never counts as a pass.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from roader.model.mesh import Mesh
from roader.validate.report import GateResult, Violation

# val3dity error code -> the FR-SV requirement it evidences.
CODE_REQ = {
    101: "FR-SV-5", 102: "FR-SV-5", 103: "FR-SV-1", 104: "FR-SV-4",
    201: "FR-SV-4", 202: "FR-SV-5", 203: "FR-SV-5", 204: "FR-SV-5", 205: "FR-SV-5",
    300: "FR-SV-2", 301: "FR-SV-1", 302: "FR-SV-1", 303: "FR-SV-2", 305: "FR-SV-6",
    306: "FR-SV-4", 307: "FR-SV-3", 309: "FR-SV-5",
    401: "FR-SV-4", 402: "FR-SV-5", 403: "FR-SV-6", 404: "FR-SV-6", 405: "FR-SV-3",
    501: "FR-SV-7", 502: "FR-SV-7", 503: "FR-SV-7",
}
GATE = "val3dity"


def find_val3dity() -> str | None:
    """Path of an executable val3dity, from $VAL3DITY or PATH."""
    env = os.environ.get("VAL3DITY", "").strip()
    if env:
        return env if os.path.isfile(env) and os.access(env, os.X_OK) else None
    return shutil.which("val3dity")


def write_obj(meshes: Sequence[Mesh], path: Path) -> dict[str, str]:
    """One OBJ object per mesh; returns object name -> mesh id.

    Coordinates are written with ``repr`` so val3dity sees the exact doubles.
    """
    names: dict[str, str] = {}
    offset = 1
    with open(path, "w", encoding="ascii") as fh:
        for k, mesh in enumerate(meshes):
            name = f"m{k}_" + re.sub(r"[^A-Za-z0-9_.:-]", "_", mesh.id)
            names[name] = mesh.id
            fh.write(f"o {name}\n")
            fh.writelines(f"v {x!r} {y!r} {z!r}\n" for x, y, z in mesh.vertices.tolist())
            fh.writelines(f"f {a} {b} {c}\n" for a, b, c in (mesh.faces + offset).tolist())
            offset += len(mesh.vertices)
    return names


def run_val3dity(
    meshes: Sequence[Mesh],
    *,
    exe: str | None = None,
    timeout_s: float = 1800.0,
    gate: str = GATE,
) -> GateResult:
    """Validate closed meshes as Solids and open meshes as CompositeSurfaces."""
    exe = exe or find_val3dity()
    if exe is None or not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        return GateResult(
            gate,
            passed=False,
            hard=True,
            skipped_reason="val3dity executable not found (set VAL3DITY or put it on PATH; see scripts/install_val3dity.sh)",
        )
    result = GateResult(gate, passed=True, hard=True)
    groups = (("Solid", [m for m in meshes if m.closed]), ("CompositeSurface", [m for m in meshes if not m.closed]))
    n_invalid = 0
    n_total = 0
    with tempfile.TemporaryDirectory(prefix="roader-val3dity-") as tmp:
        for primitive, group in groups:
            if not group:
                continue
            obj = Path(tmp) / f"{primitive}.obj"
            report = Path(tmp) / f"{primitive}.json"
            names = write_obj(group, obj)
            try:
                proc = subprocess.run(
                    [exe, str(obj), "-p", primitive, "-r", str(report)],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                result.passed = False
                result.skipped_reason = f"val3dity did not run: {exc}"
                return result
            if proc.returncode != 0 or not report.exists():
                result.passed = False
                result.skipped_reason = f"val3dity exited {proc.returncode} without a report: {proc.stderr.strip()[-500:]}"
                return result
            data = json.loads(report.read_text(encoding="utf-8"))
            for code in data.get("dataset_errors", []) or []:
                result.violations.append(Violation(_req(code), "dataset", f"val3dity {_describe(code)} ({primitive})"))
            seen = set()
            for feature in data.get("features", []):
                mesh_id = names.get(feature.get("id", ""), feature.get("id", ""))
                seen.add(mesh_id)
                n_total += 1
                if feature.get("validity", False):
                    continue
                n_invalid += 1
                for err in feature.get("errors", []):
                    code = int(err.get("code", 999))
                    info = err.get("info", "")
                    result.violations.append(
                        Violation(_req(code), mesh_id, f"val3dity {code} {err.get('description', '')} {info}".strip())
                    )
            for mesh_id in names.values():
                if mesh_id not in seen:
                    result.violations.append(Violation("AC-H1", mesh_id, f"val3dity did not report this {primitive}"))
            if not data.get("validity", False) and not result.violations:
                result.violations.append(Violation("AC-H1", "dataset", "val3dity reported the file invalid"))
    result.metrics = {"n_primitives": float(n_total), "n_invalid": float(n_invalid)}
    result.passed = not result.violations
    return result


def _req(code: int) -> str:
    return CODE_REQ.get(int(code), "AC-H1")


def _describe(code) -> str:
    if isinstance(code, dict):
        return f"{code.get('code', '')} {code.get('description', '')}".strip()
    return str(code)
