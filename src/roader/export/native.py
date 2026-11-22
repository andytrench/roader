"""Native model export (FR-EX-3): canonical JSON from roader.model.dumps (D-011)."""

from __future__ import annotations

from pathlib import Path

from roader.model.schema import RoadModel
from roader.model.serialize import dumps


def write_native(model: RoadModel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps(model))
    return path
