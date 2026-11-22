"""Text USD (.usda) writer (FR-EX-3, FR-AS-10).

upAxis "Z", metersPerUnit 1: points are the unit frame as-is (x east, y north,
z up), so no axis conversion is applied. One Mesh prim per Mesh under
/Unit/<layer>/<id-safe name>, customData carrying {mesh_id, element_id, layer,
part, material, source, confidence} (FR-DM-3, AC-H6). One UsdPreviewSurface
material per semantic slot under /Unit/Materials (FR-AS-6). The geodetic anchor
and PROJ string are in the layer's customLayerData.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from roader.export._common import (
    GENERATOR,
    check_meshes,
    element_index,
    material_pbr,
    node_extras,
    num,
    safe_name,
    unit_header,
)
from roader.model.mesh import Mesh
from roader.model.schema import RoadModel


def _str(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def _value(value: Any) -> tuple[str, str]:
    """USD type name and literal for a customData / customLayerData entry."""
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "double", num(value, 9)
    if isinstance(value, (list, tuple)):
        return "double[]", "[" + ", ".join(num(v, 9) for v in value) + "]"
    return "string", _str(str(value))


def _dictionary(entries: dict[str, Any], indent: str) -> list[str]:
    lines = []
    for key in sorted(entries):
        value = entries[key]
        if isinstance(value, dict):
            lines.append(f"{indent}dictionary {key} = {{")
            lines.extend(_dictionary(value, indent + "    "))
            lines.append(f"{indent}}}")
        else:
            tp, literal = _value(value)
            lines.append(f"{indent}{tp} {key} = {literal}")
    return lines


def _points(vertices: np.ndarray) -> str:
    return "[" + ", ".join(f"({num(x, 6)}, {num(y, 6)}, {num(z, 6)})" for x, y, z in vertices) + "]"


def usda_text(model: RoadModel, meshes: list[Mesh]) -> str:
    meshes = check_meshes(meshes)
    index = element_index(model)
    header = unit_header(model)

    lines = ["#usda 1.0", "("]
    lines.append("    customLayerData = {")
    lines.extend(_dictionary({"roader": {**header, "generator": GENERATOR}}, "        "))
    lines.append("    }")
    lines += ['    defaultPrim = "Unit"', "    metersPerUnit = 1", '    upAxis = "Z"', ")", ""]
    lines += ['def Xform "Unit" (', '    kind = "assembly"', ")", "{"]

    slots = sorted({m.material for m in meshes})
    slot_names = {slot: safe_name(slot) for slot in slots}
    lines += ['    def Scope "Materials"', "    {"]
    for slot in slots:
        name = slot_names[slot]
        rgba, metallic, roughness = material_pbr(slot)
        path = f"/Unit/Materials/{name}"
        lines += [
            f'        def Material "{name}"',
            "        {",
            f"            token outputs:surface.connect = <{path}/Shader.outputs:surface>",
            "",
            '            def Shader "Shader"',
            "            {",
            '                uniform token info:id = "UsdPreviewSurface"',
            f"                color3f inputs:diffuseColor = ({num(rgba[0], 6)}, {num(rgba[1], 6)}, {num(rgba[2], 6)})",
            f"                float inputs:metallic = {num(metallic, 6)}",
            f"                float inputs:roughness = {num(roughness, 6)}",
            "                token outputs:surface",
            "            }",
            "        }",
        ]
    lines.append("    }")

    for layer in sorted({m.layer for m in meshes}):
        lines += ["", f'    def Scope "{layer}"', "    {"]
        used: set[str] = set()
        for mesh in (m for m in meshes if m.layer == layer):
            name = safe_name(mesh.id)
            base, n = name, 1
            while name in used:
                name = f"{base}_{n}"
                n += 1
            used.add(name)
            v = mesh.vertices
            lo, hi = v.min(axis=0), v.max(axis=0)
            lines += [
                f'        def Mesh "{name}" (',
                '            prepend apiSchemas = ["MaterialBindingAPI"]',
                "            customData = {",
            ]
            lines.extend(_dictionary(node_extras(mesh, index), "                "))
            lines += [
                "            }",
                "        )",
                "        {",
                (
                    f"            float3[] extent = [({num(lo[0], 6)}, {num(lo[1], 6)}, {num(lo[2], 6)}), "
                    f"({num(hi[0], 6)}, {num(hi[1], 6)}, {num(hi[2], 6)})]"
                ),
                "            int[] faceVertexCounts = [" + ", ".join("3" for _ in range(len(mesh.faces))) + "]",
                "            int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in mesh.faces.reshape(-1)) + "]",
                f"            rel material:binding = </Unit/Materials/{slot_names[mesh.material]}>",
                '            uniform token orientation = "rightHanded"',
                f"            point3f[] points = {_points(v)}",
                '            uniform token subdivisionScheme = "none"',
                "        }",
            ]
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_usda(model: RoadModel, meshes: list[Mesh], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(usda_text(model, meshes).encode("utf-8"))
    return path
