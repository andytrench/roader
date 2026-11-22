"""Text USD (.usda) writer (FR-EX-3, FR-AS-10).

upAxis "Z", metersPerUnit 1: points are the unit frame as-is (x east, y north,
z up), so no axis conversion is applied. One Mesh prim per Mesh under
/Unit/<layer>/<id-safe name>, customData carrying {mesh_id, element_id, layer,
part, material, source, confidence} (FR-DM-3, AC-H6). One UsdPreviewSurface
material per semantic slot under /Unit/Materials (FR-AS-6). The geodetic anchor
and PROJ string are in the layer's customLayerData.

FR-AS-3..5 attributes become primvars: ``normals``, ``st`` (uv0, metres) and ``st1``
(uv1 atlas), faceVarying when stored per face corner. With an LOD chain (FR-AS-1)
the layer scopes sit in a ``lod`` variant set on /Unit, one variant per level.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    uv1_statement,
)
from roader.model.mesh import Mesh, corner_values
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


def _vec(values: np.ndarray, digits: int) -> str:
    return "[" + ", ".join("(" + ", ".join(num(c, digits) for c in row) + ")" for row in values) + "]"


def _primvars(mesh: Mesh, indent: str) -> list[str]:
    """FR-AS-3..5 attributes as primvars: per corner -> faceVarying, per vertex -> vertex."""
    lines = []
    for name, value, tp, width, digits in (
        ("normals", mesh.normals, "normal3f[]", 3, 7),
        ("st", mesh.uv0, "texCoord2f[]", 2, 6),
        ("st1", mesh.uv1, "texCoord2f[]", 2, 7),
    ):
        if value is None:
            continue
        arr = np.asarray(value)
        interpolation = "faceVarying" if arr.ndim == 3 else "vertex"
        flat = corner_values(mesh, arr, width).reshape(-1, width) if arr.ndim == 3 else arr
        lines += [
            f"{indent}{tp} primvars:{name} = {_vec(flat, digits)} (",
            f'{indent}    interpolation = "{interpolation}"',
            f"{indent})",
        ]
    return lines


def _layer_scopes(meshes: list[Mesh], index: dict, slot_names: dict[str, str], indent: str) -> list[str]:
    lines: list[str] = []
    inner = indent + "    "
    body = inner + "    "
    for layer in sorted({m.layer for m in meshes}):
        lines += ["", f'{indent}def Scope "{layer}"', f"{indent}{{"]
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
                f'{inner}def Mesh "{name}" (',
                f'{body}prepend apiSchemas = ["MaterialBindingAPI"]',
                f"{body}customData = {{",
            ]
            lines.extend(_dictionary(node_extras(mesh, index), body + "    "))
            lines += [
                f"{body}}}",
                f"{inner})",
                f"{inner}{{",
                (
                    f"{body}float3[] extent = [({num(lo[0], 6)}, {num(lo[1], 6)}, {num(lo[2], 6)}), "
                    f"({num(hi[0], 6)}, {num(hi[1], 6)}, {num(hi[2], 6)})]"
                ),
                f"{body}int[] faceVertexCounts = [" + ", ".join("3" for _ in range(len(mesh.faces))) + "]",
                f"{body}int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in mesh.faces.reshape(-1)) + "]",
                f"{body}rel material:binding = </Unit/Materials/{slot_names[mesh.material]}>",
                f'{body}uniform token orientation = "rightHanded"',
                f"{body}point3f[] points = {_points(v)}",
            ]
            lines += _primvars(mesh, body)
            lines += [
                f'{body}uniform token subdivisionScheme = "none"',
                f"{inner}}}",
            ]
        lines.append(f"{indent}}}")
    return lines


def usda_text(model: RoadModel, meshes: list[Mesh], *, lods: Mapping[int, Sequence[Mesh]] | None = None) -> str:
    """The unit as .usda; ``lods`` (level -> meshes, FR-AS-1) become the variants of a
    ``lod`` variant set on /Unit, "lod0" (``meshes``) selected."""
    meshes = check_meshes(meshes)
    levels = {0: meshes, **{k: check_meshes(v) for k, v in sorted((lods or {}).items()) if k > 0}}
    index = element_index(model)
    header = unit_header(model)
    every = [m for level in levels.values() for m in level]

    lines = ["#usda 1.0", "("]
    lines.append("    customLayerData = {")
    roader = {**header, "generator": GENERATOR}
    if any(m.uv1 is not None for m in every):
        roader["uv1_atlas"] = uv1_statement()
    lines.extend(_dictionary({"roader": roader}, "        "))
    lines.append("    }")
    lines += ['    defaultPrim = "Unit"', "    metersPerUnit = 1", '    upAxis = "Z"', ")", ""]
    if len(levels) > 1:
        lines += [
            'def Xform "Unit" (',
            '    kind = "assembly"',
            "    variants = {",
            '        string lod = "lod0"',
            "    }",
            '    prepend variantSets = "lod"',
            ")",
            "{",
        ]
    else:
        lines += ['def Xform "Unit" (', '    kind = "assembly"', ")", "{"]

    slots = sorted({m.material for m in every})
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

    if len(levels) > 1:
        lines += ["", '    variantSet "lod" = {']
        for level, level_meshes in levels.items():
            lines.append(f'        "lod{level}" {{')
            lines.extend(_layer_scopes(level_meshes, index, slot_names, "            "))
            lines.append("        }")
        lines.append("    }")
    else:
        lines.extend(_layer_scopes(meshes, index, slot_names, "    "))
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_usda(model: RoadModel, meshes: list[Mesh], path: Path, *, lods: Mapping[int, Sequence[Mesh]] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(usda_text(model, meshes, lods=lods).encode("utf-8"))
    return path
