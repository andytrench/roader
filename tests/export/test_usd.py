"""Text USD export (FR-EX-3, FR-DM-3, FR-AS-10)."""

import re

from roader.export import usda_text


def test_fr_ex_3_usda_header_and_every_mesh(model, meshes):
    text = usda_text(model, meshes)
    assert text.startswith("#usda 1.0\n")
    assert 'upAxis = "Z"' in text
    assert "metersPerUnit = 1" in text
    assert model.frame.proj in text
    for mesh in meshes:
        assert f'string mesh_id = "{mesh.id}"' in text
        assert f'string element_id = "{mesh.element_id}"' in text
    prims = re.findall(r'def Mesh "([A-Za-z_][A-Za-z0-9_]*)"', text)
    assert len(prims) == len(meshes)
    for layer in {m.layer for m in meshes}:
        assert f'def Scope "{layer}"' in text


def test_fr_dm_3_usda_custom_data_per_prim(model, meshes):
    text = usda_text(model, meshes)
    blocks = text.split('def Mesh "')[1:]
    assert len(blocks) == len(meshes)
    for block in blocks:
        for key in ("element_id", "layer", "part", "material", "source"):
            assert re.search(rf"string {key} = \"[^\"]+\"", block), key
        assert "rel material:binding = </Unit/Materials/" in block
    assert text.count("{") == text.count("}")
