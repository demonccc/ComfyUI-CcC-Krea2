"""Documentation contract for the current Krea2 CcC Edit architecture."""

from pathlib import Path


def _read(name):
    return Path(name).read_text(encoding="utf-8")


def test_readme_documents_current_split_edit_and_single_runtime():
    readme = _read("README.md")
    assert "Krea2 CcC Visual Reference" in readme
    assert "Krea2 CcC Semantic Reference" in readme
    assert "Krea2 CcC Edit" in readme
    assert "one Krea2 CcC Edit runtime" in readme
    assert "workflows/01_scene_subject.json" in readme


def test_nodes_doc_matches_current_visual_and_semantic_controls():
    nodes = _read("NODES.md")
    assert "semantic_grounding_px" in nodes
    assert "step 32" in nodes
    assert "380" in nodes
    assert "384" in nodes
    assert "prompt_annotation" in nodes
    assert "semantic_only" in nodes
    assert "Krea2 CcC Semantic Reference" in nodes
    assert "negative_prompt" not in nodes


def test_architecture_documents_single_runtime_and_separate_qwen_vae_paths():
    architecture = _read("ARCHITECTURE.md")
    assert "Single Edit Runtime" in architecture
    assert "Qwen path" in architecture
    assert "appearance path" in architecture
    assert "[text | reference 1 | reference 2 | ... | target]" in architecture


def test_removed_cache_and_geometry_nodes_are_not_documented_as_public():
    nodes = _read("NODES.md")
    readme = _read("README.md")
    architecture = _read("ARCHITECTURE.md")

    assert "Krea2 CcC Reference Cache Create" not in nodes
    assert "Krea2 CcC Cached Visual Reference" not in nodes
    assert "## Reference Cache" not in readme
    assert "## Reference cache" not in architecture
    assert "## Krea2 CcC Paint Geometry" not in nodes
    assert "Paint Prepare" in readme
    assert "paint_geometry" in nodes
