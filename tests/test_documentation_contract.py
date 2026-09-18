"""Documentation contract for the current CcC Krea2 architecture."""

from pathlib import Path


def _read(name):
    return Path(name).read_text(encoding="utf-8")


def test_readme_documents_current_split_edit_and_single_runtime():
    readme = _read("README.md")
    assert "Krea2 CcC Visual Reference" in readme
    assert "Krea2 CcC Semantic Reference" in readme
    assert "Krea2 CcC Edit" in readme
    assert "one CcC edit runtime" in readme
    assert "workflows/01_scene_subject.json" in readme


def test_nodes_doc_matches_current_visual_and_semantic_controls():
    nodes = _read("NODES.md")
    assert "semantic_grounding_px" in nodes
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
