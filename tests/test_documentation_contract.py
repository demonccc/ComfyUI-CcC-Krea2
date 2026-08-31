"""Documentation contract for the split Edit architecture."""

from pathlib import Path


def _read(name):
    return Path(name).read_text(encoding="utf-8")


def test_readme_documents_split_edit_and_single_workflow():
    readme = _read("README.md")
    assert "Krea2 CcC Visual Reference" in readme
    assert "Krea2 CcC Semantic Reference" in readme
    assert "Krea2 CcC Edit" in readme
    assert "workflows/01_scene_subject.json" in readme
    assert "Easy Edit" not in readme
    assert "Edit Advanced" not in readme


def test_nodes_doc_describes_semantic_role_behavior():
    nodes = _read("NODES.md")
    assert "semantic_role" in nodes
    assert "empty" in nodes.lower()
    assert "positional" in nodes.lower()
    assert "semantic = false" in nodes
    assert "Krea2 CcC Semantic Reference" in nodes


def test_architecture_documents_scene_subject_order():
    architecture = _read("ARCHITECTURE.md")
    assert "scene -> subject" in architecture
    assert "visual references" in architecture.lower()
    assert "semantic references" in architecture.lower()
