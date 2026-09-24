"""Documentation contract for the current public Krea2 CcC architecture."""

from pathlib import Path

from ccc_krea2.nodes import NODE_DISPLAY_NAME_MAPPINGS


def _read(name):
    return Path(name).read_text(encoding="utf-8")


def _section(text, heading):
    marker = f"## {heading}\n"
    start = text.index(marker)
    rest = text[start + len(marker):]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def test_readme_tracks_current_public_surface_and_workflows():
    readme = _read("README.md")
    assert "14 nodes" in readme
    assert "Krea2 CcC Edit Prompt Creator" in readme
    assert "Krea2 CcC Character Sheet" in readme
    assert "Krea2 CcC Paint Prepare" in readme
    assert "workflows/01_scene_subject.json" in readme
    assert "workflows/02_anypaint_remove_people.json" in readme
    assert "workflows/03_regional_attention.json" in readme
    assert "workflows/04_character_sheet_identity.json" in readme
    assert "workflows/05_edit_prompt_creator.json" in readme


def test_all_public_node_display_names_are_documented():
    nodes = _read("NODES.md")
    readme = _read("README.md")

    assert len(NODE_DISPLAY_NAME_MAPPINGS) == 14
    for display_name in NODE_DISPLAY_NAME_MAPPINGS.values():
        assert display_name in nodes
        assert display_name in readme


def test_nodes_doc_matches_current_edit_contracts():
    nodes = _read("NODES.md")

    assert "semantic_grounding_px" in nodes
    assert "contain" in _section(nodes, "Krea2 CcC Visual Reference")
    assert "prompt_annotation" in nodes
    assert "semantic_only" in nodes
    assert "create_from_image" in nodes
    assert "create_from_theme" in nodes
    assert "paint_geometry" in _section(nodes, "Krea2 CcC Paint Prepare")

    edit = _section(nodes, "Krea2 CcC Edit")
    assert "positive_prompt" in edit
    assert "negative_prompt" not in edit
    assert "fixed internally to an empty string" in edit


def test_removed_nodes_are_not_documented_as_public_sections():
    nodes = _read("NODES.md")
    assert "## Krea2 CcC Paint Geometry" not in nodes
    assert "## Krea2 CcC Reference Cache Create" not in nodes
    assert "## Krea2 CcC Reference Cache Save" not in nodes
    assert "## Krea2 CcC Reference Cache Load" not in nodes
    assert "## Krea2 CcC Cached Visual Reference" not in nodes


def test_architecture_matches_single_edit_and_three_node_paint_paths():
    architecture = _read("ARCHITECTURE.md")
    assert "Single Edit Runtime" in architecture
    assert "Qwen path" in architecture
    assert "appearance path" in architecture
    assert "[text | reference 1 | reference 2 | ... | target]" in architecture
    assert "Paint Prepare" in architecture
    assert "Paint Restore" in architecture
    assert "no public Reference Cache node family" in architecture
    assert "no public Paint Geometry node" in architecture
