"""Tests verifying documentation contracts across README.md, NODES.md, ARCHITECTURE.md, and CHANGELOG.md."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_repo_file(filename: str) -> str:
    path = os.path.join(REPO_ROOT, filename)
    assert os.path.exists(path), f"Required documentation file missing: {filename}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_readme_links_and_quickstart_contract():
    content = read_repo_file("README.md")
    assert "[NODES.md](NODES.md)" in content or "(NODES.md)" in content
    assert "[ARCHITECTURE.md](ARCHITECTURE.md)" in content or "(ARCHITECTURE.md)" in content
    assert "[CHANGELOG.md](CHANGELOG.md)" in content or "(CHANGELOG.md)" in content
    assert "every role node patches the model" not in content.lower()

    # Quick Start parameter and connection assertions
    assert "target_content" in content
    assert "geometry_mode" in content
    assert "subject_image" in content
    assert "latent" in content

    # Check Favor Subject subject_image connection and noise explanation
    assert "subject_image` input" in content
    assert "pure noise" in content or "starts from noise" in content

    # Ensure obsolete names are not present
    for obsolete in ["Target Latent Dict", "target_latent_content", "target_geometry"]:
        assert obsolete not in content


def test_architecture_doc_contracts():
    content = read_repo_file("ARCHITECTURE.md")

    # Title
    assert "# CcC Krea2 Technical Architecture and Workflow Strategy" in content

    # Section 1 parity wording: Must NOT claim Strict Upstream Parity unconditionally
    assert "Strict Upstream Parity" not in content
    assert "Tested Upstream Parity" in content
    assert "NOTICE" in content

    # Section 5 Target Latent outputs assertions
    target_latent_sec = content[content.find("## 5. Target Latent"):content.find("## 6. Target Latent")]
    assert "latent" in target_latent_sec
    assert "latent_info" in target_latent_sec
    assert "LATENT" in target_latent_sec
    assert "STRING" in target_latent_sec

    # 5-column strategy matrix headings
    headings = ["Workflow filename", "Target Content", "Target Geometry", "References", "Main objective"]
    for heading in headings:
        assert heading in content, f"Matrix heading '{heading}' missing from ARCHITECTURE.md"

    # 9 Target Latent Content/Geometry combinations in matrix
    combinations = [
        ("empty", "favor_subject"),
        ("empty", "favor_scene"),
        ("empty", "fixed"),
        ("subject", "favor_subject"),
        ("subject", "favor_scene"),
        ("subject", "fixed"),
        ("scene", "favor_subject"),
        ("scene", "favor_scene"),
        ("scene", "fixed"),
    ]
    for c_type, g_type in combinations:
        assert f"`{c_type}`" in content or c_type in content
        assert f"`{g_type}`" in content or g_type in content

    # Section 6 matrix cross-reference combinations check
    matrix_sec = content[content.find("## 6. Target Latent"):content.find("## 7. Visual Reference")]
    subj_favor_scene_line = next(line for line in matrix_sec.splitlines() if "`subject`" in line and "`favor_scene`" in line)
    assert "workflows/04_subject_scene_edit.json" in subj_favor_scene_line
    assert "subject_image" in subj_favor_scene_line
    assert "scene_image" in subj_favor_scene_line

    scene_favor_subj_line = next(line for line in matrix_sec.splitlines() if "`scene`" in line and "`favor_subject`" in line)
    assert "workflows/04_subject_scene_edit.json" in scene_favor_subj_line
    assert "scene_image" in scene_favor_subj_line
    assert "subject_image" in scene_favor_subj_line

    # Fit mode documentation
    assert "`exact`" in content
    assert "`crop_only`" in content
    assert "`crop_and_resize`" in content
    assert "`fit`" in content

    # Fit behaviors
    assert "without resize or interpolation" in content
    assert "exact target pixel dimensions" in content
    assert "`/16`" in content

    # Indirect Style documentation assertions
    assert "At least one visual span is preserved" not in content
    assert "Every visual row belonging to an indirect Style reference is removed" in content or "removed in **one single operation**" in content
    assert "No Style visual span is forcibly preserved for an indirect Style reference" in content

    # Representations
    assert "Original Image" in content
    assert "Vision Image" in content
    assert "VAE Reference Latent" in content


def test_nodes_doc_has_modular_node_class_names_and_fit_outcomes():
    content = read_repo_file("NODES.md")

    required_nodes = [
        "CcCKrea2QwenVisionImagePrep",
        "CcCKrea2SubjectImage",
        "CcCKrea2SceneImage",
        "CcCKrea2OutfitImage",
        "CcCKrea2StyleImage",
        "CcCKrea2TargetLatent",
        "CcCKrea2Edit",
    ]
    for node_name in required_nodes:
        assert node_name in content, f"Node class name {node_name} not found in NODES.md"

    # All 4 internal outcomes documented in Auto section
    assert "**`exact`**" in content
    assert "**`crop_only`**" in content
    assert "**`crop_and_resize`**" in content
    assert "**`fit`**" in content

    # crop_only must not be described as a resize
    assert "resizes without letterboxing" not in content

    # Must not list obsolete Target Latent content modes
    assert "`inpaint`" not in content
    assert "`custom`" not in content


def test_changelog_and_notice_contracts():
    changelog = read_repo_file("CHANGELOG.md")
    assert "## [Unreleased]" in changelog
    assert "Tested Upstream Parity" in changelog or "parity" in changelog.lower()
    assert "Visual Reference Fit" in changelog or "fit" in changelog.lower()

    notice = read_repo_file("NOTICE")
    assert "c6f2a8905d4b53efcf46b7a544df20faef262ad0" in notice
    assert "5f8a02c8969b821434c442436dd534ed4461bb0e" in notice
    assert "8a4d7efb32e12a45bc89a74c102a0ef87a4192b1" in notice
    assert "a7d83f12469a918a252277d34cd0e035070081d6" in notice


def test_rebuild_workflows_fixed_mp_fallback_default():
    """Verify that scratch/rebuild_all_workflows.py uses 2.0 as fallback default for fixed_mp."""
    script_content = read_repo_file("scratch/rebuild_all_workflows.py")
    assert "fixed_mp = float(wvals[3]) if len(wvals) > 3 and isinstance(wvals[3], (int, float)) else 2.0" in script_content
