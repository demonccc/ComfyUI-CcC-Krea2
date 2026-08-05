"""Tests verifying documentation contracts across README.md, NODES.md, and ARCHITECTURE.md."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_repo_file(filename: str) -> str:
    path = os.path.join(REPO_ROOT, filename)
    assert os.path.exists(path), f"Required documentation file missing: {filename}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_readme_links_to_required_docs():
    content = read_repo_file("README.md")
    assert "[NODES.md](NODES.md)" in content or "(NODES.md)" in content
    assert "[ARCHITECTURE.md](ARCHITECTURE.md)" in content or "(ARCHITECTURE.md)" in content
    assert "[CHANGELOG.md](CHANGELOG.md)" in content or "(CHANGELOG.md)" in content
    assert "every role node patches the model" not in content.lower()


def test_architecture_doc_exists_and_has_required_sections():
    content = read_repo_file("ARCHITECTURE.md")

    # Title
    assert "# CcC Krea2 Technical Architecture and Workflow Strategy" in content

    # Target Latent 9 combinations matrix
    assert "Complete Target Latent Combination Matrix" in content
    assert "Content" in content
    assert "Geometry" in content

    # Standard Target Latent terms
    for term in ["Empty", "Subject", "Scene", "Favor Subject", "Favor Scene", "Fixed"]:
        assert term in content

    # Ensure no stale terms
    for stale_term in ["`source`", "`custom`", "`fit_subject`", "`crop_subject`"]:
        assert stale_term not in content

    # Strategy terms
    assert "Maximum Identity Continuity" in content
    assert "Freer Generation" in content
    assert "Subject reference (Slot 1)" in content or "Scene (Slot 1)" in content

    # Representations
    assert "Original Image" in content
    assert "Vision Image" in content
    assert "VAE Reference Latent" in content

    # Indexing & slot concepts
    assert "Layer 1" in content
    assert "Layer 2" in content
    assert "Layer 3" in content
    assert "Layer 4" in content
    assert "Layer 5" in content


def test_nodes_doc_has_modular_node_class_names_and_accurate_claims():
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

    # Must document auto, fit, crop
    lower_content = content.lower()
    assert "auto" in lower_content
    assert "fit" in lower_content
    assert "crop" in lower_content

    # Must not claim Vision Prep returns precomputed tokens or VAE latents
    assert "pre-computed `CCC_KREA2_PREPARED_IMAGE` tokens" not in content
    assert "precomputed tokens" not in content.lower()

    # Must not list obsolete Target Latent content modes
    assert "`inpaint`" not in content
    assert "`custom`" not in content


def test_changelog_and_notice_contracts():
    changelog = read_repo_file("CHANGELOG.md")
    assert "## [Unreleased]" in changelog

    notice = read_repo_file("NOTICE")
    assert "c6f2a8905d4b53efcf46b7a544df20faef262ad0" in notice
    assert "5f8a02c8969b821434c442436dd534ed4461bb0e" in notice
    assert "8a4d7efb32e12a45bc89a74c102a0ef87a4192b1" in notice
    assert "a7d83f12469a918a252277d34cd0e035070081d6" in notice

