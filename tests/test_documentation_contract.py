"""Tests verifying documentation contracts across README.md, NODES.md, and ARCHITECTURE.md."""

import os
import pytest

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


def test_architecture_doc_exists_and_has_required_sections():
    content = read_repo_file("ARCHITECTURE.md")
    
    # Title
    assert "# CcC Krea2 Technical Architecture and Workflow Strategy" in content
    
    # Target Latent 9 combinations matrix
    assert "Complete Target Latent Combination Matrix" in content
    assert "Latent Source" in content and "Output Resolution" in content
    assert "empty" in content and "subject" in content and "scene" in content
    
    # Practical workflow strategies
    assert "Max Identity Preservation" in content
    
    # Distinctions between representations and indexings
    assert "original_image" in content and "vision_image" in content and "vae_reference_latent" in content
    assert "Logical Slots" in content or "logical slot" in content
    assert "Physical Qwen Images" in content or "physical Qwen image" in content
    assert "VAE Reference Frames" in content or "VAE frame" in content


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
    assert "auto" in content and "fit" in content and "crop" in content
    
    # Must not claim Vision Prep returns precomputed tokens
    assert "pre-computed `CCC_KREA2_PREPARED_IMAGE` tokens" not in content
    assert "precomputed tokens" not in content.lower()
    
    # Must not list obsolete Target Latent content modes
    assert "`inpaint`" not in content and "`custom`" not in content or "Latent Content Source" not in content
