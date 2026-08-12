"""Tests verifying documentation contracts across README.md, NODES.md, ARCHITECTURE.md, and CHANGELOG.md."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CANONICAL_NAMES = [
    "01_easy_subject.json",
    "02_easy_subject_scene.json",
    "03_easy_subject_outfit.json",
    "04_easy_subject_scene_outfit.json",
    "05_easy_outfit_from_scene.json",
    "06_easy_style_transfer.json",
    "07_easy_ostris.json",
    "08_advanced_krea2_edit.json",
    "09_advanced_native.json",
    "10_advanced_ostris.json",
]


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

    # Quick Start parameter and connection assertions
    assert "target_content = empty" in content or "target_content: empty" in content or "target_content" in content
    assert "geometry_mode" in content
    assert "geometry_image" in content
    assert "CcCKrea2QwenVisionImagePrep" in content
    assert "CcCKrea2QwenVisionPrep" not in content

    for cname in CANONICAL_NAMES:
        assert cname in content, f"Canonical workflow {cname} missing from README"
        assert f"workflows/{cname}" in content

    # Canonical Ostris does NOT patch MODEL
    assert "patch_ostris_model (object patch)" not in content

def test_architecture_doc_contracts():
    content = read_repo_file("ARCHITECTURE.md")

    # Canonical Target modes
    assert "empty" in content
    assert "image" in content
    assert "fixed" in content
    assert "favor_image" in content

    # Check new requirements
    assert "Layer 3: Generic Reference Image" in content or "Layer 3 (Declarative References)**: Defines per-reference specs (Generic Reference Image)" in content
    assert "Advanced pipeline is **generic infrastructure**, while the Easy Edit node provides the **opinionated recipe**." in content
    assert "VAE-encoded target_image" in content
    assert "Global Vision Directive, negative prompt" not in content # negative context shouldn't have global vision directive

def test_nodes_doc_has_modular_node_class_names_and_fit_outcomes():
    content = read_repo_file("NODES.md")

    # Ensure stale properties are removed
    assert "appearance_reference" not in content

    # Check prompt_augmentation is only in Advanced, not Easy
    easy_section = content.split("## 2. Advanced Nodes")[0]
    advanced_section = content.split("## 2. Advanced Nodes")[1]

    assert "prompt_augmentation" not in easy_section, "prompt_augmentation must not be an Easy input"
    assert "prompt_augmentation" in advanced_section, "prompt_augmentation must be documented in Advanced"

    # Check updated schemas
    assert "target_latent" in content, "target_latent output missing from NODES.md"
    assert "vision_slot" in content, "vision_slot missing from CcCKrea2ReferenceImage in NODES.md"
    assert "masked_attention_boost" in content, "masked_attention_boost missing from CcCKrea2ReferenceImage in NODES.md"

    # Target Latent type checks
    assert "include_in_vision" in content
    assert "auto" in content
    assert "yes" in content
    assert "no" in content
    assert "PREPARED_VISION_IMAGE" in content
    assert "target_image" in content
    assert "geometry_image" in content
    assert "include_in_vision (`BOOLEAN`)" not in content
    assert "`target_image` (`PREPARED_VISION_IMAGE`)" in content
    assert "`geometry_image` (`PREPARED_VISION_IMAGE`)" in content

    required_nodes = [
        "CcCKrea2EasyEdit",
        "CcCKrea2EasyEditOstris",
        "CcCKrea2QwenVisionImagePrep",
        "CcCKrea2TargetLatent",
        "CcCKrea2ReferenceImage",
        "CcCKrea2Edit",
    ]
    for node_name in required_nodes:
        assert node_name in content, f"Node class name {node_name} not found in NODES.md"

def test_changelog_contracts():
    changelog = read_repo_file("CHANGELOG.md")

    assert "195 passing" not in changelog
    assert "01_t2i_basic" not in changelog
    assert "12_full_pipeline_composition" not in changelog
    assert "Documented README Favor Subject subject_image requirement" not in changelog
    assert "768px / 1024px without upscale" not in changelog

    assert "Ostris KV cache remains unsupported and raises NotImplementedError when requested." in changelog or "Ostris KV cache remains unsupported" in changelog

def test_markdown_links():
    import re
    # check that all workflow JSON links in README are valid
    content = read_repo_file("README.md")
    links = re.findall(r'\]\((workflows/[^\)]+\.json)\)', content)
    for link in links:
        path = os.path.join(REPO_ROOT, link)
        assert os.path.exists(path), f"Markdown link {link} in README does not point to a valid file."
