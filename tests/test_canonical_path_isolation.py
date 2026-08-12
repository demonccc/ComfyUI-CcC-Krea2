"""Tests asserting legacy prompt-builder isolation and physical Qwen image count matching."""

import torch
import pytest
from unittest.mock import MagicMock, patch
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import create_target_latent
from ccc_krea2.modular_nodes.reference_node import CcCKrea2ReferenceImage
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit
from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit
import ccc_krea2.conditioning as conditioning


def test_legacy_functions_not_called_in_canonical_path():
    """Section 7: Verify that build_krea2_qwen_template and build_annotated_user_prompt are NEVER called during canonical edit orchestrator execution."""
    mock_clip = MagicMock()

    def mock_tokenize(prompt, images=None, **kwargs):
        tok_pairs = []
        if images:
            for img in images:
                tok_pairs.append([{"type": "image", "data": img}, None])
        else:
            tok_pairs.append([100, None])
        return {"qwen3vl": [tok_pairs]}

    mock_clip.tokenize.side_effect = mock_tokenize
    cond_tensor = torch.randn(1, 2000, 1536)
    mock_clip.encode_from_tokens_scheduled.return_value = [[cond_tensor, {}]]

    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")

    ref_node = CcCKrea2ReferenceImage()
    (chain,) = ref_node.process(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
        vision_instruction="Use subject identity"
    )

    target_lat, _ = create_target_latent(
        vae=mock_vae,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=0.5,
        fixed_aspect_ratio="1:1"
    )

    edit_node = CcCKrea2Edit()

    with patch.object(conditioning, "build_krea2_qwen_template", side_effect=RuntimeError("Legacy build_krea2_qwen_template called!")), \
         patch.object(conditioning, "build_annotated_user_prompt", side_effect=RuntimeError("Legacy build_annotated_user_prompt called!")):

        # Execute Advanced Edit (canonical path)
        model_out, pos_out, neg_out, lat_out, edit_info = edit_node.process(
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            references=chain,
            target_latent=target_lat,
            positive_prompt="a photo of a person",
            negative_prompt="blurry"
        )
        assert pos_out is not None

        # Execute Easy Edit with a FRESH model (not the already-patched one)
        # Chaining would raise RuntimeError — we test that separately below
        fresh_model = MagicMock()
        fresh_model.clone.return_value = fresh_model
        easy_node = CcCKrea2EasyEdit()
        easy_out = easy_node.process(
            model=fresh_model,
            clip=mock_clip,
            vae=mock_vae,
            positive_prompt="a photo",
            preset="balanced",
            subject=img
        )
        assert easy_out[0] is not None


def test_patch_krea2_model_raises_on_already_patched():
    """Verify that patch_krea2_model raises RuntimeError when MODEL is already patched."""
    from ccc_krea2.patch import patch_krea2_model, is_model_already_patched

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    img = torch.rand(1, 64, 64, 3)
    prep = prepare_vision_image(image=img, clip=MagicMock(), mode="native")
    from ccc_krea2.references import PreparedReference, ReferenceRole
    dummy_ref = PreparedReference(
        role=ReferenceRole.SUBJECT,
        grounding_image=img,
        vae_latent=torch.zeros((1, 16, 8, 8)),
        spatial_attention_mask=None,
        boost=1.0,
    )

    # First patch succeeds
    patched = patch_krea2_model(mock_model, [dummy_ref])
    # Second patch attempt on the already-patched model must raise
    with pytest.raises(RuntimeError, match="already patched"):
        patch_krea2_model(patched, [dummy_ref])


def test_physical_image_count_matches_vision_markers():
    """Section 8: Verify physical_images count matches number of vision markers in user content."""
    mock_clip = MagicMock()
    tokenize_calls = []

    def mock_tokenize(prompt, images=None, **kwargs):
        tok_pairs = []
        imgs = images or []
        tokenize_calls.append({"prompt": prompt, "images": imgs})
        for img in imgs:
            tok_pairs.append([{"type": "image", "data": img}, None])
        if not tok_pairs:
            tok_pairs.append([100, None])
        return {"qwen3vl": [tok_pairs]}

    mock_clip.tokenize.side_effect = mock_tokenize
    mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]

    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")

    ref_node = CcCKrea2ReferenceImage()
    (chain1,) = ref_node.process(
        reference_path="edit",
        prepared_image=prep,
        alias="subject",
        vision_instruction="Subject identity"
    )
    (chain2,) = ref_node.process(
        reference_path="style",
        prepared_image=prep,
        style_processing="2x2",
        previous_references=chain1
    )

    target_lat, _ = create_target_latent(
        vae=mock_vae,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=0.5,
        fixed_aspect_ratio="1:1"
    )

    edit_node = CcCKrea2Edit()
    edit_node.process(
        model=mock_model,
        clip=mock_clip,
        vae=mock_vae,
        references=chain2,
        target_latent=target_lat,
        positive_prompt="Photo of subject in style",
        negative_prompt=""
    )

    # First tokenize call is positive prompt:
    pos_call = tokenize_calls[0]
    pos_prompt = pos_call["prompt"]
    pos_images = pos_call["images"]

    # 1 edit ref + 4 style 2x2 crops = 5 physical images
    assert len(pos_images) == 5
    # Number of vision pad tokens in prompt must match physical images count
    assert pos_prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == len(pos_images)
