"""Unit tests for Edit orchestrator node execution."""

import torch
import pytest
from unittest.mock import MagicMock
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import create_target_latent
from ccc_krea2.modular_nodes.subject_node import CcCKrea2SubjectImage
from ccc_krea2.modular_nodes.scene_node import CcCKrea2SceneImage
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit


def test_edit_orchestrator_execution():
    # Setup mock clip and vae
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

    # 1 text token + 256 rows (img1 512x512) + 256 rows (img2 512x512) = 513 rows
    cond_tensor = torch.randn(1, 513, 1536)
    mock_clip.encode_from_tokens_scheduled.return_value = [[cond_tensor, {}]]

    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    # Build inputs
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="auto")
    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="auto", previous_references=chain1)

    target_lat, _ = create_target_latent(
        vae=mock_vae,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=0.5,
        fixed_aspect_ratio="1:1"
    )

    edit_node = CcCKrea2Edit()
    model_out, pos_out, neg_out, lat_out, edit_info = edit_node.process(
        model=mock_model,
        clip=mock_clip,
        vae=mock_vae,
        references=chain2,
        target_latent=target_lat,
        positive_prompt="a photo of a person",
        negative_prompt="blurry",
        global_vision_directive="High quality"
    )

    assert pos_out is not None
    assert neg_out is not None
    assert lat_out == target_lat
    assert "Subject" in edit_info
    assert "Scene" in edit_info
    assert "Global Vision Directive Active: yes" in edit_info


def test_edit_info_outfit_anchor_reporting():
    """Assert Outfit edit_info reporting lists Outfit Reference."""
    from ccc_krea2.modular_nodes.outfit_node import CcCKrea2OutfitImage

    mock_clip = MagicMock()
    mock_clip.tokenize.return_value = {"qwen3vl": [[[{"type": "image", "data": torch.rand(1, 512, 512, 3)}, None]]]}
    mock_clip.encode_from_tokens_scheduled.return_value = [[[torch.randn(1, 257, 1536), {}]]]

    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")

    outfit_node = CcCKrea2OutfitImage()
    (chain,) = outfit_node.process(prepared_image=prep, vision_slot="auto")

    target_lat, _ = create_target_latent(
        vae=mock_vae,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=0.5,
        fixed_aspect_ratio="1:1"
    )

    edit_node = CcCKrea2Edit()
    _, _, _, _, edit_info = edit_node.process(
        model=mock_model,
        clip=mock_clip,
        vae=mock_vae,
        references=chain,
        target_latent=target_lat,
        positive_prompt="a model wearing outfit",
        negative_prompt="",
    )

    assert "Reference [Slot 1 - Outfit]:" in edit_info



def test_edit_orchestrator_execution_native_5d_latent():
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
    cond_tensor = torch.randn(1, 513, 1536)
    mock_clip.encode_from_tokens_scheduled.return_value = [[cond_tensor, {}]]

    # Mock VAE returns native 5D reference latents [1, 16, 1, 64, 64]
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 1, 64, 64))}

    mock_model = MagicMock()
    mock_model.clone.return_value = mock_model

    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")

    subj_node = CcCKrea2SubjectImage()
    (chain1,) = subj_node.process(prepared_image=prep, vision_slot="auto")
    scene_node = CcCKrea2SceneImage()
    (chain2,) = scene_node.process(prepared_image=prep, vision_slot="auto", previous_references=chain1)

    target_lat = {
        "samples": torch.zeros((1, 16, 1, 144, 216)),
        "batch_index": [0]
    }

    edit_node = CcCKrea2Edit()
    model_out, pos_out, neg_out, lat_out, edit_info = edit_node.process(
        model=mock_model,
        clip=mock_clip,
        vae=mock_vae,
        references=chain2,
        target_latent=target_lat,
        positive_prompt="a photo of a person in a room",
        negative_prompt="blurry",
        global_vision_directive="High quality"
    )

    assert pos_out is not None
    assert neg_out is not None
    assert lat_out is target_lat
    assert lat_out["samples"].shape == (1, 16, 1, 144, 216)
    assert "Target Latent Geometry: 216 x 144 (Batch Size: 1)" in edit_info
    assert "Subject" in edit_info
    assert "Scene" in edit_info


def test_edit_orchestrator_invalid_latent_dimensions_raises():
    mock_clip = MagicMock()
    mock_vae = MagicMock()
    mock_model = MagicMock()

    subj_node = CcCKrea2SubjectImage()
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=mock_clip, mode="native")
    (chain,) = subj_node.process(prepared_image=prep, vision_slot="auto")

    edit_node = CcCKrea2Edit()

    with pytest.raises(ValueError, match="Target latent samples must be a 4D or 5D tensor"):
        edit_node.process(
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            references=chain,
            target_latent={"samples": torch.zeros((16, 144, 216))},
            positive_prompt="test",
            negative_prompt=""
        )

    with pytest.raises(ValueError, match="Target latent samples must be a 4D or 5D tensor"):
        edit_node.process(
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            references=chain,
            target_latent={"samples": torch.zeros((1, 16, 1, 1, 144, 216))},
            positive_prompt="test",
            negative_prompt=""
        )

