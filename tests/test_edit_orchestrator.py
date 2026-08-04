"""Unit tests for Edit orchestrator node execution."""

import torch
from unittest.mock import MagicMock
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import create_target_latent
from ccc_krea2.modular_nodes.subject_node import CcCKrea2SubjectImage
from ccc_krea2.modular_nodes.scene_node import CcCKrea2SceneImage
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit


def test_edit_orchestrator_execution():
    # Setup mock clip and vae
    mock_clip = MagicMock()
    mock_clip.tokenize.return_value = ["token1"]
    mock_clip.encode_from_tokens_scheduled.return_value = [["cond"]]

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
    assert "Logical Role: Subject" in edit_info
    assert "Logical Role: Scene" in edit_info
    assert "Global Vision Directive: yes" in edit_info
