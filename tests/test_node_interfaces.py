"""Unit tests for node class mappings, widget signatures, settings chaining, and role instructions."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ccc_krea2.conditioning import build_role_instructions
from ccc_krea2.constants import ReferenceRole
from ccc_krea2.prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION


def test_node_mappings_count_and_keys():
    assert len(NODE_CLASS_MAPPINGS) == 22
    assert len(NODE_DISPLAY_NAME_MAPPINGS) == 22

    expected_keys = [
        # Modular Reference Pipeline Nodes
        "CcCKrea2QwenVisionImagePrep",
        "CcCKrea2TargetLatent",
        "CcCKrea2SubjectImage",
        "CcCKrea2SceneImage",
        "CcCKrea2OutfitImage",
        "CcCKrea2StyleImage",
        "CcCKrea2ReferenceImage",
        "CcCKrea2Edit",
        "CcCKrea2EasyEdit",
        "CcCKrea2EasyEditOstris",

        # Existing Nodes
        "CcCKrea2Subject",
        "CcCKrea2SubjectOutfit",
        "CcCKrea2SubjectScene",
        "CcCKrea2SubjectSceneOutfit",
        "CcCKrea2Inpaint",
        "CcCKrea2InpaintSubjectOutfit",
        "CcCKrea2InpaintSubjectScene",
        "CcCKrea2ImageAdvancedSettings",
        "CcCKrea2EditAdvancedSettings",
        "CcCKrea2LoRAPromptSettings",
        "CcCKrea2LoRAStack",
        "CcCKrea2TextToImage",
    ]
    for k in expected_keys:
        assert k in NODE_CLASS_MAPPINGS
        assert k in NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2LoRAPromptSettings"] == "CcC Krea2 - LoRA Prompt Settings"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2LoRAStack"] == "CcC Krea2 - LoRA Stack"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2TextToImage"] == "CcC Krea2 - Text to Image"


def test_t2i_node_interface_contract():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2TextToImage"]
    assert cls.CATEGORY == "CcC/Krea2"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2TextToImage"] == "CcC Krea2 - Text to Image"
    assert cls.RETURN_NAMES == ("model", "positive", "negative", "latent")
    assert cls.RETURN_TYPES == ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")

    inputs = cls.INPUT_TYPES()
    req = inputs.get("required", {})
    opt = inputs.get("optional", {})
    all_inputs = {**req, **opt}

    # Required inputs include model, clip, prompt, aspect_ratio, megapixels, batch_size
    for req_inp in ("model", "clip", "prompt", "aspect_ratio", "megapixels", "batch_size"):
        assert req_inp in req, f"Missing required input '{req_inp}' on CcCKrea2TextToImage"

    # Optional inputs include negative_prompt, prompt_augmentation
    assert "negative_prompt" in opt
    assert "prompt_augmentation" in opt

    # No VAE, image, mask, image_advanced_settings, edit_advanced_settings
    for forbidden in (
        "vae",
        "subject_image",
        "scene_image",
        "outfit_image",
        "source_image",
        "image",
        "mask",
        "subject_attention_mask",
        "image_advanced_settings",
        "edit_advanced_settings",
    ):
        assert forbidden not in all_inputs, f"Forbidden input '{forbidden}' found in CcCKrea2TextToImage"



def test_main_nodes_widget_signature_refactor():
    main_node_keys = [
        "CcCKrea2Subject",
        "CcCKrea2SubjectOutfit",
        "CcCKrea2SubjectScene",
        "CcCKrea2SubjectSceneOutfit",
        "CcCKrea2Inpaint",
        "CcCKrea2InpaintSubjectOutfit",
        "CcCKrea2InpaintSubjectScene",
    ]

    removed_widgets = [
        "subject_boost",
        "scene_boost",
        "outfit_boost",
        "source_boost",
        "subject_mask_invert",
        "attention_mask_mode",
        "subject_grounding_preset",
        "subject_grounding_px",
        "width",
        "height",
        "batch_size",
        "sampling_resize_mode",
        "reference_fit_mode",
        "inpaint_mask_invert",
    ]

    new_widgets = [
        "preset",
        "output_resolution",
        "megapixels",
        "image_advanced_settings",
        "edit_advanced_settings",
        "prompt_augmentation",
    ]

    for key in main_node_keys:
        cls = NODE_CLASS_MAPPINGS[key]
        inputs = cls.INPUT_TYPES()
        req = inputs.get("required", {})
        opt = inputs.get("optional", {})
        all_inputs = {**req, **opt}

        # Check removed widgets are NOT present
        for rm in removed_widgets:
            assert rm not in all_inputs, f"Widget '{rm}' should be removed from node '{key}'"

        # Check new widgets ARE present
        for nw in new_widgets:
            assert nw in all_inputs, f"New widget '{nw}' must be present in node '{key}'"

        # Check prompt_augmentation socket type
        assert opt["prompt_augmentation"] == (CCC_KREA2_PROMPT_AUGMENTATION,)


def test_image_advanced_settings_node_widgets_and_chaining():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2ImageAdvancedSettings"]
    inputs = cls.INPUT_TYPES()
    req = inputs["required"]

    # 1. Prove override widgets no longer exist
    assert "override_attention" not in req
    assert "override_grounding" not in req
    assert "override_reference_geometry" not in req

    # 2. Prove all remaining image settings widgets remain present
    expected_widgets = [
        "role",
        "boost",
        "mask_invert",
        "grounding_resize_mode",
        "grounding_px",
        "grounding_min_px",
        "grounding_max_px",
        "grounding_resize_method",
        "reference_fit_mode",
        "reference_resize_method",
    ]
    for w in expected_widgets:
        assert w in req, f"Widget '{w}' must be present in CcCKrea2ImageAdvancedSettings"

    node = cls()

    # Step 1: Configure subject
    out1 = node.process(
        role="subject",
        boost=3.0,
        mask_invert=False,
        grounding_resize_mode="normalize",
        grounding_px=768,
        grounding_min_px=512,
        grounding_max_px=1024,
        grounding_resize_method="auto",
        reference_fit_mode="fit",
        reference_resize_method="auto",
        image_advanced_settings=None,
    )
    bundle1 = out1[0]
    assert "subject" in bundle1.role_settings
    assert bundle1.role_settings["subject"].boost == 3.0

    # Step 2: Chain scene configuration
    out2 = node.process(
        role="scene",
        boost=1.5,
        mask_invert=False,
        grounding_resize_mode="normalize",
        grounding_px=768,
        grounding_min_px=512,
        grounding_max_px=1024,
        grounding_resize_method="auto",
        reference_fit_mode="fit",
        reference_resize_method="auto",
        image_advanced_settings=bundle1,
    )
    bundle2 = out2[0]
    assert "subject" in bundle2.role_settings
    assert "scene" in bundle2.role_settings
    assert bundle2.role_settings["scene"].boost == 1.5


def test_role_instructions_formatting():
    # 1. Subject only
    inst_subj = build_role_instructions([ReferenceRole.SUBJECT])
    assert "Image 1 is the subject image." in inst_subj
    assert "Use the subject image for identity" in inst_subj

    # 2. Scene + Subject
    inst_sc_sub = build_role_instructions([ReferenceRole.SCENE, ReferenceRole.SUBJECT])
    assert "Image 1 is the scene image." in inst_sc_sub
    assert "Image 2 is the subject image." in inst_sc_sub
    assert "Use the scene image for composition" in inst_sc_sub

    # 3. Outfit + Subject
    inst_out_sub = build_role_instructions([ReferenceRole.OUTFIT, ReferenceRole.SUBJECT])
    assert "Image 1 is the outfit image." in inst_out_sub
    assert "Image 2 is the subject image." in inst_out_sub
    assert "Do not use the wearer of the outfit image" in inst_out_sub

    # 4. Source (Inpaint)
    inst_src = build_role_instructions([ReferenceRole.SOURCE])
    assert "Image 1 is the source image." in inst_src
    assert "The source image is the base image being edited." in inst_src


def test_declarative_reference_nodes_anchor_controls_contract():
    """Assert Subject, Scene, and Outfit nodes strictly expose only approved anchor controls."""
    subj_cls = NODE_CLASS_MAPPINGS["CcCKrea2SubjectImage"]
    subj_req = subj_cls.INPUT_TYPES()["required"]

    assert "masked_identity_anchor" in subj_req
    assert "pose_anchor" in subj_req
    assert "outfit_anchor" in subj_req
    assert "attention_boost" in subj_req
    assert "masked_attention_boost" in subj_req
    assert "masked_region_anchor" not in subj_req

    scene_cls = NODE_CLASS_MAPPINGS["CcCKrea2SceneImage"]
    scene_req = scene_cls.INPUT_TYPES()["required"]

    assert "scene_anchor" in scene_req
    assert "masked_region_anchor" in scene_req
    assert "attention_boost" in scene_req
    assert "masked_attention_boost" in scene_req

    outfit_cls = NODE_CLASS_MAPPINGS["CcCKrea2OutfitImage"]
    outfit_req = outfit_cls.INPUT_TYPES()["required"]

    assert "outfit_anchor" in outfit_req
    assert "attention_boost" in outfit_req
    assert "masked_attention_boost" in outfit_req
    assert "masked_identity_anchor" not in outfit_req
    assert "masked_region_anchor" not in outfit_req
