"""Central orchestration engine for Krea 2 reference-guided editing nodes."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import torch

from .constants import (
    LOGGER_PREFIX,
    ReferenceRole,
    ROLE_ORDER_SUBJECT,
    ROLE_ORDER_SUBJECT_OUTFIT,
    ROLE_ORDER_SUBJECT_SCENE,
    ROLE_ORDER_SUBJECT_SCENE_OUTFIT,
    ROLE_ORDER_INPAINT,
    ROLE_ORDER_INPAINT_SUBJECT_OUTFIT,
    ROLE_ORDER_INPAINT_SUBJECT_SCENE,
    EXPERIMENTAL_OUTFIT_WARNING,
)
from .validation import validate_krea2_model, align_dimensions
from .references import ReferenceConfig, prepare_reference
from .patch import patch_krea2_model
from .conditioning import encode_krea2_conditioning
from .latents import create_empty_latent, create_image_latent, create_inpaint_latent
from .geometry import apply_sampling_transform


@dataclass
class NodeExecutionRequest:
    node_name: str
    model: Any
    clip: Any
    prompt: str
    negative_prompt: str = ""
    vae: Optional[Any] = None

    # Geometry & Latent outputs
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    sampling_resize_mode: str = "fit"      # Controls LATENT returned to KSampler (fit, crop, stretch)
    reference_fit_mode: str = "fit"        # Controls VAE reference geometry (fit, crop)
    latent_source: str = "empty"

    # Images
    subject_image: Optional[torch.Tensor] = None
    outfit_image: Optional[torch.Tensor] = None
    scene_image: Optional[torch.Tensor] = None
    source_image: Optional[torch.Tensor] = None

    # Attention masks
    subject_attention_mask: Optional[torch.Tensor] = None
    outfit_attention_mask: Optional[torch.Tensor] = None
    scene_attention_mask: Optional[torch.Tensor] = None
    source_attention_mask: Optional[torch.Tensor] = None

    # Dials & Controls
    subject_boost: float = 2.5
    outfit_boost: float = 1.0
    scene_boost: float = 1.0
    source_boost: float = 1.0

    subject_mask_invert: bool = False
    outfit_mask_invert: bool = False
    scene_mask_invert: bool = False
    source_mask_invert: bool = False

    attention_mask_mode: str = "hard"

    # Grounding controls
    subject_grounding_preset: str = "balanced"
    subject_grounding_resize_mode: str = "normalize"
    subject_grounding_px: int = 768
    subject_grounding_min_px: int = 512
    subject_grounding_max_px: int = 1024

    outfit_grounding_preset: str = "balanced"
    outfit_grounding_resize_mode: str = "normalize"
    outfit_grounding_px: int = 768
    outfit_grounding_min_px: int = 512
    outfit_grounding_max_px: int = 1024

    scene_grounding_preset: str = "balanced"
    scene_grounding_resize_mode: str = "normalize"
    scene_grounding_px: int = 768
    scene_grounding_min_px: int = 512
    scene_grounding_max_px: int = 1024

    source_grounding_preset: str = "balanced"
    source_grounding_resize_mode: str = "normalize"
    source_grounding_px: int = 768
    source_grounding_min_px: int = 512
    source_grounding_max_px: int = 1024

    # Inpainting specific controls
    inpaint_mask: Optional[torch.Tensor] = None
    inpaint_mask_invert: bool = False
    inpaint_mask_grow: int = 0
    inpaint_mask_blur: int = 0

    role_order: List[ReferenceRole] = field(default_factory=lambda: ROLE_ORDER_SUBJECT)


class Krea2EditEngine:
    """Central engine executing the 7 reference-guided editing nodes."""

    @staticmethod
    def execute(request: NodeExecutionRequest) -> Tuple[Any, List[Any], List[Any], Dict[str, Any]]:
        # 1. Validate incoming model
        validate_krea2_model(request.model, request.node_name)

        # 2. Check for experimental workflow logging
        has_outfit = any(r == ReferenceRole.OUTFIT for r in request.role_order)
        has_three_refs = len(request.role_order) >= 3
        if has_outfit or has_three_refs:
            print(f"{LOGGER_PREFIX} Notice in node '{request.node_name}': {EXPERIMENTAL_OUTFIT_WARNING}")

        # 3. Align width and height
        target_w, target_h = align_dimensions(request.width, request.height, Multiple=16)

        # 4. Build reference configurations in strict role order
        ref_configs: List[ReferenceConfig] = []

        for role in request.role_order:
            if role == ReferenceRole.SUBJECT and request.subject_image is not None:
                ref_configs.append(ReferenceConfig(
                    role=ReferenceRole.SUBJECT,
                    image=request.subject_image,
                    attention_mask=request.subject_attention_mask,
                    boost=request.subject_boost,
                    mask_invert=request.subject_mask_invert,
                    grounding_preset=request.subject_grounding_preset,
                    grounding_resize_mode=request.subject_grounding_resize_mode,
                    grounding_px=request.subject_grounding_px,
                    grounding_min_px=request.subject_grounding_min_px,
                    grounding_max_px=request.subject_grounding_max_px,
                    reference_fit_mode=request.reference_fit_mode
                ))
            elif role == ReferenceRole.OUTFIT and request.outfit_image is not None:
                ref_configs.append(ReferenceConfig(
                    role=ReferenceRole.OUTFIT,
                    image=request.outfit_image,
                    attention_mask=request.outfit_attention_mask,
                    boost=request.outfit_boost,
                    mask_invert=request.outfit_mask_invert,
                    grounding_preset=request.outfit_grounding_preset,
                    grounding_resize_mode=request.outfit_grounding_resize_mode,
                    grounding_px=request.outfit_grounding_px,
                    grounding_min_px=request.outfit_grounding_min_px,
                    grounding_max_px=request.outfit_grounding_max_px,
                    reference_fit_mode=request.reference_fit_mode
                ))
            elif role == ReferenceRole.SCENE and request.scene_image is not None:
                ref_configs.append(ReferenceConfig(
                    role=ReferenceRole.SCENE,
                    image=request.scene_image,
                    attention_mask=request.scene_attention_mask,
                    boost=request.scene_boost,
                    mask_invert=request.scene_mask_invert,
                    grounding_preset=request.scene_grounding_preset,
                    grounding_resize_mode=request.scene_grounding_resize_mode,
                    grounding_px=request.scene_grounding_px,
                    grounding_min_px=request.scene_grounding_min_px,
                    grounding_max_px=request.scene_grounding_max_px,
                    reference_fit_mode=request.reference_fit_mode
                ))
            elif role == ReferenceRole.SOURCE and request.source_image is not None:
                ref_configs.append(ReferenceConfig(
                    role=ReferenceRole.SOURCE,
                    image=request.source_image,
                    attention_mask=request.source_attention_mask,
                    boost=request.source_boost,
                    mask_invert=request.source_mask_invert,
                    grounding_preset=request.source_grounding_preset,
                    grounding_resize_mode=request.source_grounding_resize_mode,
                    grounding_px=request.source_grounding_px,
                    grounding_min_px=request.source_grounding_min_px,
                    grounding_max_px=request.source_grounding_max_px,
                    reference_fit_mode=request.reference_fit_mode
                ))

        # 5. Prepare references (dual path: grounding + VAE reference tokens) ONCE
        prepared_refs = [
            prepare_reference(
                config=cfg,
                vae=request.vae,
                target_h=target_h,
                target_w=target_w,
                reference_fit_mode=request.reference_fit_mode,
                attention_mask_mode=request.attention_mask_mode
            )
            for cfg in ref_configs
        ]

        # 6. Patch model (per-instance ModelPatcher wrapper)
        patched_model = patch_krea2_model(request.model)

        # 7. Encode Qwen3-VL positive and negative conditioning
        positive, negative = encode_krea2_conditioning(
            clip=request.clip,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            prepared_references=prepared_refs
        )

        # 8. Select & build single output LATENT (uses sampling_resize_mode: fit, crop, stretch)
        output_latent: Dict[str, Any]

        if "inpaint" in request.node_name.lower():
            base_image: Optional[torch.Tensor] = None
            if request.source_image is not None:
                base_image = request.source_image
            elif request.subject_image is not None:
                base_image = request.subject_image
            elif request.scene_image is not None:
                base_image = request.scene_image

            if base_image is None or request.vae is None:
                output_latent = create_empty_latent(target_w, target_h, request.batch_size)
            else:
                from .masks import process_inpaint_mask
                # Apply sampling transform to base image and inpaint mask
                trans_base_img, trans_inpaint_mask = apply_sampling_transform(
                    image=base_image,
                    target_h=target_h,
                    target_w=target_w,
                    mode=request.sampling_resize_mode,
                    mask=request.inpaint_mask,
                    mask_interpolation="nearest"
                )

                processed_inpaint_mask = None
                if trans_inpaint_mask is not None:
                    processed_inpaint_mask = process_inpaint_mask(
                        mask=trans_inpaint_mask,
                        invert=request.inpaint_mask_invert,
                        grow=request.inpaint_mask_grow,
                        blur=request.inpaint_mask_blur
                    )
                output_latent = create_inpaint_latent(
                    vae=request.vae,
                    image=trans_base_img,
                    noise_mask=processed_inpaint_mask
                )
        else:
            sel_image: Optional[torch.Tensor] = None
            if request.latent_source == "subject" and request.subject_image is not None:
                sel_image = request.subject_image
            elif request.latent_source == "scene" and request.scene_image is not None:
                sel_image = request.scene_image

            if sel_image is not None and request.vae is not None:
                trans_sel_img, _ = apply_sampling_transform(
                    image=sel_image,
                    target_h=target_h,
                    target_w=target_w,
                    mode=request.sampling_resize_mode
                )
                output_latent = create_image_latent(request.vae, trans_sel_img)
            else:
                output_latent = create_empty_latent(target_w, target_h, request.batch_size)

        return patched_model, positive, negative, output_latent
