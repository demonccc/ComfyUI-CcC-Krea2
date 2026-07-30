"""Execution orchestrator engine for CcC Krea2 custom node suite."""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import torch

from .constants import ReferenceRole, EXPERIMENTAL_OUTFIT_WARNING
from .validation import validate_krea2_model
from .references import ReferenceConfig, prepare_reference, PreparedReference
from .latents import generate_krea2_latent
from .conditioning import encode_krea2_conditioning
from .patch import patch_krea2_model

logger = logging.getLogger("CcCKrea2")


@dataclass
class NodeExecutionRequest:
    node_name: str
    model: Any
    clip: Any
    prompt: str
    vae: Any
    negative_prompt: str = ""
    subject_image: Optional[torch.Tensor] = None
    scene_image: Optional[torch.Tensor] = None
    outfit_image: Optional[torch.Tensor] = None
    source_image: Optional[torch.Tensor] = None
    subject_attention_mask: Optional[torch.Tensor] = None
    scene_attention_mask: Optional[torch.Tensor] = None
    outfit_attention_mask: Optional[torch.Tensor] = None
    source_attention_mask: Optional[torch.Tensor] = None
    inpaint_mask: Optional[torch.Tensor] = None
    subject_boost: float = 2.5
    scene_boost: float = 1.0
    outfit_boost: float = 1.0
    source_boost: float = 1.0
    subject_mask_invert: bool = False
    scene_mask_invert: bool = False
    outfit_mask_invert: bool = False
    source_mask_invert: bool = False
    inpaint_mask_invert: bool = False
    inpaint_mask_grow: int = 0
    inpaint_mask_blur: int = 0
    attention_mask_mode: str = "hard"
    subject_grounding_preset: str = "balanced"
    subject_grounding_resize_mode: str = "normalize"
    subject_grounding_px: int = 768
    subject_grounding_min_px: int = 128
    subject_grounding_max_px: int = 4096
    scene_grounding_preset: str = "balanced"
    scene_grounding_resize_mode: str = "normalize"
    scene_grounding_px: int = 768
    scene_grounding_min_px: int = 128
    scene_grounding_max_px: int = 4096
    outfit_grounding_preset: str = "balanced"
    outfit_grounding_resize_mode: str = "normalize"
    outfit_grounding_px: int = 768
    outfit_grounding_min_px: int = 128
    outfit_grounding_max_px: int = 4096
    source_grounding_preset: str = "balanced"
    source_grounding_resize_mode: str = "normalize"
    source_grounding_px: int = 768
    source_grounding_min_px: int = 128
    source_grounding_max_px: int = 4096
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    sampling_resize_mode: str = "fit"
    reference_fit_mode: str = "fit"
    latent_source: str = "empty"
    role_order: List[ReferenceRole] = None


class Krea2EditEngine:
    """Orchestrates execution of CcC Krea2 custom node operations."""

    @classmethod
    def execute(cls, req: NodeExecutionRequest) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        # 1. Model compatibility validation
        validate_krea2_model(req.model, req.node_name)

        # 2. Strict VAE requirement across all 7 nodes
        if req.vae is None:
            raise ValueError(f"[{req.node_name}] VAE input is required for reference processing and latent generation.")

        # 3. Log experimental workflow warning if outfit or 3-ref is involved
        if req.outfit_image is not None or len(req.role_order or []) > 2:
            logger.warning(EXPERIMENTAL_OUTFIT_WARNING)

        # 4. Explicit inpainting & sampling base image resolution
        base_image = cls._resolve_base_image(req)

        # 5. Build reference configs in strict role order
        ref_configs = cls._build_reference_configs(req)

        # 6. Dual-path reference preparation (Grounding + VAE latents with process_latent_in)
        prepared_refs: List[PreparedReference] = []
        grounding_images: List[torch.Tensor] = []

        for cfg in ref_configs:
            prep = prepare_reference(
                config=cfg,
                vae=req.vae,
                model=req.model,
                target_h=req.height,
                target_w=req.width,
                reference_fit_mode=req.reference_fit_mode,
                attention_mask_mode=req.attention_mask_mode
            )
            if prep is not None:
                prepared_refs.append(prep)
                if prep.grounding_image is not None:
                    grounding_images.append(prep.grounding_image)

        # 7. ModelPatcher DiT forwarding patch (closure capture)
        patched_model = patch_krea2_model(
            model=req.model,
            prepared_refs=prepared_refs,
            target_h=req.height,
            target_w=req.width
        )

        # 8. Encode Qwen3-VL conditionings
        positive, negative = encode_krea2_conditioning(
            clip=req.clip,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            grounding_images=grounding_images
        )

        # 9. Generate model-driven KSampler target LATENT dictionary with batch_size honor
        latent_dict = generate_krea2_latent(
            model=req.model,
            vae=req.vae,
            width=req.width,
            height=req.height,
            batch_size=req.batch_size,
            latent_source=req.latent_source if "Inpaint" not in req.node_name else "image",
            base_image=base_image,
            inpaint_mask=req.inpaint_mask,
            inpaint_mask_invert=req.inpaint_mask_invert,
            inpaint_mask_grow=req.inpaint_mask_grow,
            inpaint_mask_blur=req.inpaint_mask_blur,
            sampling_resize_mode=req.sampling_resize_mode
        )

        return patched_model, positive, negative, latent_dict

    @classmethod
    def _resolve_base_image(cls, req: NodeExecutionRequest) -> Optional[torch.Tensor]:
        """Explicitly select base image per node contract without ambiguity."""
        if req.node_name == "CcC Krea2 - Inpaint":
            return req.source_image
        elif req.node_name == "CcC Krea2 - Inpaint Subject + Outfit":
            return req.subject_image
        elif req.node_name == "CcC Krea2 - Inpaint Subject + Scene":
            return req.scene_image

        # General editing nodes based on user's latent_source selection
        if req.latent_source == "subject":
            return req.subject_image
        elif req.latent_source == "scene":
            return req.scene_image
        elif req.latent_source == "source":
            return req.source_image
        return None

    @classmethod
    def _build_reference_configs(cls, req: NodeExecutionRequest) -> List[ReferenceConfig]:
        configs = []
        for role in req.role_order or []:
            if role == ReferenceRole.SUBJECT and req.subject_image is not None:
                configs.append(ReferenceConfig(
                    role=role,
                    image=req.subject_image,
                    attention_mask=req.subject_attention_mask,
                    boost=req.subject_boost,
                    mask_invert=req.subject_mask_invert,
                    grounding_preset=req.subject_grounding_preset,
                    grounding_resize_mode=req.subject_grounding_resize_mode,
                    grounding_px=req.subject_grounding_px,
                    grounding_min_px=req.subject_grounding_min_px,
                    grounding_max_px=req.subject_grounding_max_px
                ))
            elif role == ReferenceRole.SCENE and req.scene_image is not None:
                configs.append(ReferenceConfig(
                    role=role,
                    image=req.scene_image,
                    attention_mask=req.scene_attention_mask,
                    boost=req.scene_boost,
                    mask_invert=req.scene_mask_invert,
                    grounding_preset=req.scene_grounding_preset,
                    grounding_resize_mode=req.scene_grounding_resize_mode,
                    grounding_px=req.scene_grounding_px,
                    grounding_min_px=req.scene_grounding_min_px,
                    grounding_max_px=req.scene_grounding_max_px
                ))
            elif role == ReferenceRole.OUTFIT and req.outfit_image is not None:
                configs.append(ReferenceConfig(
                    role=role,
                    image=req.outfit_image,
                    attention_mask=req.outfit_attention_mask,
                    boost=req.outfit_boost,
                    mask_invert=req.outfit_mask_invert,
                    grounding_preset=req.outfit_grounding_preset,
                    grounding_resize_mode=req.outfit_grounding_resize_mode,
                    grounding_px=req.outfit_grounding_px,
                    grounding_min_px=req.outfit_grounding_min_px,
                    grounding_max_px=req.outfit_grounding_max_px
                ))
            elif role == ReferenceRole.SOURCE and req.source_image is not None:
                configs.append(ReferenceConfig(
                    role=role,
                    image=req.source_image,
                    attention_mask=req.source_attention_mask,
                    boost=req.source_boost,
                    mask_invert=req.source_mask_invert,
                    grounding_preset=req.source_grounding_preset,
                    grounding_resize_mode=req.source_grounding_resize_mode,
                    grounding_px=req.source_grounding_px,
                    grounding_min_px=req.source_grounding_min_px,
                    grounding_max_px=req.source_grounding_max_px
                ))
        return configs
