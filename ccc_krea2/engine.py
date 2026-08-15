"""Execution orchestrator engine for CcC Krea2 custom node suite."""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import torch

from .constants import ReferenceRole, EXPERIMENTAL_OUTFIT_WARNING, DEFAULT_SYSTEM_PROMPT
from .validation import validate_krea2_model
from .settings import (
    ImageAdvancedSettingsBundle,
    EditAdvancedSettings,
    resolve_krea2_settings,
)
from .resolution import resolve_output_resolution
from .references import ReferenceConfig, prepare_reference, PreparedReference
from .latents import generate_krea2_latent
from .conditioning import encode_krea2_conditioning, build_role_instructions
from .patch import patch_krea2_model
from .prompt_augmentation import PromptAugmentation, apply_prompt_augmentation

logger = logging.getLogger("CcCKrea2")


@dataclass
class NodeExecutionRequest:
    node_name: str
    model: Any
    clip: Any
    prompt: str
    vae: Any
    negative_prompt: str = ""
    preset: str = "balanced"
    output_resolution: str = "subject"
    megapixels: float = 1.0
    prompt_augmentation: Optional[PromptAugmentation] = None
    image_advanced_settings: Optional[ImageAdvancedSettingsBundle] = None
    edit_advanced_settings: Optional[EditAdvancedSettings] = None
    subject_image: Optional[torch.Tensor] = None
    scene_image: Optional[torch.Tensor] = None
    outfit_image: Optional[torch.Tensor] = None
    source_image: Optional[torch.Tensor] = None
    subject_attention_mask: Optional[torch.Tensor] = None
    scene_attention_mask: Optional[torch.Tensor] = None
    outfit_attention_mask: Optional[torch.Tensor] = None
    source_attention_mask: Optional[torch.Tensor] = None
    inpaint_mask: Optional[torch.Tensor] = None
    latent_source: str = "empty"
    inpaint_base_role: Optional[ReferenceRole] = None
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

        # 4. Resolve settings via precedence hierarchy
        active_roles = [r.value for r in req.role_order or []]
        settings = resolve_krea2_settings(
            preset_name=req.preset,
            edit_settings=req.edit_advanced_settings,
            image_settings=req.image_advanced_settings,
            active_roles=active_roles,
        )

        # 5. Resolve output width and height
        default_auto_role = cls._get_default_auto_role(req)
        node_images = {
            "subject": req.subject_image,
            "scene": req.scene_image,
            "outfit": req.outfit_image,
            "source": req.source_image,
        }

        out_w, out_h = resolve_output_resolution(
            output_resolution=req.output_resolution,
            megapixels=req.megapixels,
            node_images=node_images,
            default_auto_role=default_auto_role,
            custom_aspect_source=settings.custom_aspect_source,
            role_resolution_limit_mode=settings.role_resolution_limit_mode,
            role_resolution_max_megapixels=settings.role_resolution_max_megapixels,
            node_name=req.node_name,
        )

        # 6. Base image resolution for sampling/inpainting
        base_image = cls._resolve_base_image(req)

        # 7. Build reference configs using resolved role settings
        ref_configs = cls._build_reference_configs(req, settings)

        # 8. Dual-path reference preparation (Grounding + VAE latents)
        prepared_refs: List[PreparedReference] = []
        grounding_images: List[torch.Tensor] = []

        for cfg in ref_configs:
            prep = prepare_reference(
                config=cfg,
                vae=req.vae,
                model=req.model,
                target_h=out_h,
                target_w=out_w,
                reference_fit_mode=cfg.reference_fit_mode,
                attention_mask_mode=settings.attention_mask_mode,
            )
            if prep is not None:
                prepared_refs.append(prep)
                if prep.grounding_image is not None:
                    grounding_images.append(prep.grounding_image)

        # 9. Canonical model patch call: patch_krea2_model(model, prepared_refs)
        patched_model = patch_krea2_model(req.model, prepared_refs)

        # 10. System prompt role instructions & Qwen3-VL conditioning encoding
        role_instructions = build_role_instructions(req.role_order or [])
        if role_instructions:
            sys_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{role_instructions}"
        else:
            sys_prompt = DEFAULT_SYSTEM_PROMPT

        if settings.prompt_instructions_mode == "append" and settings.prompt_instructions:
            sys_prompt = f"{sys_prompt}\n\n{settings.prompt_instructions}"

        # Merge effective positive and negative prompts with prompt augmentation
        eff_prompt, eff_neg_prompt = apply_prompt_augmentation(
            positive_prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            augmentation=req.prompt_augmentation,
        )

        positive, negative = encode_krea2_conditioning(
            clip=req.clip,
            prompt=eff_prompt,
            negative_prompt=eff_neg_prompt,
            grounding_images=grounding_images,
            system_prompt=sys_prompt,
        )

        # 11. Generate target LATENT dictionary
        latent_dict = generate_krea2_latent(
            model=req.model,
            vae=req.vae,
            width=out_w,
            height=out_h,
            batch_size=settings.batch_size,
            latent_source=req.latent_source if "Inpaint" not in req.node_name else "image",
            base_image=base_image,
            inpaint_mask=req.inpaint_mask,
            inpaint_mask_invert=settings.inpaint_mask_invert,
            inpaint_mask_grow=settings.inpaint_mask_grow,
            inpaint_mask_blur=settings.inpaint_mask_blur,
            sampling_resize_mode=settings.sampling_resize_mode,
            sampling_resize_method=settings.sampling_resize_method,
        )

        return patched_model, positive, negative, latent_dict

    @classmethod
    def _get_default_auto_role(cls, req: NodeExecutionRequest) -> str:
        """Determine default auto role for custom aspect ratio calculation."""
        if req.inpaint_base_role == ReferenceRole.SOURCE:
            return "source"
        if ReferenceRole.SCENE in (req.role_order or []):
            return "scene"
        if ReferenceRole.SUBJECT in (req.role_order or []):
            return "subject"
        if ReferenceRole.SOURCE in (req.role_order or []):
            return "source"
        return "subject"

    @classmethod
    def _resolve_base_image(cls, req: NodeExecutionRequest) -> Optional[torch.Tensor]:
        """Explicitly select base image per node contract using inpaint_base_role or latent_source."""
        if req.inpaint_base_role == ReferenceRole.SOURCE:
            return req.source_image
        elif req.inpaint_base_role == ReferenceRole.SUBJECT:
            return req.subject_image
        elif req.inpaint_base_role == ReferenceRole.SCENE:
            return req.scene_image

        if req.latent_source == "subject":
            return req.subject_image
        elif req.latent_source == "scene":
            return req.scene_image
        elif req.latent_source == "source":
            return req.source_image
        return None

    @classmethod
    def _build_reference_configs(cls, req: NodeExecutionRequest, settings: Any) -> List[ReferenceConfig]:
        configs = []
        for role in req.role_order or []:
            r_str = role.value if hasattr(role, "value") else str(role)
            r_set = settings.roles.get(r_str)

            if role == ReferenceRole.SUBJECT and req.subject_image is not None:
                configs.append(
                    ReferenceConfig(
                        role=role,
                        image=req.subject_image,
                        attention_mask=req.subject_attention_mask,
                        boost=r_set.boost,
                        mask_invert=r_set.mask_invert,
                        grounding_preset="custom",
                        grounding_resize_mode=r_set.grounding_resize_mode,
                        grounding_px=r_set.grounding_px,
                        grounding_min_px=r_set.grounding_min_px,
                        grounding_max_px=r_set.grounding_max_px,
                        grounding_resize_method=r_set.grounding_resize_method,
                        reference_fit_mode=r_set.reference_fit_mode,
                        reference_resize_method=r_set.reference_resize_method,
                    )
                )
            elif role == ReferenceRole.SCENE and req.scene_image is not None:
                configs.append(
                    ReferenceConfig(
                        role=role,
                        image=req.scene_image,
                        attention_mask=req.scene_attention_mask,
                        boost=r_set.boost,
                        mask_invert=r_set.mask_invert,
                        grounding_preset="custom",
                        grounding_resize_mode=r_set.grounding_resize_mode,
                        grounding_px=r_set.grounding_px,
                        grounding_min_px=r_set.grounding_min_px,
                        grounding_max_px=r_set.grounding_max_px,
                        grounding_resize_method=r_set.grounding_resize_method,
                        reference_fit_mode=r_set.reference_fit_mode,
                        reference_resize_method=r_set.reference_resize_method,
                    )
                )
            elif role == ReferenceRole.OUTFIT and req.outfit_image is not None:
                configs.append(
                    ReferenceConfig(
                        role=role,
                        image=req.outfit_image,
                        attention_mask=req.outfit_attention_mask,
                        boost=r_set.boost,
                        mask_invert=r_set.mask_invert,
                        grounding_preset="custom",
                        grounding_resize_mode=r_set.grounding_resize_mode,
                        grounding_px=r_set.grounding_px,
                        grounding_min_px=r_set.grounding_min_px,
                        grounding_max_px=r_set.grounding_max_px,
                        grounding_resize_method=r_set.grounding_resize_method,
                        reference_fit_mode=r_set.reference_fit_mode,
                        reference_resize_method=r_set.reference_resize_method,
                    )
                )
            elif role == ReferenceRole.SOURCE and req.source_image is not None:
                configs.append(
                    ReferenceConfig(
                        role=role,
                        image=req.source_image,
                        attention_mask=req.source_attention_mask,
                        boost=r_set.boost,
                        mask_invert=r_set.mask_invert,
                        grounding_preset="custom",
                        grounding_resize_mode=r_set.grounding_resize_mode,
                        grounding_px=r_set.grounding_px,
                        grounding_min_px=r_set.grounding_min_px,
                        grounding_max_px=r_set.grounding_max_px,
                        grounding_resize_method=r_set.grounding_resize_method,
                        reference_fit_mode=r_set.reference_fit_mode,
                        reference_resize_method=r_set.reference_resize_method,
                    )
                )
        return configs
