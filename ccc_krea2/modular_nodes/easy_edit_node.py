"""CcC Krea2 Easy Edit and Easy Edit Ostris custom nodes."""

from typing import Tuple, Optional, Dict, Any
import torch

from ..easy_routing import resolve_easy_sources, route_easy_preset
from ..vision_prep import prepare_image_for_qwen
from ..reference_specs import ReferenceSpec, ReferenceChain, PreparedVisionImage
from ..target_latent import build_target_latent
from ..reference_slots import resolve_reference_slots_and_aliases
from ..conditioning import encode_krea2_qwen_context, encode_prompt_with_qwen, build_annotated_user_prompt
from ..references import PreparedReference
from ..patch import patch_krea2_model
from ..ostris_backend import patch_ostris_model, preprocess_ostris_vision_image
from ..constants import NODE_CATEGORY


class CcCKrea2EasyEdit:
    """Opinionated Easy Edit node for Krea2/Identity Edit workflows."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "preset": (
                    ["balanced", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer", "style_transfer"],
                    {"default": "balanced"}
                ),
                "outfit_source": (["outfit image", "scene image", "style image"], {"default": "outfit image"}),
                "style_source": (["style image", "scene image", "subject image"], {"default": "style image"}),
                "apply_krea2_edit_patch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject": ("IMAGE",),
                "scene": ("IMAGE",),
                "outfit": ("IMAGE",),
                "style": ("IMAGE",),
            }
        }

    def process(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        positive_prompt: str,
        preset: str = "balanced",
        outfit_source: str = "outfit image",
        style_source: str = "style image",
        apply_krea2_edit_patch: bool = True,
        negative_prompt: str = "",
        subject: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        outfit: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
    ):
        return _execute_easy_edit(
            model=model,
            clip=clip,
            vae=vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            preset=preset,
            outfit_source=outfit_source,
            style_source=style_source,
            backend_method="krea2_edit",
            apply_patch=apply_krea2_edit_patch,
            ostris_kv_cache=False,
            subject=subject,
            scene=scene,
            outfit=outfit,
            style=style,
        )


class CcCKrea2EasyEditOstris:
    """Opinionated Easy Edit node for Ostris Edit workflows."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "preset": (
                    ["balanced", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer", "style_transfer"],
                    {"default": "balanced"}
                ),
                "outfit_source": (["outfit image", "scene image", "style image"], {"default": "outfit image"}),
                "style_source": (["style image", "scene image", "subject image"], {"default": "style image"}),
                "apply_ostris_edit_patch": ("BOOLEAN", {"default": True}),
                "ostris_kv_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "subject": ("IMAGE",),
                "scene": ("IMAGE",),
                "outfit": ("IMAGE",),
                "style": ("IMAGE",),
            }
        }

    def process(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        positive_prompt: str,
        preset: str = "balanced",
        outfit_source: str = "outfit image",
        style_source: str = "style image",
        apply_ostris_edit_patch: bool = True,
        ostris_kv_cache: bool = False,
        negative_prompt: str = "",
        subject: Optional[torch.Tensor] = None,
        scene: Optional[torch.Tensor] = None,
        outfit: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
    ):
        return _execute_easy_edit(
            model=model,
            clip=clip,
            vae=vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            preset=preset,
            outfit_source=outfit_source,
            style_source=style_source,
            backend_method="ostris_edit",
            apply_patch=apply_ostris_edit_patch,
            ostris_kv_cache=ostris_kv_cache,
            subject=subject,
            scene=scene,
            outfit=outfit,
            style=style,
        )


def _execute_easy_edit(
    model: Any,
    clip: Any,
    vae: Any,
    positive_prompt: str,
    negative_prompt: str,
    preset: str,
    outfit_source: str,
    style_source: str,
    backend_method: str,
    apply_patch: bool,
    ostris_kv_cache: bool,
    subject: Optional[torch.Tensor],
    scene: Optional[torch.Tensor],
    outfit: Optional[torch.Tensor],
    style: Optional[torch.Tensor],
) -> Tuple[Any, Any, Any, Dict[str, Any], str]:
    """Internal orchestrator for Easy Edit and Easy Edit Ostris nodes."""

    # Phase 1: Source Resolution
    resolved_sources = resolve_easy_sources(
        subject=subject,
        scene=scene,
        outfit=outfit,
        style=style,
        outfit_source=outfit_source,
        style_source=style_source,
    )

    # Phase 2: Preset Routing
    route = route_easy_preset(resolved_sources, preset=preset)

    # Cache for prepared vision images
    prep_cache: Dict[int, PreparedVisionImage] = {}

    def get_prepared(img_tensor: torch.Tensor, is_ostris: bool = False) -> PreparedVisionImage:
        key = id(img_tensor)
        if key not in prep_cache:
            if is_ostris:
                ostris_img = preprocess_ostris_vision_image(img_tensor)
                prep_cache[key] = prepare_image_for_qwen(clip, ostris_img)
            else:
                prep_cache[key] = prepare_image_for_qwen(clip, img_tensor)
        return prep_cache[key]

    is_ostris = (backend_method == "ostris_edit")

    # Build reference specs chain
    chain = ReferenceChain()

    for item_img, boost, alias_role in route.edit_references:
        prep = get_prepared(item_img, is_ostris=is_ostris)
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias_role,
            attention_boost=boost,
            _legacy_role=alias_role,
        )
        chain = chain.append(spec)

    # Check effective_style
    effective_style = resolved_sources.effective_style
    style_active = False
    if effective_style is not None and preset == "style_transfer":
        prep_style = get_prepared(effective_style, is_ostris=is_ostris)
        style_spec = ReferenceSpec(
            reference_path="style",
            prepared_image=prep_style,
            alias="style",
            style_fidelity=1.0,
            _legacy_role="style",
        )
        chain = chain.append(style_spec)
        style_active = True

    # Resolve slots and physical image layout
    resolved_map, warnings = resolve_reference_slots_and_aliases(chain)

    # Build target latent
    target_prep = get_prepared(route.target_source, is_ostris=is_ostris) if route.target_source is not None else None

    latent_dict, latent_info = build_target_latent(
        vae=vae,
        target_content=route.target_content_mode,
        geometry_mode=route.geometry_mode,
        target_image=target_prep,
        batch_size=1,
    )

    # Collect physical Qwen images
    physical_images = []
    physical_image_map = []

    for item in resolved_map:
        spec = item["spec"]
        ref_path = getattr(spec, "reference_path", spec.role.lower())
        if ref_path == "style":
            # Moodboard grid
            style_proc = getattr(spec, "style_processing", "2x2")
            mb_grid = spec.prepared_image.vision_image
            # Expand moodboard tiles into physical Qwen images list
            physical_images.append(mb_grid)
            physical_image_map.append({
                "role": "style",
                "spec": spec,
                "logical_reference_id": item["logical_reference_id"],
                "logical_vision_slot": item["logical_vision_slot"],
                "physical_qwen_image_index": len(physical_images),
            })
        else:
            physical_images.append(spec.prepared_image.vision_image)
            physical_image_map.append({
                "role": "edit",
                "spec": spec,
                "logical_reference_id": item["logical_reference_id"],
                "logical_vision_slot": item["logical_vision_slot"],
                "physical_qwen_image_index": len(physical_images),
            })

    # Build Qwen user prompt (without prepended system role directives)
    annotated_prompt = build_annotated_user_prompt(
        resolved_references=resolved_map,
        user_prompt=positive_prompt,
    )

    # Tokenize and encode positive conditioning
    encoded_pos = encode_krea2_qwen_context(
        clip=clip,
        prompt=annotated_prompt,
        physical_images=physical_images,
        physical_image_map=physical_image_map,
        is_positive=True,
    )

    # Tokenize and encode negative conditioning (excluding style references)
    edit_physical_images = [
        item["spec"].prepared_image.vision_image
        for item in resolved_map
        if getattr(item["spec"], "reference_path", item["spec"].role.lower()) != "style"
    ]
    encoded_neg_cond = encode_prompt_with_qwen(clip, negative_prompt or "", images=edit_physical_images)

    # Apply model patch
    if apply_patch:
        prepared_refs: List[PreparedReference] = []
        for item in resolved_map:
            spec = item["spec"]
            ref_path = getattr(spec, "reference_path", spec.role.lower())
            if ref_path == "edit":
                vae_lat = None
                if vae is not None and spec.prepared_image is not None:
                    raw = vae.encode(spec.prepared_image.original_image)
                    vae_lat = raw["samples"] if isinstance(raw, dict) else raw

                prepared_refs.append(PreparedReference(
                    role=spec.role,
                    boost=spec.attention_boost,
                    masked_boost=spec.masked_attention_boost,
                    spatial_attention_mask=spec.attention_mask,
                    mask_mode="hard",
                    vae_latent=vae_lat,
                ))

        if backend_method == "ostris_edit":
            patched_model = patch_ostris_model(model, prepared_refs, ostris_kv_cache=ostris_kv_cache)
        else:
            patched_model = patch_krea2_model(model, prepared_refs)
    else:
        patched_model = model

    # Format edit_info report
    lines = [
        f"Backend Method: {backend_method}",
        f"Preset: {preset}",
        f"Outfit Source: {outfit_source}",
        f"Style Source: {style_source}",
        f"Model Patch Applied: {'yes' if apply_patch else 'no (skipped)'}",
        f"Ostris KV Cache: {'yes' if ostris_kv_cache else 'no'}",
        f"Resolved Edit References: {len(route.edit_references)}",
        f"Style Path Active: {'yes' if style_active else 'no'}",
        f"Warnings: {'; '.join(resolved_sources.warnings) if resolved_sources.warnings else 'none'}",
        "--- Target Latent Info ---",
        latent_info,
    ]
    edit_info = "\n".join(lines)

    return (patched_model, encoded_pos.conditioning, encoded_neg_cond, latent_dict, edit_info)
