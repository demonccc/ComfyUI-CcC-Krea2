"""CcC Krea2 Easy Edit and Easy Edit Ostris custom nodes."""

from typing import Tuple, Optional, Dict, Any
import torch

from ..easy_routing import resolve_easy_sources, route_easy_preset, get_easy_instruction_for_role
from ..grounding import prepare_easy_krea_vision_image
from ..vision_prep import prepare_image_for_qwen
from ..reference_specs import ReferenceSpec, StyleReferenceSpec, ReferenceChain
from ..target_latent import build_target_latent
from ..edit_engine import run_krea2_edit_orchestrator
from ..ostris_backend import preprocess_ostris_vision_image
from ..constants import NODE_CATEGORY


class CcCKrea2EasyEdit:
    """Opinionated Easy Edit node for Krea2/Identity Edit workflows."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"

    DESCRIPTION = (
        "Opinionated 8-preset Krea2 Edit node. "
        "Automatically routes Subject, Scene, Outfit, and Style sources to canonical target and reference channels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Diffusion MODEL to edit."}),
                "clip": ("CLIP", {"tooltip": "Krea2 Qwen CLIP text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "User prompt describing the desired edit."}),
                "preset": (
                    ["flexible", "balanced", "consistent", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer", "style_transfer"],
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."}
                ),
                "outfit_source": (["outfit image", "scene image", "style image"], {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."}),
                "style_source": (["style image", "scene image", "subject image"], {"default": "style image", "tooltip": "Source image socket to use for style conditioning."}),
                "apply_krea2_edit_patch": ("BOOLEAN", {"default": True, "tooltip": "Merged edit LoRA weights do not necessarily include the compatible runtime reference forward."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt."}),
                "subject": ("IMAGE", {"tooltip": "Subject reference image."}),
                "scene": ("IMAGE", {"tooltip": "Scene reference image."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit reference image."}),
                "style": ("IMAGE", {"tooltip": "Style reference image."}),
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

    DESCRIPTION = (
        "Opinionated 8-preset Ostris Edit node. "
        "Routes Subject, Scene, Outfit, and Style sources to Ostris edit pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Diffusion MODEL to edit."}),
                "clip": ("CLIP", {"tooltip": "Krea2 Qwen CLIP text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "User prompt describing the desired edit."}),
                "preset": (
                    ["flexible", "balanced", "consistent", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer", "style_transfer"],
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."}
                ),
                "outfit_source": (["outfit image", "scene image", "style image"], {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."}),
                "style_source": (["style image", "scene image", "subject image"], {"default": "style image", "tooltip": "Source image socket to use for style conditioning."}),
                "apply_ostris_edit_patch": ("BOOLEAN", {"default": True, "tooltip": "When enabled, CcC explicitly selects index_timestep_zero reference behavior. Disable only if the connected MODEL/runtime already provides compatible Ostris edit behavior."}),
                "ostris_kv_cache": ("BOOLEAN", {"default": False, "tooltip": "Currently unavailable in CcC. Intended only for LoRAs trained with ai-toolkit kv_cache."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt."}),
                "subject": ("IMAGE", {"tooltip": "Subject reference image."}),
                "scene": ("IMAGE", {"tooltip": "Scene reference image."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit reference image."}),
                "style": ("IMAGE", {"tooltip": "Style reference image."}),
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
    """Thin resolver for Easy Edit nodes delegating directly to shared orchestrator."""

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

    is_ostris = (backend_method == "ostris_edit")

    # Phase 3: Construct Generic ReferenceChain
    chain = ReferenceChain()

    for item_img, boost, alias_role, instruction in route.edit_references:
        if is_ostris:
            vlm_img = preprocess_ostris_vision_image(item_img)
        else:
            vlm_img = prepare_easy_krea_vision_image(item_img, preset=preset, role=alias_role)

        prep = prepare_image_for_qwen(image=vlm_img, clip=clip, original_image=item_img)
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias_role,
            vision_instruction=instruction,
            attention_boost=boost,
            appearance_reference=True,
            _legacy_role=alias_role,
        )
        chain = chain.append(spec)

    for item_img, alias_role in route.semantic_only_references:
        if is_ostris:
            vlm_img = preprocess_ostris_vision_image(item_img)
        else:
            vlm_img = prepare_easy_krea_vision_image(item_img, preset=preset, role=alias_role)

        prep = prepare_image_for_qwen(image=vlm_img, clip=clip, original_image=item_img)
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias_role,
            vision_instruction=get_easy_instruction_for_role(alias_role),
            appearance_reference=False,
            _legacy_role=alias_role,
        )
        chain = chain.append(spec)


    if route.style_active and route.style_source is not None:
        style_vlm = prepare_easy_krea_vision_image(route.style_source, preset=preset, role="style")
        prep_style = prepare_image_for_qwen(image=style_vlm, clip=clip, original_image=route.style_source)
        style_spec = StyleReferenceSpec(
            reference_path="style",
            prepared_image=prep_style,
            alias="style",
            style_fidelity=route.style_config.style_fidelity,
            style_processing=route.style_config.style_processing,
            indirect_style_transfer=route.style_config.indirect_style_transfer,
            vision_instruction=route.style_config.vision_instruction,
            _legacy_role="style",
        )
        chain = chain.append(style_spec)

    # Phase 3.5: Build Target Latent
    target_content_prep = None
    if route.target_content_source is not None:
        if is_ostris:
            t_vlm = preprocess_ostris_vision_image(route.target_content_source)
        else:
            t_vlm = prepare_easy_krea_vision_image(route.target_content_source, preset=preset, role="target_content")
        target_content_prep = prepare_image_for_qwen(image=t_vlm, clip=clip, original_image=route.target_content_source)

    geometry_prep = None
    if route.target_geometry_source is not None:
        if route.target_geometry_source is route.target_content_source:
            geometry_prep = target_content_prep
        else:
            if is_ostris:
                g_vlm = preprocess_ostris_vision_image(route.target_geometry_source)
            else:
                g_vlm = prepare_easy_krea_vision_image(route.target_geometry_source, preset=preset, role="target_geometry")
            geometry_prep = prepare_image_for_qwen(image=g_vlm, clip=clip, original_image=route.target_geometry_source)

    latent_dict, latent_info = build_target_latent(
        vae=vae,
        target_content=route.target_content_mode,
        geometry_mode=route.target_geometry_mode,
        target_image=target_content_prep,
        geometry_image=geometry_prep,
        batch_size=1,
        target_alias=route.target_content_role if route.target_content_role else "",
        target_vision_instruction=get_easy_instruction_for_role(route.target_content_role) if route.target_content_role else "",
    )

    # Delegate to canonical shared orchestrator
    patched_model, pos, neg, lat, orchestrator_info = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=latent_dict,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        reference_method=backend_method,
        apply_model_patch=apply_patch,
        ostris_kv_cache=ostris_kv_cache,
    )

    app_refs = [f"{alias}" for _, _, alias, _ in route.edit_references]
    sem_refs = [f"{alias}" for _, alias in route.semantic_only_references]

    easy_header = [
        "=== Easy Edit Routing Report ===",
        f"Preset: {preset}",
        f"Reference Contract: {backend_method}",
        f"Outfit Source Selector: {outfit_source}",
        f"Style Source Selector: {style_source}",
        f"Resolved Subject: {'present' if resolved_sources.subject is not None else 'missing'}",
        f"Resolved Scene: {'present' if resolved_sources.scene is not None else 'missing'}",
        f"Resolved Outfit Source: {outfit_source} ({'present' if resolved_sources.effective_outfit is not None else 'missing'})",
        f"Resolved Style Source: {style_source} ({'present' if resolved_sources.effective_style is not None else 'missing'})",
        f"Target Content Mode: {route.target_content_mode}",
        f"Target Content Source: {'present' if route.target_content_source is not None else 'none'}",
        f"Target Geometry Mode: {route.target_geometry_mode}",
        f"Target Geometry Source: {'present' if route.target_geometry_source is not None else 'none'}",
        f"Appearance Ref 1: {app_refs[0] if len(app_refs) > 0 else 'none'}",
        f"Appearance Ref 2: {app_refs[1] if len(app_refs) > 1 else 'none'}",
        f"Semantic-only Sources: {', '.join(sem_refs) if sem_refs else 'none'}",
        f"Style Active: {'yes' if route.style_active else 'no'}",
    ]

    if backend_method == "krea2_edit":
        easy_header.append(f"CcC Krea2 Model Patch Applied: {'yes' if apply_patch else 'no'}")
    elif backend_method == "ostris_edit":
        easy_header.append(f"Ostris Reference Method Explicitly Applied: {'yes (conditioning metadata)' if apply_patch else 'no (external runtime expected)'}")
    else:
        easy_header.append(f"Model Patch Applied: {'yes' if apply_patch else 'no'}")

    easy_header.extend([
        f"Warnings: {'; '.join(route.warnings) if route.warnings else 'none'}",
        "",
    ])

    combined_info = "\n".join(easy_header) + "\n" + orchestrator_info
    return (patched_model, pos, neg, lat, combined_info)
