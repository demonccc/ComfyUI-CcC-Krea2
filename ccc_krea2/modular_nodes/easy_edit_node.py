"""CcC Krea2 Easy Edit and Easy Edit Ostris custom nodes."""

from typing import Tuple, Optional, Dict, Any
import torch

from ..easy_routing import (
    resolve_easy_sources,
    route_easy_preset,
    get_easy_instruction_for_role,
    resolve_default_positive_prompt,
)
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
        "Opinionated 10-preset Krea2 Edit node. "
        "Automatically routes Subject, Scene, Outfit, and Style sources to canonical target and reference channels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Diffusion MODEL to edit."}),
                "clip": ("CLIP", {"tooltip": "Krea2 Qwen CLIP text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "positive_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True, "tooltip": "User prompt describing the desired edit."},
                ),
                "use_default_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically resolves system default positive prompt for the active configuration.",
                    },
                ),
                "preset": (
                    [
                        "flexible",
                        "balanced",
                        "consistent",
                        "preserve_identity",
                        "max_identity",
                        "subject_transfer",
                        "preserve_scene",
                        "outfit_transfer",
                        "style_transfer",
                        "scene_reinterpretation",
                    ],
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."},
                ),
                "outfit_source": (
                    ["none", "outfit image", "scene image", "style image"],
                    {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."},
                ),
                "style_source": (
                    ["none", "style image", "scene image", "subject image"],
                    {"default": "style image", "tooltip": "Source image socket to use for style conditioning."},
                ),
                "apply_krea2_edit_patch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Merged edit LoRA weights do not necessarily include the compatible runtime reference forward.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt."}),
                "subject": ("IMAGE", {"tooltip": "Subject reference image."}),
                "scene": ("IMAGE", {"tooltip": "Scene reference image."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit reference image."}),
                "style": ("IMAGE", {"tooltip": "Style reference image."}),
            },
        }

    def process(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        positive_prompt: str,
        use_default_prompt: bool = True,
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
            use_default_prompt=use_default_prompt,
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
        "Opinionated 10-preset Ostris Edit node. "
        "Routes Subject, Scene, Outfit, and Style sources to Ostris edit pipeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Diffusion MODEL to edit."}),
                "clip": ("CLIP", {"tooltip": "Krea2 Qwen CLIP text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "positive_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True, "tooltip": "User prompt describing the desired edit."},
                ),
                "use_default_prompt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically resolves system default positive prompt for the active configuration.",
                    },
                ),
                "preset": (
                    [
                        "flexible",
                        "balanced",
                        "consistent",
                        "preserve_identity",
                        "max_identity",
                        "subject_transfer",
                        "preserve_scene",
                        "outfit_transfer",
                        "style_transfer",
                        "scene_reinterpretation",
                    ],
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."},
                ),
                "outfit_source": (
                    ["none", "outfit image", "scene image", "style image"],
                    {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."},
                ),
                "style_source": (
                    ["none", "style image", "scene image", "subject image"],
                    {"default": "style image", "tooltip": "Source image socket to use for style conditioning."},
                ),
                "apply_ostris_edit_patch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, CcC explicitly selects index_timestep_zero reference behavior. Disable only if the connected MODEL/runtime already provides compatible Ostris edit behavior.",
                    },
                ),
                "ostris_kv_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Currently unavailable in CcC. Intended only for LoRAs trained with ai-toolkit kv_cache.",
                    },
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt."}),
                "subject": ("IMAGE", {"tooltip": "Subject reference image."}),
                "scene": ("IMAGE", {"tooltip": "Scene reference image."}),
                "outfit": ("IMAGE", {"tooltip": "Outfit reference image."}),
                "style": ("IMAGE", {"tooltip": "Style reference image."}),
            },
        }

    def process(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        positive_prompt: str,
        use_default_prompt: bool = True,
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
            use_default_prompt=use_default_prompt,
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
    use_default_prompt: bool = True,
    subject: Optional[torch.Tensor] = None,
    scene: Optional[torch.Tensor] = None,
    outfit: Optional[torch.Tensor] = None,
    style: Optional[torch.Tensor] = None,
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
        preset=preset,
    )

    # Phase 1.5: Default Positive Prompt Resolution
    has_default, default_prompt_text, default_prompt_key = resolve_default_positive_prompt(
        preset=preset,
        has_s=(resolved_sources.subject is not None),
        has_sc=(resolved_sources.scene is not None),
        has_o=(resolved_sources.effective_outfit is not None),
        has_st=(resolved_sources.effective_style is not None),
        outfit_source=outfit_source,
        style_source=style_source,
    )

    is_subject_only = (
        resolved_sources.subject is not None
        and resolved_sources.scene is None
        and resolved_sources.effective_outfit is None
        and resolved_sources.effective_style is None
    )

    effective_use_default = use_default_prompt and has_default and not is_subject_only

    if effective_use_default:
        effective_positive_prompt = default_prompt_text
        prompt_source_str = "default"
        resolved_key_str = default_prompt_key
    else:
        effective_positive_prompt = positive_prompt.strip() if positive_prompt else ""
        prompt_source_str = "custom"
        resolved_key_str = "none"

        requires_explicit_custom_prompt = preset in {"subject_transfer", "scene_reinterpretation"} and not has_default
        if requires_explicit_custom_prompt and not effective_positive_prompt:
            if preset == "subject_transfer":
                raise ValueError(
                    "Subject Transfer requires a custom positive prompt when both Subject and Scene are not available."
                )
            elif preset == "scene_reinterpretation":
                raise ValueError(
                    "Scene Reinterpretation requires a custom positive prompt when both Subject and Scene are not available."
                )

        if is_subject_only and not effective_positive_prompt:
            raise ValueError(
                "Subject-only Easy Edit requires a positive prompt because no default editing intent is available."
            )

    # Phase 2: Preset Routing
    route = route_easy_preset(resolved_sources, preset=preset)

    is_ostris = backend_method == "ostris_edit"

    # Common Multi-Reference Geometry Activation
    has_scene = resolved_sources.scene is not None
    has_subject = resolved_sources.subject is not None
    has_outfit = resolved_sources.effective_outfit is not None

    common_geometry_active = (has_scene and has_subject) or (has_subject and has_outfit)
    if common_geometry_active:
        if has_scene:
            common_geometry_anchor_role = "scene"
            common_geometry_anchor_img = resolved_sources.scene
        else:
            common_geometry_anchor_role = "subject"
            common_geometry_anchor_img = resolved_sources.subject
    else:
        common_geometry_anchor_role = "none"
        common_geometry_anchor_img = None

    # Phase 3: Construct Generic ReferenceChain
    chain = ReferenceChain()

    for item_img, boost, alias_role, instruction in route.edit_references:
        if is_ostris:
            vlm_img = preprocess_ostris_vision_image(item_img)
        else:
            vlm_img = prepare_easy_krea_vision_image(item_img, preset=preset, role=alias_role)

        fit_mode = "contain_no_upscale" if common_geometry_active else "auto"

        prep = prepare_image_for_qwen(image=vlm_img, clip=clip, original_image=item_img)
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias_role,
            vision_instruction=instruction,
            attention_boost=boost,
            appearance_reference=True,
            visual_reference_fit=fit_mode,
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

    effective_geometry_source = common_geometry_anchor_img if common_geometry_active else route.target_geometry_source
    effective_geometry_mode = "favor_image" if common_geometry_active else route.target_geometry_mode

    geometry_prep = None
    if effective_geometry_source is not None:
        if effective_geometry_source is route.target_content_source and target_content_prep is not None:
            geometry_prep = target_content_prep
        else:
            if is_ostris:
                g_vlm = preprocess_ostris_vision_image(effective_geometry_source)
            else:
                g_vlm = prepare_easy_krea_vision_image(effective_geometry_source, preset=preset, role="target_geometry")
            geometry_prep = prepare_image_for_qwen(image=g_vlm, clip=clip, original_image=effective_geometry_source)

    latent_dict, latent_info = build_target_latent(
        vae=vae,
        target_content=route.target_content_mode,
        geometry_mode=effective_geometry_mode,
        target_image=target_content_prep,
        geometry_image=geometry_prep,
        batch_size=1,
        force_target_megapixels=common_geometry_active,
        target_alias=route.target_content_role if route.target_content_role else "",
        target_vision_instruction=get_easy_instruction_for_role(route.target_content_role)
        if route.target_content_role
        else "",
    )

    # Delegate to canonical shared orchestrator
    patched_model, pos, neg, lat, orchestrator_info = run_krea2_edit_orchestrator(
        model=model,
        clip=clip,
        vae=vae,
        references=chain,
        target_latent=latent_dict,
        positive_prompt=effective_positive_prompt,
        negative_prompt=negative_prompt,
        reference_method=backend_method,
        apply_model_patch=apply_patch,
        ostris_kv_cache=ostris_kv_cache,
    )

    from ..easy_routing import (
        get_easy_preset_capabilities,
        OUTFIT_POLICY_DISABLED,
        STYLE_POLICY_SCENE_AUTO,
        STYLE_POLICY_DISABLED,
    )

    caps = get_easy_preset_capabilities(preset)
    outfit_policy_str = caps.outfit_policy
    style_policy_str = "automatic scene" if caps.style_policy == STYLE_POLICY_SCENE_AUTO else caps.style_policy

    if caps.outfit_policy == OUTFIT_POLICY_DISABLED:
        resolved_outfit_str = "ignored by preset"
    elif outfit_source == "none":
        resolved_outfit_str = "none"
    else:
        status_str = "present" if resolved_sources.effective_outfit is not None else "missing"
        resolved_outfit_str = f"{outfit_source} ({status_str})"

    if caps.style_policy == STYLE_POLICY_SCENE_AUTO:
        if resolved_sources.effective_style is not None:
            resolved_style_str = "scene image (automatic)"
        else:
            resolved_style_str = "none (scene unavailable)"
    elif caps.style_policy == STYLE_POLICY_DISABLED:
        resolved_style_str = "ignored by preset"
    elif style_source == "none":
        resolved_style_str = "none"
    else:
        status_str = "present" if resolved_sources.effective_style is not None else "missing"
        resolved_style_str = f"{style_source} ({status_str})"

    outfit_selector_report = "ignored by preset" if caps.outfit_policy == OUTFIT_POLICY_DISABLED else outfit_source
    style_selector_report = (
        "ignored by preset" if caps.style_policy in (STYLE_POLICY_SCENE_AUTO, STYLE_POLICY_DISABLED) else style_source
    )

    app_refs = [f"{alias}" for _, _, alias, _ in route.edit_references]
    sem_refs = [f"{alias}" for _, alias in route.semantic_only_references]

    easy_header = [
        "=== Easy Edit Routing Report ===",
        f"Preset: {preset}",
        f"Reference Contract: {backend_method}",
        f"Outfit Policy: {outfit_policy_str}",
        f"Style Policy: {style_policy_str}",
        f"Outfit Source Selector: {outfit_selector_report}",
        f"Style Source Selector: {style_selector_report}",
        f"Use Default Prompt: {'yes' if effective_use_default else 'no'}",
        f"Prompt Source: {prompt_source_str}",
        f"Default Prompt Key: {resolved_key_str}",
        f"Resolved Subject: {'present' if resolved_sources.subject is not None else 'missing'}",
        f"Resolved Scene: {'present' if resolved_sources.scene is not None else 'missing'}",
        f"Resolved Outfit Source: {resolved_outfit_str}",
        f"Resolved Style Source: {resolved_style_str}",
        f"Common Geometry: {'yes' if common_geometry_active else 'no'}",
        f"Common Geometry Anchor: {common_geometry_anchor_role}",
        f"Target Content Mode: {route.target_content_mode}",
        f"Target Content Source: {'present' if route.target_content_source is not None else 'none'}",
        f"Target Geometry Mode: {effective_geometry_mode}",
        f"Target Geometry Source: {'present' if effective_geometry_source is not None else 'none'}",
    ]
    for i, app_ref in enumerate(app_refs):
        easy_header.append(f"Appearance Ref {i + 1}: {app_ref}")
    if not app_refs:
        easy_header.append("Appearance Ref 1: none")

    easy_header.extend(
        [
            f"Semantic-only Sources: {', '.join(sem_refs) if sem_refs else 'none'}",
            f"Style Active: {'yes' if route.style_active else 'no'}",
        ]
    )

    if backend_method == "krea2_edit":
        easy_header.append(f"CcC Krea2 Model Patch Applied: {'yes' if apply_patch else 'no'}")
    elif backend_method == "ostris_edit":
        easy_header.append(
            f"Ostris Reference Method Explicitly Applied: {'yes (conditioning metadata)' if apply_patch else 'no (external runtime expected)'}"
        )
    else:
        easy_header.append(f"Model Patch Applied: {'yes' if apply_patch else 'no'}")

    easy_header.extend(
        [
            f"Warnings: {'; '.join(route.warnings) if route.warnings else 'none'}",
            "",
        ]
    )

    combined_info = "\n".join(easy_header) + "\n" + orchestrator_info
    return (patched_model, pos, neg, lat, combined_info)
