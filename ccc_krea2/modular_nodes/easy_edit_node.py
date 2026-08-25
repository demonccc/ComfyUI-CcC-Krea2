"""CcC Krea2 Easy Edit and Easy Edit Ostris custom nodes."""

from typing import Tuple, Optional, Dict, Any
import torch

from ..easy_routing import (
    resolve_easy_sources,
    route_easy_preset,
    get_easy_instruction_for_role,
    resolve_default_positive_prompt,
    resolve_easy_visual_reference_fit,
    resolve_easy_outfit_vision_instruction,
    EASY_PRESET_CAPABILITIES,
    get_easy_preset_capabilities,
    OUTFIT_POLICY_DISABLED,
    OUTFIT_POLICY_SUBJECT_AUTO,
    STYLE_POLICY_SCENE_AUTO,
    STYLE_POLICY_DISABLED,
)
from ..grounding import prepare_easy_krea_vision_image
from ..vision_prep import prepare_image_for_qwen
from ..reference_specs import ReferenceSpec, StyleReferenceSpec, ReferenceChain
from ..target_latent import (
    EASY_ASPECT_RATIOS,
    EASY_GEOMETRY_HARD_CAP_MEGAPIXELS,
    build_target_latent,
    calculate_canvas_aspect_geometry,
    calculate_subject_aware_scene_geometry,
)
from ..edit_engine import run_krea2_edit_orchestrator
from ..ostris_backend import preprocess_ostris_vision_image
from ..constants import NODE_CATEGORY

EASY_EDIT_PRESETS = list(EASY_PRESET_CAPABILITIES.keys())


class CcCKrea2EasyEdit:
    """Opinionated Easy Edit node for Krea2/Identity Edit workflows."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"

    DESCRIPTION = (
        f"Opinionated {len(EASY_PRESET_CAPABILITIES)}-preset Krea2 Edit node. "
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
                    EASY_EDIT_PRESETS,
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."},
                ),
                "reference_subject": (
                    "STRING",
                    {
                        "default": "main subject",
                        "multiline": False,
                        "tooltip": "Description of the target subject in the scene reference image.",
                    },
                ),
                "subject_description": (
                    "STRING",
                    {
                        "default": "main subject",
                        "multiline": False,
                        "tooltip": "Description of the subject from the subject reference image.",
                    },
                ),
                "outfit_source": (
                    ["none", "subject image", "scene image", "outfit image", "style image"],
                    {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."},
                ),
                "style_source": (
                    ["none", "scene image", "subject image", "style image"],
                    {"default": "none", "tooltip": "Source image socket to use for style conditioning."},
                ),
                "aspect_ratio": (
                    list(EASY_ASPECT_RATIOS),
                    {
                        "default": "auto",
                        "tooltip": "Output canvas ratio. Explicit ratios expand around Subject first, otherwise Scene, without resizing the anchor unless the 2.5 MP hard cap requires it.",
                    },
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
        reference_subject: str = "main subject",
        subject_description: str = "main subject",
        outfit_source: str = "outfit image",
        style_source: str = "none",
        aspect_ratio: str = "auto",
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
            reference_subject=reference_subject,
            subject_description=subject_description,
            outfit_source=outfit_source,
            style_source=style_source,
            aspect_ratio=aspect_ratio,
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
        f"Opinionated {len(EASY_PRESET_CAPABILITIES)}-preset Ostris Edit node. "
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
                    EASY_EDIT_PRESETS,
                    {"default": "balanced", "tooltip": "Selects the routing preset recipe."},
                ),
                "reference_subject": (
                    "STRING",
                    {
                        "default": "main subject",
                        "multiline": False,
                        "tooltip": "Description of the target subject in the scene reference image.",
                    },
                ),
                "subject_description": (
                    "STRING",
                    {
                        "default": "main subject",
                        "multiline": False,
                        "tooltip": "Description of the subject from the subject reference image.",
                    },
                ),
                "outfit_source": (
                    ["none", "subject image", "scene image", "outfit image", "style image"],
                    {"default": "outfit image", "tooltip": "Source image socket to use for outfit conditioning."},
                ),
                "style_source": (
                    ["none", "scene image", "subject image", "style image"],
                    {"default": "none", "tooltip": "Source image socket to use for style conditioning."},
                ),
                "aspect_ratio": (
                    list(EASY_ASPECT_RATIOS),
                    {
                        "default": "auto",
                        "tooltip": "Output canvas ratio. Explicit ratios expand around Subject first, otherwise Scene, without resizing the anchor unless the 2.5 MP hard cap requires it.",
                    },
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
        reference_subject: str = "main subject",
        subject_description: str = "main subject",
        outfit_source: str = "outfit image",
        style_source: str = "none",
        aspect_ratio: str = "auto",
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
            reference_subject=reference_subject,
            subject_description=subject_description,
            outfit_source=outfit_source,
            style_source=style_source,
            aspect_ratio=aspect_ratio,
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
    aspect_ratio: str,
    backend_method: str,
    apply_patch: bool,
    ostris_kv_cache: bool,
    use_default_prompt: bool = True,
    reference_subject: str = "main subject",
    subject_description: str = "main subject",
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
        outfit_source=resolved_sources.outfit_source,
        style_source=resolved_sources.style_source,
        reference_subject=reference_subject,
        subject_description=subject_description,
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

        requires_explicit_custom_prompt = preset == "scene_reinterpretation" and not has_default
        if requires_explicit_custom_prompt and not effective_positive_prompt:
            raise ValueError(
                "Scene Reinterpretation requires a custom positive prompt when both Subject and Scene are not available."
            )

        if is_subject_only and not effective_positive_prompt:
            raise ValueError(
                "Subject-only Easy Edit requires a positive prompt because no default editing intent is available."
            )

    # Phase 2: Preset Routing
    route = route_easy_preset(
        resolved_sources,
        preset=preset,
        reference_subject=reference_subject,
    )

    outfit_vision_instruction = resolve_easy_outfit_vision_instruction(
        outfit_source_kind=resolved_sources.outfit_source_kind,
        use_default_prompt=effective_use_default,
        reference_subject=reference_subject,
        subject_description=subject_description,
    )

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
        if alias_role == "outfit":
            instruction = outfit_vision_instruction
        elif alias_role == "scene+outfit":
            instruction = f"{get_easy_instruction_for_role('scene')}\n\n{outfit_vision_instruction}"
        if is_ostris:
            vlm_img = preprocess_ostris_vision_image(item_img)
        else:
            vlm_img = prepare_easy_krea_vision_image(item_img, preset=preset, role=alias_role)

        fit_mode = resolve_easy_visual_reference_fit(
            preset=preset, role=alias_role, common_geometry_active=common_geometry_active
        )
        if aspect_ratio != "auto" and alias_role in ("scene", "scene+outfit"):
            fit_mode = "contain_no_upscale"

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
            vision_instruction=(
                outfit_vision_instruction if alias_role == "outfit" else get_easy_instruction_for_role(alias_role)
            ),
            appearance_reference=False,
            _legacy_role=alias_role,
        )
        chain = chain.append(spec)

    if route.semantic_outfit_active and route.semantic_outfit_source is not None:
        outfit_style_vlm = prepare_easy_krea_vision_image(
            route.semantic_outfit_source, preset=preset, role="outfit"
        )
        prep_outfit_style = prepare_image_for_qwen(
            image=outfit_style_vlm,
            clip=clip,
            original_image=route.semantic_outfit_source,
        )
        outfit_style_spec = StyleReferenceSpec(
            reference_path="style",
            prepared_image=prep_outfit_style,
            alias="outfit",
            style_fidelity=route.semantic_outfit_config.style_fidelity,
            style_processing=route.semantic_outfit_config.style_processing,
            indirect_style_transfer=route.semantic_outfit_config.indirect_style_transfer,
            vision_instruction=outfit_vision_instruction,
            _legacy_role="outfit",
        )
        chain = chain.append(outfit_style_spec)

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

    canvas_geometry = None
    canvas_anchor_role = "none"
    use_single_anchor_canvas = (has_subject and not has_scene) or (has_scene and not has_subject)
    if aspect_ratio != "auto" or use_single_anchor_canvas:
        if has_subject:
            canvas_anchor_role = "subject"
            canvas_anchor_image = resolved_sources.subject
        elif has_scene:
            canvas_anchor_role = "scene"
            canvas_anchor_image = resolved_sources.scene
        else:
            canvas_anchor_image = None

        if canvas_anchor_image is not None:
            canvas_geometry = calculate_canvas_aspect_geometry(
                anchor_image=canvas_anchor_image,
                aspect_ratio=aspect_ratio,
                max_megapixels=EASY_GEOMETRY_HARD_CAP_MEGAPIXELS,
            )

    subject_geometry = None
    if (
        canvas_geometry is None
        and aspect_ratio == "auto"
        and has_scene
        and has_subject
        and effective_geometry_source is resolved_sources.scene
    ):
        subject_geometry = calculate_subject_aware_scene_geometry(
            scene_image=resolved_sources.scene,
            subject_image=resolved_sources.subject,
            max_megapixels=EASY_GEOMETRY_HARD_CAP_MEGAPIXELS,
        )

    explicit_target_width = None
    explicit_target_height = None
    if canvas_geometry is not None:
        explicit_target_width = canvas_geometry.target_width
        explicit_target_height = canvas_geometry.target_height
    elif subject_geometry is not None:
        explicit_target_width = subject_geometry.target_width
        explicit_target_height = subject_geometry.target_height

    effective_target_content_fit = (
        "contain_no_upscale"
        if canvas_geometry is not None and route.target_content_source is not None
        else route.target_content_fit
    )

    latent_dict, latent_info = build_target_latent(
        vae=vae,
        target_content=route.target_content_mode,
        content_fit=effective_target_content_fit,
        geometry_mode=effective_geometry_mode,
        target_image=target_content_prep,
        geometry_image=geometry_prep,
        batch_size=1,
        force_target_megapixels=(
            common_geometry_active and subject_geometry is None and canvas_geometry is None
        ),
        target_width=explicit_target_width,
        target_height=explicit_target_height,
        target_alias=route.target_content_role if route.target_content_role else "",
        target_vision_instruction=get_easy_instruction_for_role(route.target_content_role)
        if route.target_content_role
        else "",
    )

    if subject_geometry is not None:
        latent_info = "\n".join(
            [
                latent_info,
                "Subject-Aware Scene Geometry: yes",
                f"Scene Original Size: {subject_geometry.scene_width} x {subject_geometry.scene_height}",
                f"Subject Original Size: {subject_geometry.subject_width} x {subject_geometry.subject_height}",
                f"Subject /16 Conditioning Size: {subject_geometry.aligned_subject_width} x {subject_geometry.aligned_subject_height}",
                f"Subject-Aware Target Size: {subject_geometry.target_width} x {subject_geometry.target_height}",
                f"Geometry Hard Cap: {subject_geometry.max_megapixels:.1f} MP",
                f"Scene Downscaled To Hard Cap: {'yes' if subject_geometry.scene_was_downscaled else 'no'}",
                f"Latent Expanded For Subject: {'yes' if subject_geometry.latent_was_expanded else 'no'}",
                f"Latent Reached Hard Cap: {'yes' if subject_geometry.latent_was_capped else 'no'}",
                f"Subject Requires Downscale: {'yes' if subject_geometry.subject_requires_downscale else 'no'}",
            ]
        )

    if canvas_geometry is not None:
        latent_info = "\n".join(
            [
                latent_info,
                f"Aspect Ratio: {canvas_geometry.aspect_ratio}",
                f"Canvas Anchor: {canvas_anchor_role}",
                f"Canvas Anchor Original Size: {canvas_geometry.anchor_width} x {canvas_geometry.anchor_height}",
                f"Canvas Anchor /16 Size: {canvas_geometry.aligned_anchor_width} x {canvas_geometry.aligned_anchor_height}",
                f"Canvas Target Size: {canvas_geometry.target_width} x {canvas_geometry.target_height}",
                f"Canvas Hard Cap: {canvas_geometry.max_megapixels:.1f} MP",
                f"Canvas Reached Hard Cap: {'yes' if canvas_geometry.latent_was_capped else 'no'}",
                f"Canvas Anchor Requires Downscale: {'yes' if canvas_geometry.anchor_requires_downscale else 'no'}",
            ]
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

    caps = get_easy_preset_capabilities(preset)
    outfit_policy_str = caps.outfit_policy
    effective_outfit_selector = resolved_sources.outfit_source
    if caps.style_policy == STYLE_POLICY_SCENE_AUTO:
        style_policy_str = "automatic scene"
    else:
        style_policy_str = caps.style_policy

    if caps.outfit_policy == OUTFIT_POLICY_SUBJECT_AUTO:
        resolved_outfit_str = (
            "subject image (automatic)"
            if resolved_sources.effective_outfit is not None
            else "none (subject unavailable)"
        )
        outfit_selector_report = "automatic (subject image)"
    elif caps.outfit_policy == OUTFIT_POLICY_DISABLED:
        resolved_outfit_str = "ignored by preset"
        outfit_selector_report = "ignored by preset"
    elif effective_outfit_selector == "none":
        resolved_outfit_str = "none"
        outfit_selector_report = effective_outfit_selector
    else:
        status_str = "present" if resolved_sources.effective_outfit is not None else "missing"
        resolved_outfit_str = f"{effective_outfit_selector} ({status_str})"
        outfit_selector_report = effective_outfit_selector

    if caps.style_policy == STYLE_POLICY_SCENE_AUTO:
        if resolved_sources.effective_style is not None:
            resolved_style_str = "scene image (automatic)"
        else:
            resolved_style_str = "none (scene unavailable)"
        style_selector_report = "automatic (scene image)"
    elif caps.style_policy == STYLE_POLICY_DISABLED:
        resolved_style_str = "ignored by preset"
        style_selector_report = "ignored by preset"
    elif style_source == "none":
        resolved_style_str = "none"
        style_selector_report = style_source
    else:
        status_str = "present" if resolved_sources.effective_style is not None else "missing"
        resolved_style_str = f"{style_source} ({status_str})"
        style_selector_report = style_source

    app_refs = []
    for _, boost, alias, _ in route.edit_references:
        fit_str = resolve_easy_visual_reference_fit(
            preset=preset, role=alias, common_geometry_active=common_geometry_active
        )
        if aspect_ratio != "auto" and alias in ("scene", "scene+outfit"):
            fit_str = "contain_no_upscale"
        app_refs.append(f"{alias} (boost={boost:.1f}, fit={fit_str})")
    sem_refs = [f"{alias}" for _, alias in route.semantic_only_references]

    if route.target_content_mode == "empty" or route.target_content_source is None:
        resolved_latent_source_str = "empty"
        target_content_role_str = "none"
    else:
        src = route.target_content_source
        is_sc = resolved_sources.scene is not None and src is resolved_sources.scene
        is_s = resolved_sources.subject is not None and src is resolved_sources.subject
        is_o = resolved_sources.effective_outfit is not None and src is resolved_sources.effective_outfit

        if is_sc and is_o:
            resolved_latent_source_str = "scene image (also Outfit source)"
        elif is_sc:
            resolved_latent_source_str = "scene image"
        elif is_s:
            resolved_latent_source_str = "subject image"
        elif is_o:
            resolved_latent_source_str = "outfit image"
        else:
            resolved_latent_source_str = "present"

        if route.target_content_role and route.target_content_role != "":
            target_content_role_str = route.target_content_role
        else:
            target_content_role_str = (
                "scene+outfit"
                if (is_sc and is_o)
                else ("scene" if is_sc else ("subject" if is_s else ("outfit" if is_o else "none")))
            )

    if effective_geometry_source is None:
        target_geometry_source_str = "none"
    else:
        is_sc_geom = resolved_sources.scene is not None and effective_geometry_source is resolved_sources.scene
        is_s_geom = resolved_sources.subject is not None and effective_geometry_source is resolved_sources.subject
        is_o_geom = (
            resolved_sources.effective_outfit is not None
            and effective_geometry_source is resolved_sources.effective_outfit
        )
        if is_sc_geom and is_o_geom:
            target_geometry_source_str = "scene image (also Outfit source)"
        elif is_sc_geom:
            target_geometry_source_str = "scene image"
        elif is_s_geom:
            target_geometry_source_str = "subject image"
        elif is_o_geom:
            target_geometry_source_str = "outfit image"
        else:
            target_geometry_source_str = "present"

    ref_subj_str = reference_subject.strip() if reference_subject and reference_subject.strip() else "main subject"
    subj_desc_str = subject_description.strip() if subject_description and subject_description.strip() else "main subject"

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
        f"Reference Subject: {ref_subj_str}",
        f"Subject Description: {subj_desc_str}",
        f"Effective Positive Prompt: {effective_positive_prompt}",
        f"Resolved Subject: {'present' if resolved_sources.subject is not None else 'missing'}",
        f"Resolved Scene: {'present' if resolved_sources.scene is not None else 'missing'}",
        f"Resolved Outfit Source: {resolved_outfit_str}",
        f"Resolved Style Source: {resolved_style_str}",
        f"Aspect Ratio: {aspect_ratio}",
        f"Canvas Anchor: {canvas_anchor_role if canvas_geometry is not None else 'automatic'}",
        f"Common Geometry: {'yes' if common_geometry_active else 'no'}",
        f"Common Geometry Anchor: {common_geometry_anchor_role}",
        f"Target Content Mode: {route.target_content_mode}",
        f"Target Content Role: {target_content_role_str}",
        f"Resolved Latent Source: {resolved_latent_source_str}",
        f"Target Content Fit: {effective_target_content_fit}",
        f"Target Geometry Mode: {effective_geometry_mode}",
        f"Target Geometry Source: {target_geometry_source_str}",
    ]
    for i, app_ref in enumerate(app_refs):
        easy_header.append(f"Appearance Ref {i + 1}: {app_ref}")
    if not app_refs:
        easy_header.append("Appearance Ref 1: none")
    elif len(app_refs) == 1:
        easy_header.append("Appearance Ref 2: none")

    easy_header.extend(
        [
            f"Semantic-only Sources: {', '.join(sem_refs) if sem_refs else 'none'}",
            f"Semantic Outfit Active: {'yes' if route.semantic_outfit_active else 'no'}",
            f"Outfit Selection Mode: {'placeholder-aware automatic' if effective_use_default else 'visual interpretation'}",
            f"Style Active: {'yes' if route.style_active else 'no'}",
        ]
    )

    if route.semantic_outfit_active:
        easy_header.extend(
            [
                "Outfit Style Active: yes",
                f"Outfit Style Source: {resolved_sources.outfit_source}",
                f"Outfit Style Processing: {route.semantic_outfit_config.style_processing}",
                f"Outfit Style Fidelity: {route.semantic_outfit_config.style_fidelity:.1f}",
                f"Outfit Style Indirect: {'yes' if route.semantic_outfit_config.indirect_style_transfer else 'no'}",
                "Outfit Style Instruction Active: yes",
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
