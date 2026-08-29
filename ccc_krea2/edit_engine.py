"""Core orchestrator engine for CcC Krea2 Edit."""

import torch
from typing import Tuple, Dict, Any, Optional, List
from .references import PreparedReference
from .constants import ReferenceRole
from .reference_specs import ReferenceChain, ReferenceSpec, StyleReferenceSpec
from .reference_slots import resolve_reference_slots_and_aliases
from .krea2edit_geometry import resolve_krea2edit_geometry, process_image_and_mask_geometry
from .style_processing import expand_style_reference_spans
from .prompt_augmentation import apply_prompt_augmentation, PromptAugmentation
from .conditioning import (
    encode_krea2_qwen_context,
    build_krea2_user_content,
    build_krea2_negative_user_content,
    attach_reference_latents_to_conditioning,
    attach_reference_latents_method_to_conditioning,
)
from .patch import patch_krea2_model, check_patch_safety
from .ostris_backend import (
    preprocess_ostris_ref_pixel_image,
    preprocess_ostris_vision_image,
    build_ostris_qwen_prompt,
)
from .target_latent import should_include_target_in_vision, TargetVisionContext


def detect_model_default_ref_method(model: Any) -> Optional[str]:
    """Safely inspect connected model patcher for default_ref_method without mutating."""
    if model is None:
        return None
    try:
        model_obj = (
            model.get_model_object("diffusion_model")
            if hasattr(model, "get_model_object")
            else getattr(model, "model", None)
        )
        if model_obj is not None:
            if hasattr(model_obj, "default_ref_method") and getattr(model_obj, "default_ref_method") is not None:
                return str(getattr(model_obj, "default_ref_method"))
            if hasattr(model_obj, "ref_latents_method") and getattr(model_obj, "ref_latents_method") is not None:
                return str(getattr(model_obj, "ref_latents_method"))
        if hasattr(model, "default_ref_method") and getattr(model, "default_ref_method") is not None:
            return str(getattr(model, "default_ref_method"))
    except Exception:
        pass
    return None


def run_krea2_edit_orchestrator(
    model: Any,
    clip: Any,
    vae: Any,
    references: ReferenceChain,
    target_latent: Dict[str, torch.Tensor],
    positive_prompt: str,
    negative_prompt: str,
    reference_method: str = "krea2_edit",
    ostris_kv_cache: bool = False,
    prompt_augmentation: Optional[PromptAugmentation] = None,
    global_vision_directive: str = "",
    **kwargs: Any,
) -> Tuple[Any, Any, Any, Dict[str, torch.Tensor], str]:
    """Execute the modular Krea 2 Edit orchestrator pipeline."""
    if ostris_kv_cache or kwargs.get("ostris_kv_cache", False):
        raise NotImplementedError(
            "[CcC Krea2] ostris_kv_cache=True is currently unsupported. "
            "Intended only for LoRAs trained with ai-toolkit kv_cache."
        )

    # Step 1: Target latent geometry inspection
    samples = target_latent["samples"]
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError(
            f"Target latent samples must be a 4D or 5D tensor, got shape {getattr(samples, 'shape', None)}"
        )

    bs = samples.shape[0]
    lh, lw = samples.shape[-2:]
    target_h = lh * 8
    target_w = lw * 8

    # Target Vision Context inspection
    target_vctx: Optional[TargetVisionContext] = target_latent.get("target_vision_context")

    # Step 2: Unified slot resolution for Target Vision Context and References
    effective_chain = references
    if target_vctx is not None and should_include_target_in_vision(target_vctx, references):
        if target_vctx.target_image is not None:
            target_spec = ReferenceSpec(
                reference_path="edit",
                prepared_image=target_vctx.target_image,
                requested_vision_slot=target_vctx.target_vision_slot,
                alias=target_vctx.target_alias,
                vision_instruction=target_vctx.target_vision_instruction,
                appearance_reference=False,
                _legacy_role="target",
            )
            if target_spec.requested_vision_slot is None or target_spec.requested_vision_slot == 0:
                effective_chain = ReferenceChain((target_spec,) + references.references)
            else:
                effective_chain = effective_chain.append(target_spec)

    resolved_refs, slot_warnings = resolve_reference_slots_and_aliases(effective_chain)
    for i in range(len(resolved_refs) - 1):
        if resolved_refs[i]["resolved_slot"] >= resolved_refs[i + 1]["resolved_slot"]:
            raise ValueError(
                f"[CcC Krea2] Resolved references out of slot order: slot {resolved_refs[i]['resolved_slot']} "
                f"comes before slot {resolved_refs[i + 1]['resolved_slot']}."
            )

    # Step 3: Prompt augmentation layering
    pos_base, neg_base = apply_prompt_augmentation(
        positive_prompt=positive_prompt, negative_prompt=negative_prompt, augmentation=prompt_augmentation
    )

    if global_vision_directive.strip():
        pos_base = f"Global Vision Directive:\n{global_vision_directive.strip()}\n\n{pos_base}"

    # Step 4: Build Qwen vision image maps and reference specs
    pos_qwen_images: List[torch.Tensor] = []
    neg_qwen_images: List[torch.Tensor] = []
    pos_qwen_image_map: List[Dict[str, Any]] = []
    neg_qwen_image_map: List[Dict[str, Any]] = []

    vae_ref_specs: List[Dict[str, Any]] = []
    style_ref_specs: List[Dict[str, Any]] = []
    semantic_ref_specs: List[Dict[str, Any]] = []
    ostris_pic_counter = 1

    for ref_item in resolved_refs:
        spec = ref_item["spec"]
        slot = ref_item["resolved_slot"]
        ref_path = getattr(spec, "reference_path", spec.role.lower())
        ref_role = spec.role.lower()
        is_appearance = getattr(spec, "appearance_reference", True)

        if ref_path == "edit":
            src_img = spec.prepared_image.original_image
            src_h, src_w = src_img.shape[1], src_img.shape[2]

            if is_appearance:
                if reference_method == "ostris_edit":
                    proc_img = preprocess_ostris_ref_pixel_image(src_img)
                    proc_h, proc_w = proc_img.shape[1], proc_img.shape[2]
                    vlm_img = preprocess_ostris_vision_image(src_img)
                    vlm_h, vlm_w = vlm_img.shape[1], vlm_img.shape[2]

                    fit_mask = getattr(spec, "attention_mask", None)
                    encoded = vae.encode(proc_img) if vae is not None else None
                    lat_tokens = None
                    if encoded is not None:
                        lat_tokens = (
                            encoded["samples"]
                            if isinstance(encoded, dict)
                            else (encoded.sample() if hasattr(encoded, "sample") else encoded)
                        )

                    vae_ref_specs.append(
                        {
                            "backend": "ostris_edit",
                            "role": ref_role,
                            "slot": slot,
                            "picture_number": ostris_pic_counter,
                            "latent_tokens": lat_tokens,
                            "mask": fit_mask,
                            "source_size": (src_w, src_h),
                            "vlm_size": (vlm_w, vlm_h),
                            "vae_prep_size": (proc_w, proc_h),
                            "spec": spec,
                            "ref_item": ref_item,
                        }
                    )
                    ostris_pic_counter += 1
                elif reference_method == "native":
                    encoded = vae.encode(src_img) if vae is not None else None
                    lat_tokens = None
                    if encoded is not None:
                        lat_tokens = (
                            encoded["samples"]
                            if isinstance(encoded, dict)
                            else (encoded.sample() if hasattr(encoded, "sample") else encoded)
                        )

                    vae_ref_specs.append(
                        {
                            "backend": "native",
                            "role": ref_role,
                            "slot": slot,
                            "latent_tokens": lat_tokens,
                            "mask": getattr(spec, "attention_mask", None),
                            "source_size": (src_w, src_h),
                            "vae_size": (src_w, src_h),
                            "spec": spec,
                            "ref_item": ref_item,
                        }
                    )
                else:  # krea2_edit
                    fit_mode = getattr(spec, "visual_reference_fit", getattr(spec, "visual_fit_mode", "auto"))
                    geom = resolve_krea2edit_geometry(
                        src_h=src_h, src_w=src_w, tgt_h=target_h, tgt_w=target_w, fit_mode=fit_mode
                    )

                    fit_img, fit_mask = process_image_and_mask_geometry(
                        image=src_img, mask=getattr(spec, "attention_mask", None), geom=geom
                    )

                    encoded = vae.encode(fit_img) if vae is not None else None
                    lat_tokens = None
                    if encoded is not None:
                        lat_tokens = (
                            encoded["samples"]
                            if isinstance(encoded, dict)
                            else (encoded.sample() if hasattr(encoded, "sample") else encoded)
                        )

                    vae_ref_specs.append(
                        {
                            "backend": "krea2_edit",
                            "role": ref_role,
                            "slot": slot,
                            "latent_tokens": lat_tokens,
                            "mask": fit_mask,
                            "geom": geom,
                            "spec": spec,
                            "ref_item": ref_item,
                        }
                    )
            else:
                semantic_ref_specs.append(
                    {
                        "role": ref_role,
                        "slot": slot,
                        "spec": spec,
                        "ref_item": ref_item,
                        "has_instruction": bool(getattr(spec, "vision_instruction", "").strip()),
                    }
                )

            if not getattr(spec, "include_in_vision", True):
                continue

            # Vision image enters positive Qwen list
            pos_idx = len(pos_qwen_images) + 1
            # For Ostris: apply VLM preprocessing to ALL edit-path refs (appearance + semantic-only)
            # For other backends: use prepared vision_image
            if reference_method == "ostris_edit":
                raw_img = spec.prepared_image.original_image
                vis_img = preprocess_ostris_vision_image(raw_img)
            else:
                vis_img = spec.prepared_image.vision_image
            pos_qwen_images.append(vis_img)
            pos_qwen_image_map.append(
                {
                    "role": ref_role,
                    "slot": slot,
                    "logical_reference_id": slot,
                    "logical_role": ref_role,
                    "logical_vision_slot": slot,
                    "physical_qwen_image_index": pos_idx,
                    "style_group_id": None,
                    "crop_tile_index": None,
                    "image": vis_img,
                    "spec": spec,
                }
            )

            # Negative Qwen images: empty for Ostris edit (text-only negative), active for Krea2/native appearance refs
            if reference_method != "ostris_edit" and is_appearance:
                neg_idx = len(neg_qwen_images) + 1
                neg_qwen_images.append(spec.prepared_image.vision_image)
                neg_qwen_image_map.append(
                    {
                        "role": ref_role,
                        "slot": slot,
                        "logical_reference_id": slot,
                        "logical_role": ref_role,
                        "logical_vision_slot": slot,
                        "physical_qwen_image_index": neg_idx,
                        "style_group_id": None,
                        "crop_tile_index": None,
                        "image": spec.prepared_image.vision_image,
                        "spec": spec,
                    }
                )

        elif ref_path == "style":
            assert isinstance(spec, StyleReferenceSpec) or getattr(spec, "reference_path", "") == "style"

            prep_crops, s_start, s_end = expand_style_reference_spans(spec, start_slot=slot, clip=clip)
            phys_range = ref_item.get("physical_qwen_range", (s_start, s_end))
            for crop_idx, crop_prep in enumerate(prep_crops):
                pos_idx = len(pos_qwen_images) + 1
                pos_qwen_images.append(crop_prep.vision_image)
                pos_qwen_image_map.append(
                    {
                        "role": "style",
                        "slot": slot,
                        "logical_reference_id": slot,
                        "logical_role": "style",
                        "logical_vision_slot": slot,
                        "physical_qwen_image_index": pos_idx,
                        "style_group_id": slot,
                        "crop_tile_index": crop_idx,
                        "image": crop_prep.vision_image,
                        "spec": spec,
                    }
                )

            style_ref_specs.append(
                {"role": "style", "slot": slot, "spans": phys_range, "spec": spec, "ref_item": ref_item}
            )

    # Format user content using canonical Qwen prompt formatters
    if reference_method == "ostris_edit":
        user_content = build_ostris_qwen_prompt(resolved_references=resolved_refs, user_prompt=pos_base)
        neg_user_content = neg_base
    else:
        user_content = build_krea2_user_content(resolved_references=resolved_refs, user_prompt=pos_base)
        neg_user_content = build_krea2_negative_user_content(
            resolved_references=resolved_refs, user_negative_prompt=neg_base
        )

    # Step 5: Encode Qwen Contexts for positive and negative
    pos_qwen_context = encode_krea2_qwen_context(
        clip=clip,
        prompt=user_content,
        physical_images=pos_qwen_images,
        physical_image_map=pos_qwen_image_map,
        is_positive=True,
    )

    neg_qwen_context = encode_krea2_qwen_context(
        clip=clip,
        prompt=neg_user_content,
        physical_images=neg_qwen_images,
        physical_image_map=neg_qwen_image_map,
        is_positive=False,
    )

    # Step 6: Prepare model patching references and VAE latents list
    prepared_refs: List[PreparedReference] = []
    vae_latents_for_transport: List[torch.Tensor] = []

    for ref_dict in vae_ref_specs:
        sp = ref_dict["spec"]
        r_role = (
            ReferenceRole(ref_dict["role"])
            if ref_dict["role"] in [r.value for r in ReferenceRole]
            else ref_dict["role"]
        )
        base_boost = getattr(sp, "attention_boost", 1.0)
        masked_boost = getattr(sp, "masked_attention_boost", 1.0)

        if ref_dict.get("latent_tokens") is not None:
            vae_latents_for_transport.append(ref_dict["latent_tokens"])

        if reference_method == "krea2_edit":
            pr = PreparedReference(
                role=r_role,
                grounding_image=sp.prepared_image.vision_image,
                vae_latent=ref_dict["latent_tokens"],
                spatial_attention_mask=ref_dict.get("mask"),
                boost=base_boost,
                masked_boost=masked_boost,
                spatial_hw=ref_dict["geom"].vae_input_pixel_size,
                lat_hw=ref_dict["geom"].vae_latent_grid_size,
                mask_mode="hard",
                ref_fit_meta={"geom": ref_dict["geom"]},
                rope_position=getattr(sp, "rope_position", "none"),
            )
            prepared_refs.append(pr)

    # Step 7: Apply model patches or native reference latents
    apply_model_patch = kwargs.get("apply_model_patch", kwargs.get("apply_patch", True))

    native_def_method = None
    if reference_method == "native":
        check_patch_safety(model, "native")
        patched_model = model
        pos_conditioning = attach_reference_latents_to_conditioning(
            pos_qwen_context.conditioning, vae_latents_for_transport
        )
        neg_conditioning = attach_reference_latents_to_conditioning(
            neg_qwen_context.conditioning, vae_latents_for_transport
        )
        native_def_method = detect_model_default_ref_method(model)
        if vae_latents_for_transport and native_def_method is None:
            slot_warnings.append(
                "reference_latents were attached, but no native default reference method was detected; the connected MODEL/runtime must provide compatible reference behavior."
            )
    elif reference_method == "ostris_edit":
        check_patch_safety(model, "ostris_edit")
        pos_conditioning = attach_reference_latents_to_conditioning(
            pos_qwen_context.conditioning, vae_latents_for_transport
        )
        if apply_model_patch:
            pos_conditioning = attach_reference_latents_method_to_conditioning(pos_conditioning, "index_timestep_zero")
        else:
            slot_warnings.append(
                "Ostris reference method was not explicitly applied; the connected MODEL/runtime must already provide compatible index_timestep_zero behavior."
            )
        neg_conditioning = neg_qwen_context.conditioning
        patched_model = model
    else:  # krea2_edit
        check_patch_safety(model, "krea2_edit")
        if apply_model_patch:
            patched_model = patch_krea2_model(model=model, prepared_refs=prepared_refs)
            pos_conditioning = pos_qwen_context.conditioning
            neg_conditioning = neg_qwen_context.conditioning
        else:
            patched_model = model
            pos_conditioning = attach_reference_latents_to_conditioning(
                pos_qwen_context.conditioning, vae_latents_for_transport
            )
            neg_conditioning = attach_reference_latents_to_conditioning(
                neg_qwen_context.conditioning, vae_latents_for_transport
            )
            slot_warnings.append(
                "CcC Krea2 model patch was skipped; reference latents attached to conditioning require compatible runtime support."
            )

    # Step 8: Build edit_info report
    if reference_method == "krea2_edit":
        patch_text = "applied" if apply_model_patch else "skipped"
    else:
        patch_text = "not applicable"

    ref_transport_text = "CcC Krea2 wrapper"
    if reference_method == "native":
        ref_transport_text = "standard ComfyUI reference_latents"
    elif reference_method == "ostris_edit":
        ref_transport_text = "Ostris index_timestep_zero" if apply_model_patch else "standard ComfyUI reference_latents"
    elif not apply_model_patch:
        ref_transport_text = "standard ComfyUI reference_latents"

    ostris_applied_str = (
        "yes (conditioning metadata)"
        if (reference_method == "ostris_edit" and apply_model_patch)
        else ("no (external runtime expected)" if reference_method == "ostris_edit" else "no")
    )
    kv_cache_str = "disabled (feature currently unsupported)" if reference_method == "ostris_edit" else "not applicable"

    info_lines = [
        "=== CcC Krea2 Edit Pipeline Report ===",
        f"Reference Contract: {reference_method}",
        f"CcC Model Patch: {patch_text}",
        f"Reference Transport: {ref_transport_text}",
    ]
    if reference_method == "native":
        info_lines.append(
            f"Native Default Reference Method: {native_def_method if native_def_method is not None else 'none'}"
        )
    elif reference_method == "ostris_edit":
        ostris_method_status = (
            "explicitly applied via conditioning" if apply_model_patch else "not injected; external runtime expected"
        )
        info_lines.extend(
            [
                f"Ostris Reference Method: {ostris_method_status}",
                f"Ostris Reference Method Applied: {ostris_applied_str}",
                f"Ostris KV Cache: {kv_cache_str}",
            ]
        )
        if apply_model_patch:
            info_lines.append("Reference Latents Method: index_timestep_zero")

    info_lines.extend(
        [
            f"Target Pixel Geometry: {target_w} x {target_h} (Target MP: {(target_h * target_w) / 1_000_000.0:.3f} MP)",
            f"Target Latent Geometry: {lw} x {lh} (Batch Size: {bs})",
            "",
            "Conditioning Summary:",
            f"  Positive Row Count Before Moodboard: {pos_qwen_context.pos_rows_before}",
            f"  Positive Row Count After Moodboard: {pos_qwen_context.pos_rows_after}",
            f"  Negative Row Count: {neg_qwen_context.neg_rows}",
            f"  Positive Physical Qwen Image Count: {len(pos_qwen_images)}",
            f"  Negative Physical Qwen Image Count: {len(neg_qwen_images)}",
            f"  Token Stream Key: {pos_qwen_context.token_stream_key}",
            f"  Global Vision Directive Active: {'yes' if global_vision_directive.strip() else 'no'}",
            f"  Prompt Augmentation Active: {'yes' if prompt_augmentation is not None else 'no'}",
            "",
        ]
    )

    for ref in vae_ref_specs:
        sp = ref["spec"]
        ref_item = ref["ref_item"]
        b_boost = getattr(sp, "attention_boost", 1.0)
        m_boost = getattr(sp, "masked_attention_boost", 1.0)
        aliases_str = ", ".join(ref_item.get("expanded_aliases", ()))
        phys_range = ref_item.get("physical_qwen_range")
        phys_idx = phys_range[0] if phys_range else None
        span_str = (
            str(pos_qwen_context.vision_row_spans[phys_idx - 1])
            if phys_idx is not None and (phys_idx - 1) < len(pos_qwen_context.vision_row_spans)
            else "N/A"
        )
        is_appearance = getattr(sp, "appearance_reference", True)

        info_lines.extend(
            [
                f"Reference [Slot {ref['slot']} - {ref['role'].capitalize()}]:",
                f"  Logical Vision Slot: {ref['slot']}",
                f"  Physical Qwen Image Index: {phys_idx if phys_idx is not None else 'none'}",
                f"  Actual Conditioning Row Span: {span_str}",
                f"  VAE Reference Frame: {ref_item.get('vae_reference_frame') if is_appearance else 'none'}",
                f"  Expanded Aliases: {aliases_str if aliases_str else 'none'}",
            ]
        )

        if reference_method == "krea2_edit":
            geom = ref.get("geom")
            if geom:
                rope_position = getattr(sp, "rope_position", "none")
                target_grid_w, target_grid_h = geom.target_grid_size
                ref_grid_w, ref_grid_h = geom.vae_latent_grid_size
                rope_y, rope_x = geom.centered_fractional_offset
                if rope_position == "up":
                    rope_y = -float(ref_grid_h)
                elif rope_position == "down":
                    rope_y = float(target_grid_h)
                elif rope_position == "left":
                    rope_x = -float(ref_grid_w)
                elif rope_position == "right":
                    rope_x = float(target_grid_w)
                info_lines.extend(
                    [
                        f"  Requested Visual Reference Fit: {geom.mode_requested}",
                        f"  Resolved Visual Reference Fit: {geom.mode_resolved}",
                        f"  Source Crop Rectangle: {geom.crop_rectangle}",
                        f"  VAE Input Size: {geom.vae_input_pixel_size[0]} x {geom.vae_input_pixel_size[1]}",
                        f"  VAE Latent Grid: {geom.vae_latent_grid_size[0]} x {geom.vae_latent_grid_size[1]}",
                        f"  Target Grid: {geom.target_grid_size[0]} x {geom.target_grid_size[1]}",
                        f"  Centered RoPE Offset: Y={geom.centered_fractional_offset[0]:.2f}, X={geom.centered_fractional_offset[1]:.2f}",
                        f"  RoPE Position Override: {rope_position}",
                        f"  Effective RoPE Offset: Y={rope_y:.2f}, X={rope_x:.2f}",
                    ]
                )
            info_lines.extend(
                [
                    f"  Base Attention Boost: {b_boost:.2f} | Masked Attention Boost: {m_boost:.2f}",
                    f"  Has Attention Mask: {'yes' if ref.get('mask') is not None else 'no'}",
                ]
            )
        elif reference_method == "ostris_edit":
            info_lines.extend(
                [
                    f"  Ostris Picture Number: {ref.get('picture_number', phys_idx)}",
                    f"  Source Pixels: {ref.get('source_size', (0, 0))[0]} x {ref.get('source_size', (0, 0))[1]}",
                    f"  VLM Pixels: {ref.get('vlm_size', (0, 0))[0]} x {ref.get('vlm_size', (0, 0))[1]}",
                    f"  VAE Prep Pixels: {ref.get('vae_prep_size', (0, 0))[0]} x {ref.get('vae_prep_size', (0, 0))[1]}",
                ]
            )
        else:  # native
            info_lines.extend(
                [
                    f"  Source Pixels: {ref.get('source_size', (0, 0))[0]} x {ref.get('source_size', (0, 0))[1]}",
                    f"  VAE Pixels: {ref.get('vae_size', (0, 0))[0]} x {ref.get('vae_size', (0, 0))[1]}",
                ]
            )
        info_lines.append("")

    for sem in semantic_ref_specs:
        sp = sem["spec"]
        ref_item = sem["ref_item"]
        aliases_str = ", ".join(ref_item.get("expanded_aliases", ()))
        phys_idx = ref_item.get("physical_qwen_range", (1, 1))[0]
        span_str = (
            str(pos_qwen_context.vision_row_spans[phys_idx - 1])
            if (phys_idx - 1) < len(pos_qwen_context.vision_row_spans)
            else "N/A"
        )

        info_lines.extend(
            [
                f"Semantic-only Reference [Slot {sem['slot']} - {sem['role'].capitalize()}]:",
                f"  Logical Vision Slot: {sem['slot']}",
                f"  Physical Qwen Image Index: {phys_idx}",
                f"  Actual Conditioning Row Span: {span_str}",
                f"  Expanded Aliases: {aliases_str if aliases_str else 'none'}",
                f"  Vision Instruction Active: {'yes' if sem.get('has_instruction') else 'no'}",
                "  Appearance Reference: no",
                "  VAE Reference Frame: none",
                "  Negative Conditioning Included: no",
                "",
            ]
        )

    for st in style_ref_specs:
        sp = st["spec"]
        ref_item = st["ref_item"]
        aliases_str = ", ".join(ref_item.get("expanded_aliases", ()))
        phys_range = ref_item.get("physical_qwen_range", st["spans"])
        s_start, s_end = phys_range
        st_spans = (
            pos_qwen_context.vision_row_spans[s_start - 1 : s_end]
            if (s_start - 1) < len(pos_qwen_context.vision_row_spans)
            else []
        )
        shuffle_str = (
            "SHUFFLE_2X2"
            if sp.style_processing == "2x2"
            else ("SHUFFLE_4X4" if sp.style_processing == "4x4" else "identity")
        )
        style_total_rows = sum(e - s for s, e in st_spans)
        rows_rem = style_total_rows if sp.indirect_style_transfer else 0
        status_str = (
            "indirect (rows removed post-encoding)" if sp.indirect_style_transfer else "direct (rows preserved)"
        )

        info_lines.extend(
            [
                f"Style [Slot {st['slot']}]:",
                f"  Logical Vision Slot: {st['slot']}",
                f"  Physical Qwen Image Start: {s_start}",
                f"  Physical Qwen Image End: {s_end}",
                f"  Physical Qwen Image Count: {s_end - s_start + 1}",
                f"  Actual Conditioning Row Spans: {st_spans}",
                f"  Total Rows for Style: {style_total_rows}",
                f"  Rows Removed for Style: {rows_rem}",
                f"  Direct/Indirect Status: {status_str}",
                f"  Style Reference Processing: {sp.style_processing}",
                f"  Crop Shuffle Order: {shuffle_str}",
                f"  Style Fidelity: {sp.style_fidelity:.2f}",
                f"  Indirect Style Transfer: {sp.indirect_style_transfer}",
                "  VAE Reference Frame: none",
                f"  Expanded Aliases: {aliases_str if aliases_str else 'none'}",
                "",
            ]
        )

    all_warnings = list(slot_warnings) + pos_qwen_context.warnings + neg_qwen_context.warnings
    if all_warnings:
        info_lines.append("Warnings:")
        for w in all_warnings:
            info_lines.append(f"  - {w}")

    edit_info = "\n".join(info_lines)

    return patched_model, pos_conditioning, neg_conditioning, target_latent, edit_info
