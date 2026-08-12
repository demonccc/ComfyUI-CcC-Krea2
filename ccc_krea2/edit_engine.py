"""Core orchestrator engine for CcC Krea2 Edit."""

import torch
from typing import Tuple, Dict, Any, Optional, List
from .references import PreparedReference
from .constants import ReferenceRole
from .reference_specs import (
    ReferenceChain,
    StyleReferenceSpec
)
from .reference_slots import resolve_reference_slots_and_aliases
from .reference_directives import build_automatic_role_directive
from .krea2edit_geometry import (
    resolve_krea2edit_geometry,
    process_image_and_mask_geometry
)
from .style_processing import expand_style_reference_spans
from .prompt_augmentation import apply_prompt_augmentation, PromptAugmentation
from .conditioning import encode_krea2_qwen_context, build_annotated_user_prompt
from .patch import patch_krea2_model
from .ostris_backend import patch_ostris_model
from .target_latent import should_include_target_in_vision, TargetVisionContext


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
    **kwargs: Any
) -> Tuple[Any, Any, Any, Dict[str, torch.Tensor], str]:
    """Execute the modular Krea 2 Edit orchestrator pipeline."""
    # Step 1: Target latent geometry inspection
    samples = target_latent["samples"]
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError(
            f"Target latent samples must be a 4D or 5D tensor, "
            f"got shape {getattr(samples, 'shape', None)}"
        )

    bs = samples.shape[0]
    lh, lw = samples.shape[-2:]
    target_h = lh * 8
    target_w = lw * 8

    # Target Vision Context inspection
    target_vctx: Optional[TargetVisionContext] = target_latent.get("target_vision_context")

    # Step 2: Resolve reference slots, aliases, VAE frames, and physical Qwen indices
    resolved_refs, slot_warnings = resolve_reference_slots_and_aliases(references)
    for i in range(len(resolved_refs) - 1):
        if resolved_refs[i]["resolved_slot"] >= resolved_refs[i + 1]["resolved_slot"]:
            raise ValueError(
                f"[CcC Krea2] Resolved references out of slot order: slot {resolved_refs[i]['resolved_slot']} "
                f"comes before slot {resolved_refs[i+1]['resolved_slot']}."
            )

    # Step 3: Prompt augmentation layering
    pos_base, neg_base = apply_prompt_augmentation(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        augmentation=prompt_augmentation
    )

    # Step 4: Build Qwen vision image maps and reference specs
    pos_qwen_images: List[torch.Tensor] = []
    neg_qwen_images: List[torch.Tensor] = []
    pos_qwen_image_map: List[Dict[str, Any]] = []
    neg_qwen_image_map: List[Dict[str, Any]] = []

    vae_ref_specs: List[Dict[str, Any]] = []
    style_ref_specs: List[Dict[str, Any]] = []

    role_directives_pos: List[str] = []
    role_directives_neg: List[str] = []

    if global_vision_directive.strip():
        pos_dir_str = f"Global Vision Directive:\n{global_vision_directive.strip()}"
        role_directives_pos.append(pos_dir_str)
        role_directives_neg.append(pos_dir_str)

    # Add Target Vision Context image if enabled and non-duplicate
    if target_vctx is not None and should_include_target_in_vision(target_vctx, references):
        if target_vctx.target_image is not None:
            t_prep = target_vctx.target_image
            pos_idx = len(pos_qwen_images) + 1
            pos_qwen_images.append(t_prep.vision_image)
            pos_qwen_image_map.append({
                "role": "target",
                "slot": target_vctx.target_vision_slot or (len(resolved_refs) + 1),
                "logical_reference_id": "target",
                "logical_role": "target",
                "logical_vision_slot": target_vctx.target_vision_slot or (len(resolved_refs) + 1),
                "physical_qwen_image_index": pos_idx,
                "image": t_prep.vision_image,
                "spec": None
            })

    for ref_item in resolved_refs:
        spec = ref_item["spec"]
        slot = ref_item["resolved_slot"]
        ref_path = getattr(spec, "reference_path", spec.role.lower())
        ref_role = spec.role.lower()

        auto_dir = build_automatic_role_directive(ref_item)

        if ref_path == "edit":
            if auto_dir:
                role_directives_pos.append(auto_dir)
                role_directives_neg.append(auto_dir)

            src_img = spec.prepared_image.original_image
            src_h, src_w = src_img.shape[1], src_img.shape[2]

            fit_mode = getattr(spec, "visual_reference_fit", getattr(spec, "visual_fit_mode", "auto"))
            geom = resolve_krea2edit_geometry(
                src_h=src_h,
                src_w=src_w,
                tgt_h=target_h,
                tgt_w=target_w,
                fit_mode=fit_mode
            )

            fit_img, fit_mask = process_image_and_mask_geometry(
                image=src_img,
                mask=getattr(spec, "attention_mask", None),
                geom=geom
            )

            # VAE encode reference latent
            encoded = vae.encode(fit_img)
            lat_tokens = encoded["samples"] if isinstance(encoded, dict) else (encoded.sample() if hasattr(encoded, "sample") else encoded)

            vae_ref_specs.append({
                "role": ref_role,
                "slot": slot,
                "latent_tokens": lat_tokens,
                "mask": fit_mask,
                "geom": geom,
                "spec": spec,
                "ref_item": ref_item
            })

            # Non-style images enter both positive and negative Qwen lists
            pos_idx = len(pos_qwen_images) + 1
            pos_qwen_images.append(spec.prepared_image.vision_image)
            pos_qwen_image_map.append({
                "role": ref_role,
                "slot": slot,
                "logical_reference_id": slot,
                "logical_role": ref_role,
                "logical_vision_slot": slot,
                "physical_qwen_image_index": pos_idx,
                "style_group_id": None,
                "crop_tile_index": None,
                "image": spec.prepared_image.vision_image,
                "spec": spec
            })

            neg_idx = len(neg_qwen_images) + 1
            neg_qwen_images.append(spec.prepared_image.vision_image)
            neg_qwen_image_map.append({
                "role": ref_role,
                "slot": slot,
                "logical_reference_id": slot,
                "logical_role": ref_role,
                "logical_vision_slot": slot,
                "physical_qwen_image_index": neg_idx,
                "style_group_id": None,
                "crop_tile_index": None,
                "image": spec.prepared_image.vision_image,
                "spec": spec
            })

        elif ref_path == "style":
            assert isinstance(spec, StyleReferenceSpec) or getattr(spec, "reference_path", "") == "style"
            if auto_dir:
                role_directives_pos.append(auto_dir)

            prep_crops, s_start, s_end = expand_style_reference_spans(spec, start_slot=slot, clip=clip)
            phys_range = ref_item.get("physical_qwen_range", (s_start, s_end))
            for crop_idx, crop_prep in enumerate(prep_crops):
                pos_idx = len(pos_qwen_images) + 1
                pos_qwen_images.append(crop_prep.vision_image)
                pos_qwen_image_map.append({
                    "role": "style",
                    "slot": slot,
                    "logical_reference_id": slot,
                    "logical_role": "style",
                    "logical_vision_slot": slot,
                    "physical_qwen_image_index": pos_idx,
                    "style_group_id": slot,
                    "crop_tile_index": crop_idx,
                    "image": crop_prep.vision_image,
                    "spec": spec
                })

            style_ref_specs.append({
                "role": "style",
                "slot": slot,
                "spans": phys_range,
                "spec": spec,
                "ref_item": ref_item
            })

    # Format user prompt with slot annotations
    annotated_prompt = build_annotated_user_prompt(
        resolved_references=resolved_refs,
        user_prompt=pos_base,
        target_vision_context=target_vctx,
    )

    pos_dir_block = "\n\n".join(role_directives_pos)
    neg_dir_block = "\n\n".join(role_directives_neg)

    full_positive_prompt = f"{pos_dir_block}\n\n{annotated_prompt}".strip() if pos_dir_block else annotated_prompt
    full_negative_prompt = f"{neg_dir_block}\n\n{neg_base}".strip() if neg_dir_block else neg_base

    # Step 5: Encode Qwen Contexts for positive and negative
    pos_qwen_context = encode_krea2_qwen_context(
        clip=clip,
        prompt=full_positive_prompt,
        physical_images=pos_qwen_images,
        physical_image_map=pos_qwen_image_map,
        is_positive=True
    )

    neg_qwen_context = encode_krea2_qwen_context(
        clip=clip,
        prompt=full_negative_prompt,
        physical_images=neg_qwen_images,
        physical_image_map=neg_qwen_image_map,
        is_positive=False
    )

    # Step 6: Prepare model patching references
    prepared_refs: List[PreparedReference] = []
    for ref_dict in vae_ref_specs:
        sp = ref_dict["spec"]
        r_role = ReferenceRole(ref_dict["role"]) if ref_dict["role"] in [r.value for r in ReferenceRole] else ReferenceRole.SUBJECT
        base_boost = getattr(sp, "attention_boost", 1.0)
        masked_boost = getattr(sp, "masked_attention_boost", 1.0)

        pr = PreparedReference(
            role=r_role,
            grounding_image=sp.prepared_image.vision_image,
            vae_latent=ref_dict["latent_tokens"],
            spatial_attention_mask=ref_dict["mask"],
            boost=base_boost,
            masked_boost=masked_boost,
            spatial_hw=ref_dict["geom"].vae_input_pixel_size,
            lat_hw=ref_dict["geom"].vae_latent_grid_size,
            mask_mode="hard",
            ref_fit_meta={"geom": ref_dict["geom"]}
        )
        prepared_refs.append(pr)

    # Step 7: Apply model patches according to reference_method
    if reference_method == "native":
        patched_model = model
    elif reference_method == "ostris_edit":
        patched_model = patch_ostris_model(model=model, prepared_refs=prepared_refs, ostris_kv_cache=ostris_kv_cache)
    else:
        # Default: "krea2_edit"
        patched_model = patch_krea2_model(model=model, prepared_refs=prepared_refs)

    # Step 8: Build edit_info report
    info_lines = [
        "=== CcC Krea2 Edit Pipeline Report ===",
        f"Reference Method: {reference_method}",
        f"Ostris KV Cache: {'yes' if ostris_kv_cache else 'no'}",
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
        f"  Template Prefix Rows Removed: {pos_qwen_context.template_prefix_rows_removed}",
        f"  Global Vision Directive Active: {'yes' if global_vision_directive.strip() else 'no'}",
        f"  Prompt Augmentation Active: {'yes' if prompt_augmentation is not None else 'no'}",
        ""
    ]

    for ref in vae_ref_specs:
        sp = ref["spec"]
        geom = ref["geom"]
        ref_item = ref["ref_item"]
        b_boost = getattr(sp, "attention_boost", 1.0)
        m_boost = getattr(sp, "masked_attention_boost", 1.0)
        aliases_str = ", ".join(ref_item.get("expanded_aliases", ()))
        phys_idx = ref_item.get("physical_qwen_range", (1, 1))[0]
        span_str = str(pos_qwen_context.vision_row_spans[phys_idx - 1]) if (phys_idx - 1) < len(pos_qwen_context.vision_row_spans) else "N/A"
        role_name = ref["role"].lower()
        if role_name == "subject":
            dir_controls_str = "Pose Anchor, Outfit Anchor, Masked Identity Anchor"
        elif role_name == "outfit":
            dir_controls_str = "Outfit Anchor"
        elif role_name == "scene":
            dir_controls_str = "Scene Anchor, Masked Region Anchor"
        else:
            dir_controls_str = "none"

        info_lines.extend([
            f"Reference [Slot {ref['slot']} - {ref['role'].capitalize()}]:",
            f"  Logical Vision Slot: {ref['slot']}",
            f"  Physical Qwen Image Index: {phys_idx}",
            f"  Actual Conditioning Row Span: {span_str}",
            f"  VAE Reference Frame: {ref_item.get('vae_reference_frame')}",
            f"  Expanded Aliases: {aliases_str if aliases_str else 'none'}",
            "  Anchor Implementation Type: Vision directive only",
            f"  Directive-only Anchor Controls: {dir_controls_str}",
            f"  Requested Visual Reference Fit: {geom.mode_requested}",
            f"  Resolved Visual Reference Fit: {geom.mode_resolved}",
            f"  Source Crop Rectangle: {geom.crop_rectangle}",
            f"  VAE Input Size: {geom.vae_input_pixel_size[0]} x {geom.vae_input_pixel_size[1]}",
            f"  VAE Latent Grid: {geom.vae_latent_grid_size[0]} x {geom.vae_latent_grid_size[1]}",
            f"  Target Grid: {geom.target_grid_size[0]} x {geom.target_grid_size[1]}",
            f"  RoPE Offset: Y={geom.centered_fractional_offset[0]:.2f}, X={geom.centered_fractional_offset[1]:.2f}",
            f"  Base Attention Boost: {b_boost:.2f} | Masked Attention Boost: {m_boost:.2f}",
            f"  Has Attention Mask: {'yes' if ref['mask'] is not None else 'no'}",
            ""
        ])



    for st in style_ref_specs:
        sp = st["spec"]
        ref_item = st["ref_item"]
        aliases_str = ", ".join(ref_item.get("expanded_aliases", ()))
        phys_range = ref_item.get("physical_qwen_range", st["spans"])
        s_start, s_end = phys_range
        st_spans = pos_qwen_context.vision_row_spans[s_start - 1 : s_end] if (s_start - 1) < len(pos_qwen_context.vision_row_spans) else []
        shuffle_str = "SHUFFLE_2X2" if sp.style_processing == "2x2" else ("SHUFFLE_4X4" if sp.style_processing == "4x4" else "identity")
        style_total_rows = sum(e - s for s, e in st_spans)
        rows_rem = style_total_rows if sp.indirect_style_transfer else 0
        status_str = "indirect (rows removed post-encoding)" if sp.indirect_style_transfer else "direct (rows preserved)"

        info_lines.extend([
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
            f"  Style Directive: {sp.style_directive}",
            "  VAE Reference Frame: none",
            f"  Expanded Aliases: {aliases_str if aliases_str else 'none'}",
            ""
        ])

    all_warnings = list(slot_warnings) + pos_qwen_context.warnings + neg_qwen_context.warnings
    if all_warnings:
        info_lines.append("Warnings:")
        for w in all_warnings:
            info_lines.append(f"  - {w}")

    edit_info = "\n".join(info_lines)

    return patched_model, pos_qwen_context.conditioning, neg_qwen_context.conditioning, target_latent, edit_info
