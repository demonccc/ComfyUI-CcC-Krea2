"""Core orchestrator engine for CcC Krea2 Edit."""

import torch
from typing import Tuple, Dict, Any, Optional, List
from ccc_krea2.references import PreparedReference
from ccc_krea2.constants import ReferenceRole
from ccc_krea2.reference_specs import (
    ReferenceChain,
    StyleReferenceSpec
)
from ccc_krea2.reference_slots import resolve_reference_slots_and_aliases
from ccc_krea2.reference_directives import build_automatic_role_directive
from ccc_krea2.krea2edit_geometry import resolve_visual_reference_fit
from ccc_krea2.style_processing import expand_style_reference_spans
from ccc_krea2.conditioning import encode_prompt_with_qwen
from ccc_krea2.patch import patch_krea2_model


def run_krea2_edit_orchestrator(
    model: Any,
    clip: Any,
    vae: Any,
    references: ReferenceChain,
    target_latent: Dict[str, torch.Tensor],
    positive_prompt: str,
    negative_prompt: str,
    prompt_augmentation: Optional[Any] = None,
    global_vision_directive: str = ""
) -> Tuple[Any, Any, Any, Dict[str, torch.Tensor], str]:
    """Execute the full modular Krea 2 Edit pipeline."""
    # 1. Inspect target latent geometry
    samples = target_latent["samples"]
    bs, c, lh, lw = samples.shape
    target_h = lh * 8
    target_w = lw * 8

    # 2. Resolve slots and parse aliases
    resolved_refs, slot_warnings = resolve_reference_slots_and_aliases(references)

    # Sort resolved references by physical vision slot
    resolved_refs.sort(key=lambda r: r["resolved_slot"])

    # 3. Process visual references and prepare Qwen vision image list
    qwen_vision_images: List[torch.Tensor] = []
    vae_ref_specs: List[Dict[str, Any]] = []
    style_ref_specs: List[Dict[str, Any]] = []

    directives: List[str] = []
    if global_vision_directive.strip():
        directives.append(global_vision_directive.strip())

    for ref_item in resolved_refs:
        spec = ref_item["spec"]
        slot = ref_item["resolved_slot"]
        ref_role = spec.role.lower()

        # Build automatic vision directive
        auto_dir = build_automatic_role_directive(ref_item)
        if auto_dir:
            directives.append(auto_dir)

        if ref_role in ("subject", "scene", "outfit"):
            # Transform reference image and mask against target latent geometry
            fit_img, fit_mask, fit_meta = resolve_visual_reference_fit(
                image=spec.prepared_image.original_image,
                target_h=target_h,
                target_w=target_w,
                mode=getattr(spec, "visual_fit_mode", "auto"),
                mask=getattr(spec, "attention_mask", None)
            )

            # Encode VAE reference
            encoded = vae.encode(fit_img)
            lat_tokens = encoded["samples"] if isinstance(encoded, dict) else (encoded.sample() if hasattr(encoded, "sample") else encoded)

            vae_ref_specs.append({
                "role": ref_role,
                "slot": slot,
                "latent_tokens": lat_tokens,
                "mask": fit_mask,
                "fit_meta": fit_meta,
                "spec": spec
            })

            qwen_vision_images.append(spec.prepared_image.vision_image)

        elif ref_role == "style":
            assert isinstance(spec, StyleReferenceSpec)
            prep_crops, s_start, s_end = expand_style_reference_spans(spec, start_slot=slot, clip=clip)
            for crop_prep in prep_crops:
                qwen_vision_images.append(crop_prep.vision_image)

            style_ref_specs.append({
                "role": "style",
                "slot": slot,
                "spans": (s_start, s_end),
                "spec": spec
            })

    combined_directives_text = "\n\n".join(directives)

    # 4. Prompt Augmentation Layering
    pos_text = positive_prompt
    neg_text = negative_prompt
    if prompt_augmentation is not None and hasattr(prompt_augmentation, "augment_prompt"):
        pos_text = prompt_augmentation.augment_prompt(pos_text)
        if neg_text.strip():
            neg_text = prompt_augmentation.augment_prompt(neg_text)

    # Append combined directives text to positive prompt context
    full_positive_prompt = f"{combined_directives_text}\n\n{pos_text}".strip() if combined_directives_text else pos_text

    # 5. Encode Positive & Negative Conditioning
    pos_cond = encode_prompt_with_qwen(
        clip=clip,
        prompt=full_positive_prompt,
        images=qwen_vision_images
    )
    neg_cond = encode_prompt_with_qwen(
        clip=clip,
        prompt=neg_text,
        images=qwen_vision_images if qwen_vision_images else None
    )

    # 6. Patch Model
    prepared_refs: List[PreparedReference] = []
    for ref_dict in vae_ref_specs:
        sp = ref_dict["spec"]
        r_role = ReferenceRole(ref_dict["role"]) if ref_dict["role"] in [r.value for r in ReferenceRole] else ReferenceRole.SUBJECT
        pr = PreparedReference(
            role=r_role,
            grounding_image=sp.prepared_image.vision_image,
            vae_latent=ref_dict["latent_tokens"],
            spatial_attention_mask=ref_dict["mask"],
            boost=getattr(sp, "attention_boost", 1.0),
            spatial_hw=ref_dict["fit_meta"]["spatial_hw"],
            lat_hw=(ref_dict["latent_tokens"].shape[-2], ref_dict["latent_tokens"].shape[-1]),
            mask_mode="hard",
            ref_fit_meta=ref_dict["fit_meta"]
        )
        prepared_refs.append(pr)

    patched_model = patch_krea2_model(
        model=model,
        prepared_refs=prepared_refs
    )

    # 7. Generate edit_info
    info_lines = [
        "Target",
        f"Pixel Geometry: {target_w} x {target_h}",
        f"Latent Geometry: {lw} x {lh}",
        f"Target MP: {(target_h * target_w) / 1_000_000.0:.3f}",
        ""
    ]

    for ref in vae_ref_specs:
        sp = ref["spec"]
        fit_m = ref["fit_meta"]
        info_lines.extend([
            "Reference",
            f"Logical Role: {ref['role'].capitalize()}",
            f"Resolved Vision Slot: {ref['slot']}",
            f"Visual Fit Requested: {getattr(sp, 'visual_fit_mode', 'auto')}",
            f"Visual Fit Resolved: {fit_m['mode_resolved']}",
            f"VAE Input Size: {fit_m['spatial_hw'][1]} x {fit_m['spatial_hw'][0]}",
            f"Attention Boost: {getattr(sp, 'attention_boost', 1.0):.2f}",
            ""
        ])

    for st in style_ref_specs:
        sp = st["spec"]
        info_lines.extend([
            "Style",
            f"Logical Slot: {st['slot']}",
            f"Processing: {sp.style_processing}",
            f"Physical Vision Spans: {st['spans'][0]}-{st['spans'][1]}",
            f"Style Fidelity: {sp.style_fidelity:.2f}",
            f"Indirect Style Transfer: {sp.indirect_style_transfer}",
            ""
        ])

    info_lines.extend([
        "Conditioning",
        f"Positive Prompt Length: {len(full_positive_prompt)}",
        f"Negative Prompt Length: {len(neg_text)}",
        f"Global Vision Directive: {'yes' if global_vision_directive.strip() else 'no'}",
        f"Qwen Vision Physical Images: {len(qwen_vision_images)}",
        f"VAE Patch References: {len(vae_ref_specs)}"
    ])

    if slot_warnings:
        info_lines.append("")
        info_lines.append("Warnings")
        for w in slot_warnings:
            info_lines.append(f"- {w}")

    return patched_model, pos_cond, neg_cond, target_latent, "\n".join(info_lines)
