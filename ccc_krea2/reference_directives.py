"""Automatic vision directive generation for Subject, Scene, Outfit, and Style roles."""

from typing import List, Dict, Any
from ccc_krea2.reference_specs import (
    SubjectReferenceSpec,
    SceneReferenceSpec,
    OutfitReferenceSpec,
    StyleReferenceSpec
)


def build_automatic_role_directive(spec_dict: Dict[str, Any]) -> str:
    """Generate automatic per-reference text directives from specification parameters using physical Qwen mapping."""
    spec = spec_dict["spec"]
    slot = spec_dict["resolved_slot"]
    aliases = spec_dict.get("expanded_aliases", ())
    phys_range = spec_dict.get("physical_qwen_range", (slot, slot))

    start_phys, end_phys = phys_range
    if start_phys == end_phys:
        img_id = f"Image {start_phys}"
    else:
        img_id = f"Images {start_phys} through {end_phys}"

    alias_str = ", ".join(f"'{a}'" for a in aliases) if aliases else f"'{img_id}'"

    role = spec.role.lower()
    directives: List[str] = []

    if role == "subject":
        assert isinstance(spec, SubjectReferenceSpec)
        directives.append(f"{img_id} (referred to as {alias_str}) is the subject image.")
        directives.append("Use the subject image for identity, facial features, body structure, and person appearance.")
        if spec.pose_anchor > 0.0:
            directives.append(f"Anchor the subject pose with weight {spec.pose_anchor:.2f}.")
        if spec.outfit_anchor > 0.0:
            directives.append(f"Anchor the subject outfit with weight {spec.outfit_anchor:.2f}.")
        if getattr(spec, "masked_identity_anchor", 0.0) > 0.0 and spec.attention_mask is not None:
            directives.append(f"Anchor masked identity region with weight {spec.masked_identity_anchor:.2f}.")

    elif role == "scene":
        assert isinstance(spec, SceneReferenceSpec)
        directives.append(f"{img_id} (referred to as {alias_str}) is the scene image.")
        directives.append("Use the scene image for composition, background, environment, lighting, and camera framing.")
        if spec.scene_anchor > 0.0:
            directives.append(f"Anchor the scene structure with weight {spec.scene_anchor:.2f}.")
        if getattr(spec, "masked_region_anchor", 0.0) > 0.0 and spec.attention_mask is not None:
            directives.append(f"Anchor masked scene region with weight {spec.masked_region_anchor:.2f}.")

    elif role == "outfit":
        assert isinstance(spec, OutfitReferenceSpec)
        directives.append(f"{img_id} (referred to as {alias_str}) is the outfit reference image.")
        directives.append("Use only the clothing, garments, and accessories from the outfit image.")
        directives.append("Do not use the wearer's face, identity, body, pose, or background from the outfit image.")
        if spec.outfit_anchor > 0.0:
            directives.append(f"Anchor the outfit design with weight {spec.outfit_anchor:.2f}.")

    elif role == "style":
        assert isinstance(spec, StyleReferenceSpec)
        if spec.style_directive:
            directives.append(f"{img_id} (referred to as {alias_str}) is the style reference image.")
            directives.append("Transfer only the color palette, lighting, texture, linework, rendering style, tone, mood, and artistic finish.")
            directives.append("Do not copy subjects, identities, outfits, objects, poses, backgrounds, or composition layout from the style image.")
            directives.append(f"Apply style fidelity weight {float(spec.style_fidelity):.2f}.")

    if spec.extra_vision_directive:
        directives.append(spec.extra_vision_directive.strip())

    return "\n".join(directives)
