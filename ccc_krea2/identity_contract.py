"""Krea2 Identity Edit conditioning contract helpers."""

from typing import Any, Dict, List

from .constants import VISION_PAD_TOKEN


def _alias_for(item: Dict[str, Any], spec: Any) -> str:
    expanded = item.get("expanded_aliases", ())
    if expanded:
        return ", ".join(expanded)
    return str(getattr(spec, "alias", "") or "")


def _style_block_count(spec: Any) -> int:
    processing = getattr(spec, "style_processing", "2x2")
    if processing == "full":
        return 1
    if processing == "4x4":
        return 16
    return 4


def build_grounded_positive_user_content(
    resolved_references: List[Dict[str, Any]],
    user_prompt: str = "",
) -> str:
    """Build the Krea2 positive user turn.

    Physical vision blocks stay contiguous and preserve Krea2 reference order. Appearance
    references are positional by default; when a Visual Reference carries prompt_annotation,
    it is appended as Image N: <annotation> after the complete vision prefix. Semantic/style
    references keep their own optional annotations. The raw user edit instruction remains last.
    """
    blocks: List[str] = []
    annotations: List[str] = []
    physical_index = 1

    for item in resolved_references:
        spec = item.get("spec")
        if spec is None or not getattr(spec, "include_in_vision", True):
            continue

        ref_path = getattr(spec, "reference_path", getattr(spec, "role", "").lower())
        is_appearance = bool(getattr(spec, "appearance_reference", True))
        count = _style_block_count(spec) if ref_path == "style" else 1
        blocks.extend([VISION_PAD_TOKEN] * count)

        alias = _alias_for(item, spec).strip()
        instruction = str(getattr(spec, "vision_instruction", "") or "").strip()

        if ref_path == "edit" and is_appearance:
            label = f"Image {physical_index}"
            if instruction:
                annotations.append(f"{label}: {instruction}")
            elif alias:
                # Compatibility for older role-based reference nodes.
                annotations.append(f"{label}: {alias}")
            physical_index += count
            continue

        if alias or instruction:
            if count == 1:
                label = f"Image {physical_index}"
            else:
                label = f"Images {physical_index}-{physical_index + count - 1}"
            if alias and instruction:
                annotations.append(f"{label} ({alias}): {instruction}")
            elif alias:
                annotations.append(f"{label}: {alias}")
            else:
                annotations.append(f"{label}: {instruction}")

        physical_index += count

    prefix = "".join(blocks)
    text_parts = annotations + ([user_prompt] if user_prompt else [])
    return prefix + ("\n".join(text_parts) if text_parts else "")


def build_grounded_negative_user_content(
    resolved_references: List[Dict[str, Any]],
    user_negative_prompt: str = "",
) -> str:
    """Build the Identity Edit grounded unconditional branch.

    The negative branch receives the same appearance images in the same order, but no role text
    or edit instruction. Split CcC Edit fixes ``user_negative_prompt`` to the empty string and
    keeps negative reference boosts neutral at 1.0.
    """
    blocks = []
    for item in resolved_references:
        spec = item.get("spec")
        if spec is None or not getattr(spec, "include_in_vision", True):
            continue
        ref_path = getattr(spec, "reference_path", getattr(spec, "role", "").lower())
        if ref_path == "style" or not getattr(spec, "appearance_reference", True):
            continue
        blocks.append(VISION_PAD_TOKEN)

    return "".join(blocks) + (user_negative_prompt or "")
