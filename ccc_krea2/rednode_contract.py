"""Small compatibility helpers mirroring the proven RedNode/Krea2Moodboard edit contract."""

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

    Identity/Edit appearance references intentionally match RedNode exactly: their physical
    vision blocks are contiguous and are followed directly by the user's edit instruction.
    Per-reference Visual Reference labels/instructions are metadata only on this path; injecting
    them changes the Qwen sequence from the proven ``VISION_BLOCK * N + instruction`` contract.

    CcC-only semantic/style references may still contribute ordinary annotation text after the
    complete image prefix because they are extensions rather than Identity Edit appearance refs.
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

        # Proven Identity Edit / RedNode path: appearance edit refs are positional. Scene is
        # frame/image 1 and subject is frame/image 2 by workflow order; no label text is injected.
        if ref_path == "edit" and is_appearance:
            physical_index += count
            continue

        alias = _alias_for(item, spec)
        instruction = str(getattr(spec, "vision_instruction", "") or "").strip()
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
    """Build the Identity Edit grounded negative without positive semantic annotations.

    Training unconditional = same appearance images + empty text. A user-supplied negative prompt
    may still be appended because the public CcC Edit node exposes that field, but semantic roles
    and per-reference instructions intentionally never enter the negative pass.
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
