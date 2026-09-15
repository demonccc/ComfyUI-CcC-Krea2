"""Small compatibility helpers mirroring the proven RedNode/Krea2Moodboard edit contract."""

from typing import Any, Dict, List

from .constants import VISION_PAD_TOKEN


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
