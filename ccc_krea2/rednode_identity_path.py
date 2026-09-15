"""Direct Krea2 Identity Edit conditioning path.

This module intentionally bypasses the generic CcC edit orchestrator for visual-only
Identity Edit workflows. It mirrors RedNodeAI/ComfyUI-Krea2Moodboard's Krea2IdentityEdit
encode contract while keeping CcC's already-resolved target geometry and RoPE placement.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .constants import VISION_PAD_TOKEN
from .krea2edit_geometry import ResolvedGeometry, process_image_and_mask_geometry, resolve_krea2edit_geometry

try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except ImportError:
    KREA2_TEMPLATE = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


@dataclass(frozen=True)
class DirectIdentityResult:
    positive: Any
    negative: Any
    geometries: Tuple[ResolvedGeometry, ...]
    qwen_sizes: Tuple[Tuple[int, int], ...]
    reference_latent_shapes: Tuple[Tuple[int, ...], ...]
    positive_text: str
    negative_text: str


def _as_single_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"[CcC Krea2] Visual reference must be BHWC image data, got {tuple(image.shape)}")
    return image[:1, :, :, :3]


def _grounding_image(image: torch.Tensor, grounding_px: int) -> torch.Tensor:
    """Match RedNode Identity Edit grounding: longest-side AREA cap, never upscale."""
    image = _as_single_rgb(image)
    samples = image.movedim(-1, 1)
    height, width = int(samples.shape[2]), int(samples.shape[3])

    if grounding_px and max(height, width) > int(grounding_px):
        scale_by = float(grounding_px) / float(max(height, width))
        out_w = max(1, round(width * scale_by))
        out_h = max(1, round(height * scale_by))
        try:
            import comfy.utils

            samples = comfy.utils.common_upscale(samples, out_w, out_h, "area", "disabled")
        except (ImportError, AttributeError):
            samples = F.interpolate(samples.float(), size=(out_h, out_w), mode="area")

    return samples.movedim(1, -1)[:, :, :, :3]


def _normalize_vae_latent(encoded: Any) -> torch.Tensor:
    if torch.is_tensor(encoded):
        return encoded
    if isinstance(encoded, dict) and torch.is_tensor(encoded.get("samples")):
        return encoded["samples"]
    if hasattr(encoded, "sample"):
        sampled = encoded.sample()
        if torch.is_tensor(sampled):
            return sampled
    raise TypeError(f"[CcC Krea2] Unsupported VAE encode result: {type(encoded).__name__}")


def _conditioning_set_values(conditioning: Any, values: dict, append: bool = True) -> Any:
    try:
        import node_helpers

        return node_helpers.conditioning_set_values(conditioning, values, append=append)
    except (ImportError, AttributeError):
        updated = []
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                updated.append(item)
                continue
            tensor, extras = item[0], dict(item[1]) if isinstance(item[1], dict) else {}
            for key, value in values.items():
                if append and key in extras and isinstance(extras[key], (list, tuple)) and isinstance(value, (list, tuple)):
                    extras[key] = list(extras[key]) + list(value)
                else:
                    extras[key] = value
            updated.append([tensor, extras])
        return updated


def _encode_qwen(clip: Any, text: str, images: List[torch.Tensor]) -> Any:
    try:
        if images:
            tokens = clip.tokenize(text, images=images, llama_template=KREA2_TEMPLATE)
        else:
            tokens = clip.tokenize(text, llama_template=KREA2_TEMPLATE)
    except Exception as exc:
        raise ValueError(
            "[CcC Krea2] Direct Identity path could not encode the Krea2 Qwen image prompt. "
            f"Load a Qwen3-VL encoder with CLIP type 'krea2'. Inner error: {exc}"
        ) from exc

    if isinstance(tokens, dict) and "qwen3vl_4b" not in tokens:
        raise ValueError(
            "[CcC Krea2] Direct Identity path expected token stream 'qwen3vl_4b'; "
            f"received: {', '.join(tokens.keys())}"
        )
    return clip.encode_from_tokens_scheduled(tokens)


def encode_visual_identity_direct(
    clip: Any,
    vae: Any,
    visual_entries: Sequence[Any],
    target_latent: dict,
    positive_prompt: str,
    negative_prompt: str = "",
) -> DirectIdentityResult:
    """Encode visual Identity refs directly using the proven RedNode contract.

    Visual-reference order is preserved exactly (scene first, subject second by workflow
    convention). CcC semantic_role/instruction fields remain UI/report metadata and do not
    alter the Identity Qwen stream. References with ``semantic=False`` still participate in
    the VAE in-context path but are deliberately omitted from Qwen grounding.
    """
    if vae is None:
        raise ValueError("[CcC Krea2] Visual Identity references require a VAE.")
    samples = target_latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim not in (4, 5):
        raise ValueError("[CcC Krea2] Direct Identity path requires a resolved target latent.")

    target_h = int(samples.shape[-2]) * 8
    target_w = int(samples.shape[-1]) * 8

    qwen_images: List[torch.Tensor] = []
    ref_latents: List[torch.Tensor] = []
    geometries: List[ResolvedGeometry] = []
    rope_positions: List[str] = []
    boosts: List[float] = []

    for entry in visual_entries:
        raw = _as_single_rgb(entry.image)
        src_h, src_w = int(raw.shape[1]), int(raw.shape[2])
        geom = resolve_krea2edit_geometry(
            src_h=src_h,
            src_w=src_w,
            tgt_h=target_h,
            tgt_w=target_w,
            fit_mode="fit",
        )
        fitted, _ = process_image_and_mask_geometry(raw, None, geom)
        ref_latents.append(_normalize_vae_latent(vae.encode(fitted)))
        geometries.append(geom)
        rope_positions.append(str(entry.rope_position))
        boosts.append(float(entry.boost))

        if bool(entry.semantic):
            qwen_images.append(_grounding_image(raw, int(entry.grounding_px)))

    vision_prefix = VISION_PAD_TOKEN * len(qwen_images)
    positive_text = vision_prefix + (positive_prompt or "")
    negative_text = vision_prefix + (negative_prompt or "")

    positive = _encode_qwen(clip, positive_text, qwen_images)
    negative = _encode_qwen(clip, negative_text, qwen_images)

    common = {
        "reference_latents": ref_latents,
        "reference_fit": [True] * len(ref_latents),
        "reference_rope_positions": rope_positions,
    }
    positive_values = dict(common)
    if any(boost != 1.0 for boost in boosts):
        positive_values["reference_boosts"] = boosts

    # RedNode's grounded negative uses the same VAE refs and fit geometry but no boost
    # override. The Krea2 runtime therefore resolves every negative ref boost to 1.0.
    positive = _conditioning_set_values(positive, positive_values, append=True)
    negative = _conditioning_set_values(negative, common, append=True)

    return DirectIdentityResult(
        positive=positive,
        negative=negative,
        geometries=tuple(geometries),
        qwen_sizes=tuple((int(img.shape[2]), int(img.shape[1])) for img in qwen_images),
        reference_latent_shapes=tuple(tuple(int(v) for v in latent.shape) for latent in ref_latents),
        positive_text=positive_text,
        negative_text=negative_text,
    )
