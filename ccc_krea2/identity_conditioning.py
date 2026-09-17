"""Direct Krea 2 Identity Edit conditioning for visual-only workflows.

Qwen owns semantic grounding. Appearance references are prepared here in pixel space
and handed to the per-MODEL Identity runtime instead of being transported through
CONDITIONING. This keeps the appearance path isolated from third-party global Krea2
conditioning patches while preserving CcC target geometry and RoPE controls.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import torch


from .constants import VISION_PAD_TOKEN
from .grounding import resize_grounding_image
from .krea2edit_geometry import ResolvedGeometry, process_image_and_mask_geometry, resolve_krea2edit_geometry

try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except ImportError:
    KREA2_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )


@dataclass(frozen=True)
class DirectIdentityResult:
    positive: Any
    negative: Any
    geometries: Tuple[ResolvedGeometry, ...]
    qwen_sizes: Tuple[Tuple[int, int], ...]
    reference_latents: Tuple[torch.Tensor, ...]
    reference_latent_shapes: Tuple[Tuple[int, ...], ...]
    reference_boosts: Tuple[float, ...]
    reference_rope_positions: Tuple[str, ...]
    positive_text: str
    negative_text: str


def _as_single_rgb(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"[CcC Krea2] Visual reference must be BHWC image data, got {tuple(image.shape)}")
    return image[:1, :, :, :3]


def _grounding_image(
    image: torch.Tensor,
    semantic_resize: bool,
    grounding_px: int,
    resize_method: str,
) -> torch.Tensor:
    """Prepare the Qwen copy: optional downscale-only cap, never upscale."""
    image = _as_single_rgb(image)
    if not semantic_resize:
        return image
    return resize_grounding_image(
        image=image,
        resize_mode="downscale_only",
        grounding_px=int(grounding_px),
        grounding_preset="custom",
        resize_method=resize_method,
    )[:, :, :, :3]


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
    """Prepare visual refs and grounded Qwen conditioning without appearance metadata.

    Reference order is preserved exactly. For two-reference Identity Edit that means scene
    first and subject second. ``semantic=False`` removes that reference only from Qwen; it
    remains present in the VAE appearance path.
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
        rope_parts = str(entry.rope_position).split(":")
        if len(rope_parts) == 3:
            _, grid_horizontal, grid_vertical = rope_parts
        else:
            grid_horizontal, grid_vertical = "center", "center"
        geom = resolve_krea2edit_geometry(
            src_h=src_h,
            src_w=src_w,
            tgt_h=target_h,
            tgt_w=target_w,
            fit_mode=entry.resolved_fit_mode,
            grid_horizontal_position=grid_horizontal,
            grid_vertical_position=grid_vertical,
            resize_method=entry.resize_method,
        )
        fitted, _ = process_image_and_mask_geometry(raw, None, geom)
        ref_latents.append(_normalize_vae_latent(vae.encode(fitted)))
        geometries.append(geom)
        rope_positions.append(str(entry.rope_position))
        boosts.append(float(entry.boost))

        if bool(entry.semantic):
            qwen_images.append(
                _grounding_image(
                    raw,
                    bool(entry.semantic_resize),
                    int(entry.semantic_grounding_px),
                    str(entry.semantic_resize_method),
                )
            )

    vision_prefix = VISION_PAD_TOKEN * len(qwen_images)
    annotations = []
    qwen_index = 1
    for entry in visual_entries:
        if bool(entry.semantic):
            annotation = str(getattr(entry, "prompt_annotation", "") or "").strip()
            if annotation:
                annotations.append(f"Image {qwen_index}: {annotation}")
            qwen_index += 1
    suffix = "\n".join(annotations + ([positive_prompt] if positive_prompt else []))
    positive_text = vision_prefix + suffix
    negative_text = vision_prefix + (negative_prompt or "")

    # Appearance references deliberately do NOT travel in conditioning on this path.
    # The model wrapper owns them, matching the source-patch architecture and avoiding
    # interference from other packages that globally extend Krea2.extra_conds.
    positive = _encode_qwen(clip, positive_text, qwen_images)
    negative = _encode_qwen(clip, negative_text, qwen_images)

    return DirectIdentityResult(
        positive=positive,
        negative=negative,
        geometries=tuple(geometries),
        qwen_sizes=tuple((int(img.shape[2]), int(img.shape[1])) for img in qwen_images),
        reference_latents=tuple(ref_latents),
        reference_latent_shapes=tuple(tuple(int(v) for v in latent.shape) for latent in ref_latents),
        reference_boosts=tuple(boosts),
        reference_rope_positions=tuple(rope_positions),
        positive_text=positive_text,
        negative_text=negative_text,
    )
