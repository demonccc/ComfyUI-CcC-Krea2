"""Qwen3-VL text/vision conditioning builder for positive and negative prompts."""

from typing import List, Tuple, Any, Dict
import torch

from .constants import (
    LOGGER_PREFIX,
    VISION_PAD_TOKEN,
    DEFAULT_SYSTEM_PROMPT
)
from .references import PreparedReference

try:
    import node_helpers
except ImportError:
    node_helpers = None


def build_krea2_qwen_template(num_images: int, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Builds the Qwen3-VL template dynamically matching the exact count of vision images."""
    vision_blocks = VISION_PAD_TOKEN * num_images
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{vision_blocks}{{prompt}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def encode_krea2_conditioning(
    clip: Any,
    prompt: str,
    negative_prompt: str,
    prepared_references: List[PreparedReference]
) -> Tuple[List[Any], List[Any]]:
    """Encodes Qwen3-VL text and vision tokens for Positive and Negative conditionings.

    Dynamically constructs template with exactly one vision placeholder per image supplied.
    Both conditionings share identical reference metadata, RoPE indices, VAE reference latents,
    and grounding image geometry. Only prompt text differs.
    """
    if clip is None:
        raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

    grounding_images = []
    ref_latents = []
    ref_boosts = []
    ref_masks = []
    ref_roles = []
    ref_fit = []

    for ref in prepared_references:
        if ref.grounding_image is not None:
            img = ref.grounding_image
            if img.ndim == 3:
                img = img.unsqueeze(0)
            grounding_images.append(img[:, :, :, :3])

        if ref.vae_latent is not None:
            if isinstance(ref.vae_latent, dict) and "samples" in ref.vae_latent:
                lat_tensor = ref.vae_latent["samples"]
            else:
                lat_tensor = ref.vae_latent
            ref_latents.append(lat_tensor)
            ref_boosts.append(float(ref.boost))
            ref_masks.append(ref.attention_mask)
            ref_roles.append(ref.role.value)
            ref_fit.append(ref.reference_fit_mode == "fit")

    num_images = len(grounding_images)
    dynamic_template = build_krea2_qwen_template(num_images=num_images)

    def _encode_text(text: str) -> List[Any]:
        try:
            tokens = clip.tokenize(text, images=grounding_images, llama_template=dynamic_template)
        except TypeError:
            tokens = clip.tokenize(text, images=grounding_images)
        except Exception as e:
            raise ValueError(
                f"{LOGGER_PREFIX} Failed to tokenize prompt with Qwen3-VL CLIP text encoder: {e}"
            ) from e

        cond = clip.encode_from_tokens_scheduled(tokens)

        if ref_latents:
            extra_dict: Dict[str, Any] = {
                "reference_latents": ref_latents,
                "reference_boosts": ref_boosts,
                "reference_masks": ref_masks,
                "reference_roles": ref_roles,
                "reference_fit": ref_fit,
            }
            if node_helpers is not None and hasattr(node_helpers, "conditioning_set_values"):
                cond = node_helpers.conditioning_set_values(cond, extra_dict, append=True)
            else:
                new_cond = []
                for c, extra in cond:
                    merged_extra = extra.copy()
                    for k, v in extra_dict.items():
                        merged_extra[k] = v
                    new_cond.append([c, merged_extra])
                cond = new_cond

        return cond

    pos_conditioning = _encode_text(prompt or "")
    neg_conditioning = _encode_text(negative_prompt or "")

    return pos_conditioning, neg_conditioning
