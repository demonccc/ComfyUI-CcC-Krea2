"""Qwen3-VL text/vision conditioning builder for positive and negative prompts."""

from typing import List, Tuple, Any

from .constants import (
    LOGGER_PREFIX,
    VISION_PAD_TOKEN,
    DEFAULT_SYSTEM_PROMPT
)


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
    grounding_images: List[Any],
) -> Tuple[List[Any], List[Any]]:
    """Encodes Qwen3-VL text and vision tokens for Positive and Negative conditionings.

    Single canonical contract: encode_krea2_conditioning(clip, prompt, negative_prompt, grounding_images)
    Performs purely Qwen3-VL semantic grounding. Does NOT attach VAE reference latents or masks to conditioning.
    """
    if clip is None:
        raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

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

        return clip.encode_from_tokens_scheduled(tokens)

    pos_conditioning = _encode_text(prompt or "")
    neg_conditioning = _encode_text(negative_prompt or "")

    return pos_conditioning, neg_conditioning
