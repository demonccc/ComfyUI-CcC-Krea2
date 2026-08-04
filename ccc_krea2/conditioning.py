"""Qwen3-VL text/vision conditioning builder for positive and negative prompts."""

from typing import List, Tuple, Any, Optional

from .constants import (
    LOGGER_PREFIX,
    VISION_PAD_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    ReferenceRole,
)


def build_role_instructions(role_order: List[ReferenceRole]) -> str:
    """Build Qwen3-VL system prompt role instructions from active image order.

    Canonical terminology: subject image, scene image, outfit image, source image.
    """
    if not role_order:
        return ""

    lines = []
    for i, role in enumerate(role_order, start=1):
        r_val = role.value if hasattr(role, "value") else str(role)
        lines.append(f"Image {i} is the {r_val} image.")

    role_vals = [r.value if hasattr(r, "value") else str(r) for r in role_order]
    instructions = []

    if "scene" in role_vals:
        instructions.append(
            "Use the scene image for composition, pose, environment, interactions and lighting."
        )

    if "outfit" in role_vals:
        instructions.append(
            "Use the outfit image for the outfit and garment details.\n"
            "Do not use the wearer of the outfit image as the subject identity."
        )

    if "subject" in role_vals:
        instructions.append(
            "Use the subject image for identity, facial features, hair, anatomy, body shape and body proportions."
        )

    if "source" in role_vals:
        instructions.append("The source image is the base image being edited.")

    header = "\n".join(lines)
    body = "\n\n".join(instructions) if instructions else ""

    if header and body:
        return f"{header}\n\n{body}"
    return header or body


def build_krea2_qwen_template(num_images: int, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Builds the Qwen3-VL template dynamically matching the exact count of vision images.

    Contract expected by ComfyUI / Qwen3-VL tokenizer uses positional placeholder '{}'.
    """
    vision_blocks = VISION_PAD_TOKEN * num_images
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{vision_blocks}{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def encode_krea2_conditioning(
    clip: Any,
    prompt: str,
    negative_prompt: str,
    grounding_images: List[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[List[Any], List[Any]]:
    """Encodes Qwen3-VL text and vision tokens for Positive and Negative conditionings.

    Single canonical contract: encode_krea2_conditioning(clip, prompt, negative_prompt, grounding_images, system_prompt)
    Performs purely Qwen3-VL semantic grounding. Does NOT attach VAE reference latents or masks to conditioning.
    """
    if clip is None:
        raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

    num_images = len(grounding_images)
    dynamic_template = build_krea2_qwen_template(num_images=num_images, system_prompt=system_prompt)

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


def encode_prompt_with_qwen(
    clip: Any,
    prompt: str,
    images: Optional[List[Any]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Any]:
    """Helper to encode a single prompt (positive or negative) with Qwen3-VL CLIP text encoder."""
    if clip is None:
        return []

    imgs = images if images is not None else []
    num_images = len(imgs)
    dynamic_template = build_krea2_qwen_template(num_images=num_images, system_prompt=system_prompt)

    try:
        tokens = clip.tokenize(prompt or "", images=imgs, llama_template=dynamic_template)
    except TypeError:
        tokens = clip.tokenize(prompt or "", images=imgs)
    except Exception:
        try:
            tokens = clip.tokenize(prompt or "")
        except Exception:
            return []

    return clip.encode_from_tokens_scheduled(tokens)
