"""Qwen3-VL text/vision conditioning builder for positive and negative prompts."""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict
from ccc_krea2.constants import (
    LOGGER_PREFIX,
    VISION_PAD_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    ReferenceRole,
)
from ccc_krea2.style_processing import apply_statistical_style_fidelity


@dataclass
class EncodedQwenContext:
    tokens: Any
    conditioning: List[Any]
    physical_images: List[Any]
    physical_image_map: List[Dict[str, Any]]
    vision_row_spans: List[Tuple[int, int]]


def build_role_instructions(role_order: List[ReferenceRole]) -> str:
    """Build Qwen3-VL system prompt role instructions from active image order."""
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
    """Builds the Qwen3-VL template dynamically matching the exact count of vision images."""
    vision_blocks = VISION_PAD_TOKEN * num_images
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{vision_blocks}{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def encode_krea2_qwen_context(
    clip: Any,
    prompt: str,
    physical_images: List[Any],
    physical_image_map: List[Dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    is_positive: bool = True
) -> EncodedQwenContext:
    """Tokenize and encode prompt with Qwen, returning EncodedQwenContext with span metadata."""
    if clip is None:
        raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

    num_images = len(physical_images)
    dynamic_template = build_krea2_qwen_template(num_images=num_images, system_prompt=system_prompt)

    try:
        tokens = clip.tokenize(prompt or "", images=physical_images, llama_template=dynamic_template)
    except TypeError:
        tokens = clip.tokenize(prompt or "", images=physical_images)
    except Exception as e:
        raise ValueError(
            f"{LOGGER_PREFIX} Failed to tokenize prompt with Qwen CLIP text encoder: {e}"
        ) from e

    conditioning = clip.encode_from_tokens_scheduled(tokens)

    # Calculate row spans per image in conditioning
    vision_row_spans: List[Tuple[int, int]] = []
    curr_idx = 0
    for item in physical_image_map:
        # Check image tensor size to calculate vision token count
        img = item.get("image")
        if img is not None and hasattr(img, "shape") and len(img.shape) >= 3:
            h, w = img.shape[-3], img.shape[-2]
            # Standard Qwen3-VL factor 32
            tokens_count = (h // 32) * (w // 32)
        else:
            tokens_count = 256  # standard default grid

        vision_row_spans.append((curr_idx, curr_idx + tokens_count - 1))
        curr_idx += tokens_count

    # Apply Moodboard style processing (Fidelity & Indirect Transfer) for positive conditioning
    if is_positive and conditioning:
        for idx, item in enumerate(physical_image_map):
            if item.get("role") == "style":
                spec = item.get("spec")
                fidelity = getattr(spec, "style_fidelity", 1.0)
                indirect = getattr(spec, "indirect_style_transfer", False)

                if fidelity < 1.0:
                    for cond_entry in conditioning:
                        if isinstance(cond_entry, (list, tuple)) and len(cond_entry) > 0:
                            cond_tensor = cond_entry[0]
                            if isinstance(cond_tensor, torch.Tensor):
                                # Transform conditioning tensor
                                cond_entry[0] = apply_statistical_style_fidelity(cond_tensor, fidelity)

                if indirect:
                    # Remove style visual rows for indirect style transfer
                    span_start, span_end = vision_row_spans[idx]
                    for cond_entry in conditioning:
                        if isinstance(cond_entry, (list, tuple)) and len(cond_entry) > 0:
                            cond_tensor = cond_entry[0]
                            if isinstance(cond_tensor, torch.Tensor) and cond_tensor.shape[1] > span_end:
                                # Slice out style tokens from sequence dimension
                                pre = cond_tensor[:, :span_start, :]
                                post = cond_tensor[:, span_end + 1 :, :]
                                cond_entry[0] = torch.cat([pre, post], dim=1)

    return EncodedQwenContext(
        tokens=tokens,
        conditioning=conditioning,
        physical_images=physical_images,
        physical_image_map=physical_image_map,
        vision_row_spans=vision_row_spans,
    )


def encode_krea2_conditioning(
    clip: Any,
    prompt: str,
    negative_prompt: str,
    grounding_images: List[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[List[Any], List[Any]]:
    """Legacy helper encoding Qwen3-VL text and vision tokens for Positive and Negative conditionings."""
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
                f"{LOGGER_PREFIX} Failed to tokenize prompt with Qwen CLIP text encoder: {e}"
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
    """Helper to encode a single prompt with Qwen3-VL CLIP text encoder."""
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
