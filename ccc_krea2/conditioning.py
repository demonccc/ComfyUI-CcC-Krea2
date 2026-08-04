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
    warnings: List[str]


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


IM_START, USER, NEWLINE = 151644, 872, 198


def extract_vision_spans_from_tokens(
    tokens: Any,
    physical_image_map: List[Dict[str, Any]],
    factor: int = 32
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Extract vision row spans from Qwen token stream using real Qwen grid geometry after template prefix stripping."""
    warnings: List[str] = []
    tok_pairs = []

    if isinstance(tokens, dict):
        # Resolve active token stream robustly
        found_key = None
        for k in ("qwen3vl_4b", "qwen_vl", "qwen3vl"):
            if k in tokens and tokens[k]:
                found_key = k
                tok_pairs = tokens[k][0] if isinstance(tokens[k], list) and len(tokens[k]) > 0 else tokens[k]
                break

        if not found_key and physical_image_map:
            # Check for any dictionary entry containing token list
            for k, v in tokens.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    found_key = k
                    tok_pairs = v[0]
                    break

        if not found_key and physical_image_map:
            raise ValueError(
                f"{LOGGER_PREFIX} Incompatible CLIP token structure. Available token keys: {list(tokens.keys())}"
            )
    elif isinstance(tokens, list):
        tok_pairs = tokens

    # Calculate real Qwen output rows per physical image based on prepared dimensions and encoder factor
    expected_rows: List[int] = []
    for item in physical_image_map:
        img = item.get("image")
        if img is not None and hasattr(img, "shape") and len(img.shape) >= 2:
            ph, pw = img.shape[-2], img.shape[-1]
            rows = max(1, (ph // factor) * (pw // factor))
        else:
            rows = 256
        expected_rows.append(rows)

    if not tok_pairs:
        # Fallback when CLIP is a mock object without token pairs
        warnings.append("Token pair stream empty; using prepared image grid dimensions for vision spans.")
        spans = []
        curr_idx = 0
        for r in expected_rows:
            spans.append((curr_idx, curr_idx + r))
            curr_idx += r
        return spans, warnings

    spans = []
    rows_counter = 0
    template_end = -1
    count_im_start = 0
    ids = []
    img_idx = 0

    for v in tok_pairs:
        elem = v[0] if isinstance(v, (list, tuple)) else v
        if isinstance(elem, dict):
            # embedded image dictionary
            n = expected_rows[img_idx] if img_idx < len(expected_rows) else 256
            spans.append((rows_counter, rows_counter + n))
            ids.append(None)
            rows_counter += n
            img_idx += 1
        else:
            try:
                tid = int(elem) if not torch.is_tensor(elem) else -1
            except (ValueError, TypeError):
                tid = -1
            if tid == IM_START and count_im_start < 2:
                template_end = rows_counter
                count_im_start += 1
            ids.append(tid)
            rows_counter += 1

    if template_end >= 0 and len(ids) > (template_end + 3):
        if ids[template_end + 1] == USER and ids[template_end + 2] == NEWLINE:
            template_end += 3

    template_end = max(template_end, 0)
    adjusted_spans = [(max(s - template_end, 0), e - template_end) for s, e in spans if e > template_end]

    # Validate spans
    if physical_image_map and len(adjusted_spans) != len(physical_image_map):
        warnings.append(
            f"Mapped vision spans count ({len(adjusted_spans)}) differs from physical images ({len(physical_image_map)})."
        )

    for i in range(len(adjusted_spans) - 1):
        s1, e1 = adjusted_spans[i]
        s2, e2 = adjusted_spans[i + 1]
        if s2 < e1:
            raise ValueError(f"{LOGGER_PREFIX} Overlapping vision row spans detected: {adjusted_spans}.")

    return adjusted_spans, warnings


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

    warnings_list: List[str] = []
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

    # Extract vision row spans using real Qwen image grid geometry
    vision_row_spans, span_warnings = extract_vision_spans_from_tokens(
        tokens=tokens,
        physical_image_map=physical_image_map,
        factor=32
    )
    warnings_list.extend(span_warnings)

    # Apply Moodboard style processing (Fidelity & Indirect Transfer) for positive conditioning
    if is_positive and conditioning and physical_image_map:
        spans_info: List[Tuple[Tuple[int, int], float, bool]] = []
        for idx, item in enumerate(physical_image_map):
            if item.get("role") == "style" and idx < len(vision_row_spans):
                spec = item.get("spec")
                fidelity = getattr(spec, "style_fidelity", 1.0)
                indirect = getattr(spec, "indirect_style_transfer", False)
                spans_info.append((vision_row_spans[idx], fidelity, indirect))

        if spans_info:
            new_conditioning = []
            for cond_entry in conditioning:
                if isinstance(cond_entry, (list, tuple)) and len(cond_entry) > 0:
                    cond_tensor = cond_entry[0]
                    extras = cond_entry[1] if len(cond_entry) > 1 else {}
                    new_extras = extras.copy() if isinstance(extras, dict) else extras

                    if isinstance(cond_tensor, torch.Tensor):
                        transformed_tensor, indirect_applied, removed_indices = apply_statistical_style_fidelity(
                            cond_tensor=cond_tensor,
                            spans_info=spans_info
                        )

                        # Metadata repair for attention_mask if indirect rows were removed
                        if indirect_applied and isinstance(new_extras, dict):
                            att_mask = new_extras.get("attention_mask")
                            if isinstance(att_mask, torch.Tensor) and att_mask.shape[-1] == cond_tensor.shape[1]:
                                keep = torch.ones(cond_tensor.shape[1], dtype=torch.bool, device=att_mask.device)
                                keep[removed_indices] = False
                                new_extras["attention_mask"] = att_mask[..., keep]
                            else:
                                new_extras.pop("attention_mask", None)

                        new_conditioning.append([transformed_tensor, new_extras])
                    else:
                        new_conditioning.append(list(cond_entry))
                else:
                    new_conditioning.append(cond_entry)
            conditioning = new_conditioning

    return EncodedQwenContext(
        tokens=tokens,
        conditioning=conditioning,
        physical_images=physical_images,
        physical_image_map=physical_image_map,
        vision_row_spans=vision_row_spans,
        warnings=warnings_list,
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
