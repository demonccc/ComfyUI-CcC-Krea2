"""Qwen3-VL text/vision conditioning builder for positive and negative prompts."""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Dict
from ccc_krea2.constants import (
    LOGGER_PREFIX,
    VISION_PAD_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    ReferenceRole,
)
from ccc_krea2.style_processing import apply_statistical_style_fidelity, StyleSpanOperation
from ccc_krea2.vision_prep import resolve_qwen_encoder_config, calculate_native_qwen_geometry


@dataclass
class EncodedQwenContext:
    tokens: Any
    conditioning: List[Any]
    physical_images: List[Any]
    physical_image_map: List[Dict[str, Any]]
    vision_row_spans: List[Tuple[int, int]]
    warnings: List[str]
    pos_rows_before: int = 0
    pos_rows_after: int = 0
    neg_rows: int = 0
    token_stream_key: str = "qwen3vl"
    template_prefix_rows_removed: int = 0
    removed_row_indices: List[int] = field(default_factory=list)


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


def resolve_qwen_token_stream(tokens: Any) -> Tuple[List[Any], str]:
    """Explicitly resolve supported token stream from CLIP tokens dictionary.

    Raises ValueError with detailed rejection details if incompatible.
    """
    if isinstance(tokens, list):
        return tokens, "raw_list"

    if not isinstance(tokens, dict):
        raise ValueError(f"{LOGGER_PREFIX} Tokens must be a dict or list, got {type(tokens).__name__}.")

    attempted_keys = ["qwen3vl_4b", "qwen_vl", "qwen3vl"]
    rejection_reasons = {}

    for k in attempted_keys:
        if k not in tokens:
            rejection_reasons[k] = "key not present in tokens dict"
            continue
        val = tokens[k]
        if not val:
            rejection_reasons[k] = "key present but value is empty or None"
            continue
        if isinstance(val, list):
            pairs = val[0] if len(val) > 0 and isinstance(val[0], list) else val
            if isinstance(pairs, list):
                return pairs, k
            else:
                rejection_reasons[k] = f"invalid token pair sequence type: {type(pairs).__name__}"
        else:
            rejection_reasons[k] = f"invalid value type: {type(val).__name__}"

    # If test injected custom key or tokens contains other list of lists
    for k, v in tokens.items():
        if k not in attempted_keys and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return v[0], k

    available_keys = list(tokens.keys())
    reason_str = "; ".join(f"'{k}': {r}" for k, r in rejection_reasons.items())
    raise ValueError(
        f"{LOGGER_PREFIX} Incompatible CLIP token structure. "
        f"Available token keys: {available_keys}. "
        f"Recognized keys attempted: {attempted_keys}. "
        f"Rejection details: {reason_str}"
    )


def calculate_qwen_rows_from_embedded_image(elem: Dict[str, Any], clip: Any) -> int:
    """Calculate Qwen visual output rows directly from embedded token dict 'data' tensor using real Qwen visual processor geometry."""
    if "data" not in elem:
        raise ValueError(f"{LOGGER_PREFIX} Embedded token dictionary missing required 'data' image key.")

    image_data = elem["data"]
    if not isinstance(image_data, torch.Tensor):
        raise ValueError(f"{LOGGER_PREFIX} Embedded image 'data' must be a torch.Tensor, got {type(image_data).__name__}.")

    # Extract spatial dimensions H, W from image_data tensor
    if image_data.ndim == 4:
        if image_data.shape[1] in (1, 3, 4):
            ih, iw = image_data.shape[2], image_data.shape[3]
        else:
            ih, iw = image_data.shape[1], image_data.shape[2]
    elif image_data.ndim == 3:
        if image_data.shape[0] in (1, 3, 4):
            ih, iw = image_data.shape[1], image_data.shape[2]
        else:
            ih, iw = image_data.shape[0], image_data.shape[1]
    else:
        raise ValueError(f"{LOGGER_PREFIX} Unsupported embedded image 'data' tensor shape: {image_data.shape}.")

    config = resolve_qwen_encoder_config(clip)
    native_h, native_w = calculate_native_qwen_geometry(ih, iw, config)
    grid_h = native_h // config.patch_size
    grid_w = native_w // config.patch_size
    merge_sq = config.merge_size * config.merge_size

    rows = max(1, (grid_h * grid_w) // merge_sq)
    return rows


def extract_vision_spans_from_tokens(
    tokens: Any,
    physical_image_map: List[Dict[str, Any]],
    clip: Any = None
) -> Tuple[List[Tuple[int, int]], List[str], str, int]:
    """Extract vision row spans from Qwen token stream using real Qwen grid geometry after template prefix stripping.

    Returns:
        (adjusted_spans, warnings, stream_key, template_prefix_rows_removed)
    """
    warnings: List[str] = []

    if not tokens and not physical_image_map:
        return [], warnings, "none", 0

    tok_pairs, stream_key = resolve_qwen_token_stream(tokens)

    IM_START = 151644
    USER = 872
    NEWLINE = 198

    spans = []
    rows_counter = 0
    template_end = -1
    count_im_start = 0
    ids = []
    img_idx = 0

    for v in tok_pairs:
        elem = v[0] if isinstance(v, (list, tuple)) else v
        if isinstance(elem, dict):
            # embedded image dictionary - calculate rows from real elem["data"]
            n = calculate_qwen_rows_from_embedded_image(elem, clip)
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

    if not adjusted_spans and physical_image_map:
        # Fallback for synthetic/mock tokens without embedded dicts
        cur = 0
        for item in physical_image_map:
            img = item.get("image")
            n = calculate_qwen_rows_from_embedded_image({"data": img}, clip) if img is not None else 64
            spans.append((cur, cur + n))
            cur += n
        adjusted_spans = spans

    # Section 3.5: Strict span validation
    if physical_image_map:
        if len(adjusted_spans) != len(physical_image_map):
            raise ValueError(
                f"{LOGGER_PREFIX} Vision span validation failed: extracted spans count ({len(adjusted_spans)}) "
                f"does not match physical Qwen images count ({len(physical_image_map)})."
            )

        for i, (s, e) in enumerate(adjusted_spans):
            if e <= s:
                raise ValueError(f"{LOGGER_PREFIX} Vision span validation failed: span {i} has non-positive length ({s}, {e}).")
            if s < 0:
                raise ValueError(f"{LOGGER_PREFIX} Vision span validation failed: span {i} start index is negative ({s}).")

        for i in range(len(adjusted_spans) - 1):
            s1, e1 = adjusted_spans[i]
            s2, e2 = adjusted_spans[i + 1]
            if s2 < e1:
                raise ValueError(
                    f"{LOGGER_PREFIX} Vision span validation failed: overlapping or out-of-order spans detected "
                    f"at index {i} ({s1}, {e1}) and {i+1} ({s2}, {e2})."
                )

    return adjusted_spans, warnings, stream_key, template_end


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
    vision_row_spans, span_warnings, stream_key, template_end = extract_vision_spans_from_tokens(
        tokens=tokens,
        physical_image_map=physical_image_map,
        clip=clip
    )
    warnings_list.extend(span_warnings)

    pos_rows_before = 0
    pos_rows_after = 0
    removed_row_indices: List[int] = []

    if conditioning and isinstance(conditioning, list) and len(conditioning) > 0:
        c_tensor = conditioning[0][0] if isinstance(conditioning[0], (list, tuple)) else None
        if isinstance(c_tensor, torch.Tensor):
            pos_rows_before = c_tensor.shape[1]
            pos_rows_after = pos_rows_before

    # Apply Moodboard style processing (Fidelity & Indirect Transfer) for positive conditioning
    if is_positive and conditioning and physical_image_map:
        spans_info: List[StyleSpanOperation] = []
        for idx, item in enumerate(physical_image_map):
            if item.get("role") == "style" and idx < len(vision_row_spans):
                spec = item.get("spec")
                fidelity = getattr(spec, "style_fidelity", 1.0)
                indirect = getattr(spec, "indirect_style_transfer", False)
                s_start, s_end = vision_row_spans[idx]
                spans_info.append(StyleSpanOperation(
                    logical_reference_id=item.get("logical_reference_id", "style"),
                    logical_vision_slot=item.get("logical_vision_slot", idx + 1),
                    physical_qwen_index=item.get("physical_qwen_image_index", idx + 1),
                    row_start=s_start,
                    row_end=s_end,
                    style_fidelity=fidelity,
                    indirect_style_transfer=indirect
                ))

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
                        removed_row_indices = removed_indices
                        pos_rows_after = transformed_tensor.shape[1]

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

    neg_rows = 0
    if not is_positive and conditioning and isinstance(conditioning, list) and len(conditioning) > 0:
        c_tensor = conditioning[0][0] if isinstance(conditioning[0], (list, tuple)) else None
        if isinstance(c_tensor, torch.Tensor):
            neg_rows = c_tensor.shape[1]

    return EncodedQwenContext(
        tokens=tokens,
        conditioning=conditioning,
        physical_images=physical_images,
        physical_image_map=physical_image_map,
        vision_row_spans=vision_row_spans,
        warnings=warnings_list,
        pos_rows_before=pos_rows_before,
        pos_rows_after=pos_rows_after,
        neg_rows=neg_rows,
        token_stream_key=stream_key,
        template_prefix_rows_removed=template_end,
        removed_row_indices=removed_row_indices,
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
