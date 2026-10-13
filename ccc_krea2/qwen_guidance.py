"""Shared Qwen Vision guidance generation presets."""

MAX_GUIDANCE_SEED = 0xFFFFFFFFFFFFFFFF

GUIDANCE_SEED_MODES = ("fixed", "incremental", "random")
GUIDANCE_DETAIL_MODES = ("compact", "balanced", "detailed")
VISION_GUIDANCE_MODES = ("conservative", "balanced", "flexible")

GUIDANCE_DETAIL_PRESETS = {
    "compact": {"max_length": 256},
    "balanced": {"max_length": 512},
    "detailed": {"max_length": 1024},
}

VISION_GUIDANCE_PRESETS = {
    "conservative": {"temperature": 0.15, "top_p": 0.85},
    "balanced": {"temperature": 0.25, "top_p": 0.90},
    "flexible": {"temperature": 0.45, "top_p": 0.95},
}


def guidance_generation_kwargs(
    *,
    guidance_seed: int,
    guidance_detail: str = "balanced",
    vision_guidance: str = "balanced",
):
    """Build the fixed Qwen sampling kwargs shared by all guidance consumers."""
    try:
        detail = GUIDANCE_DETAIL_PRESETS[guidance_detail]
    except KeyError as exc:
        raise ValueError(
            f"[Krea2 CcC] Unsupported Guidance Detail preset: {guidance_detail!r}."
        ) from exc

    try:
        vision = VISION_GUIDANCE_PRESETS[vision_guidance]
    except KeyError as exc:
        raise ValueError(
            f"[Krea2 CcC] Unsupported Vision Guidance preset: {vision_guidance!r}."
        ) from exc

    return {
        "do_sample": True,
        "max_length": int(detail["max_length"]),
        "temperature": float(vision["temperature"]),
        "top_k": 64,
        "top_p": float(vision["top_p"]),
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": int(guidance_seed),
        "mtp": True,
    }
