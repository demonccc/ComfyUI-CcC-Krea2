"""Shared Qwen Vision guidance generation presets."""

MAX_GUIDANCE_SEED = 0xFFFFFFFFFFFFFFFF

GUIDANCE_SEED_MODES = ("fixed", "incremental", "random")
GUIDANCE_DETAIL_MODES = ("compact", "balanced", "detailed")
VISION_GUIDANCE_MODES = ("conservative", "balanced", "flexible")

GUIDANCE_DETAIL_PRESETS = {
    "compact": {
        "max_length": 256,
        "instruction": (
            "GUIDANCE DETAIL: compact. Use the smallest useful description while preserving the exact critical facts "
            "that belong to this Vision mode. Do not replace omitted details with guesses, and do not include attributes "
            "from another Vision mode."
        ),
    },
    "balanced": {
        "max_length": 512,
        "instruction": "",
    },
    "detailed": {
        "max_length": 1024,
        "instruction": (
            "GUIDANCE DETAIL: detailed. Use the larger token budget only to add more exact facts that belong to this "
            "Vision mode. Never add attributes from another Vision mode just to make the description richer. If a detail "
            "is outside the current mode, omit it rather than infer or transfer it."
        ),
    },
}

VISION_GUIDANCE_PRESETS = {
    "conservative": {"temperature": 0.15, "top_p": 0.85},
    "balanced": {"temperature": 0.25, "top_p": 0.90},
    "flexible": {"temperature": 0.45, "top_p": 0.95},
}


def guidance_detail_instruction(guidance_detail: str = "balanced") -> str:
    """Return the scope-safe prompt directive for a Guidance Detail preset."""
    try:
        return str(GUIDANCE_DETAIL_PRESETS[guidance_detail]["instruction"])
    except KeyError as exc:
        raise ValueError(
            f"[Krea2 CcC] Unsupported Guidance Detail preset: {guidance_detail!r}."
        ) from exc


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
