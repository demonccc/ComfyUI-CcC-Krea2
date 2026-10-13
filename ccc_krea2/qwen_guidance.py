"""Shared Qwen Vision guidance generation presets."""

MAX_GUIDANCE_SEED = 0xFFFFFFFFFFFFFFFF

GUIDANCE_SEED_MODES = ("fixed", "incremental", "random")
GUIDANCE_DETAIL_MODES = ("compact", "balanced", "detailed")
VISION_GUIDANCE_MODES = ("conservative", "balanced", "flexible")

GUIDANCE_DETAIL_PRESETS = {
    "compact": {
        "max_length": 256,
        "instruction": (
            "DETAIL POLICY: compact. Keep only the highest-impact facts that belong to the selected Vision mode. "
            "Compress aggressively, but preserve concrete specifics instead of replacing them with vague summaries."
        ),
    },
    "balanced": {
        "max_length": 512,
        "instruction": (
            "DETAIL POLICY: balanced. Capture all major and secondary observable facts needed to reproduce the selected "
            "Vision mode faithfully. Do not stop at a generic summary when pose, gaze, composition, colors, materials, "
            "rendering traits, or spatial relationships are relevant to that mode."
        ),
    },
    "detailed": {
        "max_length": 1024,
        "instruction": (
            "DETAIL POLICY: detailed. Be exhaustive within the selected Vision mode. Include fine observable details, "
            "secondary relationships, exact spatial cues, subtle visual traits, and small but reproducible distinctions. "
            "Do not invent hidden information and do not cross into another Vision mode."
        ),
    },
}

VISION_GUIDANCE_PRESETS = {
    "conservative": {
        "temperature": 0.05,
        "top_p": 0.70,
        "top_k": 32,
        "min_p": 0.10,
    },
    "balanced": {
        "temperature": 0.25,
        "top_p": 0.90,
        "top_k": 64,
        "min_p": 0.05,
    },
    "flexible": {
        "temperature": 0.70,
        "top_p": 0.99,
        "top_k": 128,
        "min_p": 0.01,
    },
}


def guidance_detail_instruction(guidance_detail: str = "balanced") -> str:
    """Return the extraction-depth instruction for a Guidance Detail preset."""
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
        "top_k": int(vision["top_k"]),
        "top_p": float(vision["top_p"]),
        "min_p": float(vision["min_p"]),
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": int(guidance_seed),
        "mtp": True,
    }
