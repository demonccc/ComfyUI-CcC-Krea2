"""Validation routines for CcC Krea2 model and input parameters."""

from typing import Any, Tuple
from .constants import LOGGER_PREFIX


class Krea2ModelValidationError(ValueError):
    """Raised when an incoming model is incompatible with the CcC Krea2 patch."""
    pass


def validate_krea2_model(model: Any, node_name: str) -> None:
    """Validate that the given ComfyUI MODEL object is compatible with Krea 2.

    If incompatible, raises Krea2ModelValidationError with a clear, descriptive
    message containing:
    - Node name
    - Detected model class
    - Expected model family
    - Concise suggestion
    """
    if model is None:
        raise Krea2ModelValidationError(
            f"{LOGGER_PREFIX} Error in node '{node_name}': MODEL input is None. "
            "Please connect a valid Krea 2 model."
        )

    # Check for model object structure
    model_obj = getattr(model, "model", None)
    if model_obj is None:
        raise Krea2ModelValidationError(
            f"{LOGGER_PREFIX} Error in node '{node_name}': Received invalid model object of type "
            f"'{type(model).__name__}'. Expected a ComfyUI ModelPatcher instance wrapping a Krea 2 model."
        )

    diffusion_model = getattr(model_obj, "diffusion_model", None)
    model_class_name = type(model_obj).__name__
    diff_class_name = type(diffusion_model).__name__ if diffusion_model is not None else "None"

    # Check if the model is a Krea 2 model family (SingleStreamDiT / Krea2)
    # ComfyUI Krea2 models typically use SingleStreamDiT or Krea2 model_base
    is_krea2 = False
    if "Krea2" in model_class_name or "SingleStreamDiT" in diff_class_name:
        is_krea2 = True
    elif hasattr(diffusion_model, "pe_embedder") and hasattr(diffusion_model, "txtfusion"):
        is_krea2 = True
    elif hasattr(model_obj, "latent_format") and "Krea2" in type(model_obj.latent_format).__name__:
        is_krea2 = True

    if not is_krea2:
        raise Krea2ModelValidationError(
            f"{LOGGER_PREFIX} Error in node '{node_name}': Incompatible model architecture detected.\n"
            f"- Detected model class: {model_class_name} (diffusion_model: {diff_class_name})\n"
            f"- Expected model family: Krea 2 (SingleStreamDiT / Krea2)\n"
            f"- Suggestion: Ensure you are loading a Krea 2 diffusion model checkpoint with the appropriate UNETLoader/CheckpointLoader."
        )


def align_dimensions(width: int, height: int, Multiple: int = 16) -> Tuple[int, int]:
    """Align width and height to the nearest multiple required by Krea 2 latents."""
    aligned_w = max(Multiple, (width // Multiple) * Multiple)
    aligned_h = max(Multiple, (height // Multiple) * Multiple)
    return aligned_w, aligned_h
