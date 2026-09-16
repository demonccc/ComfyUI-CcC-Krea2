"""Orchestrated Krea2 Edit MODEL wrapper.

This restores the original CcC transport contract: prepared appearance references are
captured by the MODEL wrapper produced by the edit orchestrator. Conditioning remains
responsible for Qwen semantic context; appearance latents, boosts and RoPE placement stay
with the patched MODEL for the sampling run.
"""

from typing import Any, List

import torch

from .references import PreparedReference, _process_latent_in_if_available
from .patch import _register_wrapper, is_model_already_patched, krea2_dit_incontext_forward


def patch_krea2_orchestrated_model(model: Any, prepared_refs: List[PreparedReference] | None = None) -> Any:
    """Clone MODEL and capture the orchestrator-prepared appearance references in its wrapper."""
    if is_model_already_patched(model, "ccc_krea2_edit"):
        raise RuntimeError(
            "[CcC Krea2] Input MODEL is already patched by CcC Krea2 Edit. "
            "Connect Edit to the unpatched upstream MODEL."
        )

    patched_model = model.clone()
    patched_model._ccc_patch_key = "ccc_krea2_edit"

    processed_ref_latents: list[torch.Tensor] = []
    ref_boosts: list[float] = []
    ref_fit: list[bool] = []
    ref_rope_positions: list[str] = []

    for ref in prepared_refs or []:
        if ref.vae_latent is None:
            continue
        processed = _process_latent_in_if_available(patched_model, ref.vae_latent)
        processed_ref_latents.append(processed)
        ref_boosts.append(float(ref.boost))
        # The orchestrator already applied the Identity Edit pixel-space fit.
        # Keep each reference on its own fitted latent grid in the DiT.
        ref_fit.append(True)
        ref_rope_positions.append(str(getattr(ref, "rope_position", "none")))

    patched_model._ccc_orchestrated_reference_count = len(processed_ref_latents)
    patched_model._ccc_orchestrated_reference_shapes = [
        tuple(int(v) for v in latent.shape) for latent in processed_ref_latents
    ]
    patched_model._ccc_orchestrated_reference_boosts = list(ref_boosts)
    patched_model._ccc_orchestrated_reference_rope = list(ref_rope_positions)

    def krea2_edit_wrapper(
        executor: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *wargs: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        dit_model = getattr(executor, "class_obj", None)
        if dit_model is None or not processed_ref_latents:
            return executor(x, timesteps, context, *wargs, **kwargs)

        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None and wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        if transformer_options is None:
            transformer_options = {}

        return krea2_dit_incontext_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=processed_ref_latents,
            ref_boosts=ref_boosts,
            ref_fit=ref_fit,
            ref_rope_positions=ref_rope_positions,
            transformer_options=transformer_options,
        )

    _register_wrapper(patched_model, krea2_edit_wrapper)
    return patched_model
