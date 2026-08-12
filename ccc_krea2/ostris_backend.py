"""Ostris Edit backend implementation for CcC Krea2 suite.

Based on Ostris's ai-toolkit edit implementation.
Original work Copyright (c) Ostris / ai-toolkit contributors.
Licensed under the MIT License.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange

from .references import PreparedReference, _process_latent_in_if_available
from .patch import is_model_already_patched, _pad_to_patch_size, _repeat_to_batch_size, _timestep_embedding
from .geometry import resize_tensor


# Ostris vision pixel budget (~384x384 = 147456 pixels)
OSTRIS_VISION_PIXEL_BUDGET = 384 * 384
# Ostris VAE latent max pixel budget (1024x1024 = 1048576 pixels)
OSTRIS_VAE_MAX_PIXELS = 1024 * 1024


def preprocess_ostris_vision_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess vision image for Ostris backend: area-constrained to ~384x384, never upscaled.

    Uses AREA downscaling. NO /16 snapping is performed on vision input images.
    """
    if image_tensor is None:
        return image_tensor

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    bs, h, w, c = image_tensor.shape
    curr_area = h * w

    if curr_area > OSTRIS_VISION_PIXEL_BUDGET:
        scale = math.sqrt(OSTRIS_VISION_PIXEL_BUDGET / float(curr_area))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = resize_tensor(image_tensor, target_h=new_h, target_w=new_w, method="area")
        return torch.clamp(resized, 0.0, 1.0)

    return image_tensor


def preprocess_ostris_ref_pixel_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess pixel image for Ostris VAE reference encoding: max 1MP (1024x1024), /16 snapped."""
    if image_tensor is None:
        return image_tensor

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    bs, h, w, c = image_tensor.shape
    curr_area = h * w

    target_h, target_w = h, w
    if curr_area > OSTRIS_VAE_MAX_PIXELS:
        scale = math.sqrt(OSTRIS_VAE_MAX_PIXELS / float(curr_area))
        target_h = int(round(h * scale))
        target_w = int(round(w * scale))

    snapped_h = max(16, int(round(target_h / 16.0)) * 16)
    snapped_w = max(16, int(round(target_w / 16.0)) * 16)

    if (snapped_h, snapped_w) != (h, w):
        method = "area" if curr_area > OSTRIS_VAE_MAX_PIXELS else "auto"
        resized = resize_tensor(image_tensor, target_h=snapped_h, target_w=snapped_w, method=method)
        return torch.clamp(resized, 0.0, 1.0)

    return image_tensor


def build_ostris_qwen_prompt(
    resolved_references: List[Dict[str, Any]],
    user_prompt: str = "",
) -> str:
    """Format Qwen text prompt for Ostris backend using 'Picture 1: <vision block>' style layout."""
    lines = []
    for idx, item in enumerate(resolved_references, start=1):
        spec = item.get("spec")
        expanded_aliases = item.get("expanded_aliases", ())
        alias_str = ", ".join(expanded_aliases) if expanded_aliases else (getattr(spec, "alias", "") if spec else "")
        instruction = getattr(spec, "vision_instruction", "") if spec else ""

        annotation = ""
        if alias_str and instruction:
            annotation = f" ({alias_str}): {instruction}"
        elif alias_str:
            annotation = f" ({alias_str})"
        elif instruction:
            annotation = f": {instruction}"

        lines.append(f"Picture {idx}: <|vision_start|><|image_pad|><|vision_end|>{annotation}")

    body = "\n".join(lines)
    if body and user_prompt:
        return f"{body}\n\n{user_prompt}"
    return body or (user_prompt or "")


def patch_ostris_model(
    model: Any,
    prepared_refs: List[PreparedReference],
    ostris_kv_cache: bool = False
) -> Any:
    """Clone MODEL and register canonical Ostris DIFFUSION_MODEL wrapper.

    Contract: patch_ostris_model(model, prepared_refs, ostris_kv_cache=False)
    """
    if ostris_kv_cache:
        raise NotImplementedError(
            "[CcC Krea2] ostris_kv_cache=True is not safely supported in the current ComfyUI runtime environment. "
            "Set ostris_kv_cache=False to proceed."
        )

    if is_model_already_patched(model, "ccc_ostris_edit"):
        return model

    patched_model = model.clone()
    patched_model._ccc_patch_key = "ccc_ostris_edit"

    processed_ref_latents: List[torch.Tensor] = []

    for ref in prepared_refs:
        if ref.vae_latent is not None:
            proc_lat = _process_latent_in_if_available(patched_model, ref.vae_latent)
            processed_ref_latents.append(proc_lat)

    def ostris_edit_wrapper(executor: Any, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, *wargs: Any, **kwargs: Any) -> torch.Tensor:
        """Ostris DIFFUSION_MODEL wrapper signature."""
        dit_model = getattr(executor, "class_obj", None)

        transformer_options = {}
        if wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        elif "transformer_options" in kwargs:
            transformer_options = kwargs["transformer_options"]

        if not processed_ref_latents or dit_model is None:
            return executor(x, timesteps, context, *wargs, **kwargs)

        return ostris_dit_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=processed_ref_latents,
            ostris_kv_cache=ostris_kv_cache,
            transformer_options=transformer_options,
        )

    _register_ostris_wrapper(patched_model, ostris_edit_wrapper)
    return patched_model


def _register_ostris_wrapper(patched_model: Any, wrapper: Any) -> None:
    """Register Ostris wrapper using ComfyUI wrapper registration."""
    registered = False
    wrapper_type = "diffusion_model"
    try:
        import comfy.patcher_extension
        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:
        wrapper_type = "diffusion_model"

    if hasattr(patched_model, "add_wrapper_with_key"):
        try:
            patched_model.add_wrapper_with_key(wrapper_type, "ccc_ostris_edit", wrapper)
            registered = True
        except Exception:
            pass

    if not registered:
        if not hasattr(patched_model, "model_options"):
            patched_model.model_options = {}
        t_options = patched_model.model_options.setdefault("transformer_options", {})
        wrappers = t_options.setdefault("wrappers", {})
        w_key = str(wrapper_type)
        if isinstance(wrappers, dict):
            diff_wrappers = wrappers.setdefault(w_key, {})
            if isinstance(diff_wrappers, dict):
                diff_wrappers["ccc_ostris_edit"] = [wrapper]


def ostris_dit_forward(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ostris_kv_cache: bool,
    transformer_options: Dict[str, Any],
) -> torch.Tensor:
    """Execute Ostris DiT edit forward pass: timestep 0 for references, target frame 0, refs frames 1..N."""
    orig_ndim = x.ndim
    if orig_ndim == 5:
        b_orig, c_orig, t_orig, h_orig, w_orig = x.shape
        x_4d = rearrange(x, "b c t h w -> (b t) c h w")
    else:
        x_4d = x
        t_orig = 1

    bs, c, target_h, target_w = x_4d.shape
    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    channels = getattr(dit_model, "channels", c)
    orig_tgt_h, orig_tgt_w = target_h, target_w

    x_padded = _pad_to_patch_size(x_4d, patch_size)
    padded_h, padded_w = x_padded.shape[-2], x_padded.shape[-1]

    target_gh = padded_h // patch_size
    target_gw = padded_w // patch_size
    tgt_n_toks = target_gh * target_gw

    x_patch = rearrange(x_padded, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size)

    ref_patches: List[torch.Tensor] = []
    ref_token_grids: List[Tuple[int, int]] = []
    ref_token_lens: List[int] = []

    for ref_lat in ref_latents:
        if ref_lat.ndim == 5:
            ref_lat = rearrange(ref_lat, "b c t h w -> (b t) c h w")

        ref_lat = ref_lat.to(device=x.device, dtype=x.dtype)
        ref_lat_b = _repeat_to_batch_size(ref_lat, bs)
        ref_padded = _pad_to_patch_size(ref_lat_b, patch_size)

        r_ph, r_pw = ref_padded.shape[-2], ref_padded.shape[-1]
        r_gh = r_ph // patch_size
        r_gw = r_pw // patch_size

        r_patch = rearrange(ref_padded, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size)
        ref_patches.append(r_patch)
        ref_token_grids.append((r_gh, r_gw))
        ref_token_lens.append(r_gh * r_gw)

    ctx = dit_model._unpack_context(context) if hasattr(dit_model, "_unpack_context") else context
    if hasattr(dit_model, "txtfusion"):
        ctx = dit_model.txtfusion(ctx, mask=None, transformer_options=transformer_options)
    if hasattr(dit_model, "txtmlp"):
        ctx = dit_model.txtmlp(ctx)

    txt_len = ctx.shape[1] if ctx is not None else 0

    target_emb = dit_model.first(x_patch) if hasattr(dit_model, "first") else x_patch
    ref_embs = [dit_model.first(rp) if hasattr(dit_model, "first") else rp for rp in ref_patches]

    seq_components = []
    if ctx is not None:
        seq_components.append(ctx)
    seq_components.extend(ref_embs)
    seq_components.append(target_emb)

    full_seq = torch.cat(seq_components, dim=1)

    # 3D RoPE position IDs: references have frame indices 1..N, target has frame index 0
    rope_pos_ids = _build_ostris_3d_rope_pos_ids(
        batch_size=bs,
        txt_len=txt_len,
        ref_token_grids=ref_token_grids,
        target_grid=(target_gh, target_gw),
        device=x.device
    )

    freqs = dit_model.pe_embedder(rope_pos_ids) if hasattr(dit_model, "pe_embedder") else None

    # Timestep embeddings: target uses timesteps, reference tokens use timestep 0
    tdim = getattr(dit_model, "tdim", 256)
    t_target = _timestep_embedding(timesteps, tdim).unsqueeze(1).to(x.dtype)

    t = dit_model.tmlp(t_target) if hasattr(dit_model, "tmlp") else t_target
    tvec = dit_model.tproj(t) if hasattr(dit_model, "tproj") else t

    h_seq = full_seq
    blocks = getattr(dit_model, "blocks", [])
    total_blocks = len(blocks)
    total_ref_len = sum(ref_token_lens)

    for i, block in enumerate(blocks):
        t_opts = transformer_options.copy()
        t_opts["total_blocks"] = total_blocks
        t_opts["block_type"] = "single"
        t_opts["img_slice"] = [slice(txt_len + total_ref_len, None)]
        t_opts["block_index"] = i
        t_opts["ostris_kv_cache"] = ostris_kv_cache

        h_seq = block(
            h_seq,
            tvec,
            freqs,
            attn_bias=None,
            transformer_options=t_opts
        )

    out_seq = dit_model.last(h_seq, t) if hasattr(dit_model, "last") else h_seq

    # Slice target tokens only (last tgt_n_toks)
    tgt_tokens = out_seq[:, -tgt_n_toks:, :]

    out_4d = rearrange(
        tgt_tokens,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=target_gh,
        w=target_gw,
        p1=patch_size,
        p2=patch_size,
        c=channels
    )

    out_cropped = out_4d[:, :, :orig_tgt_h, :orig_tgt_w]

    if orig_ndim == 5:
        return rearrange(out_cropped, "(b t) c h w -> b c t h w", t=t_orig)

    return out_cropped


def _build_ostris_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device
) -> torch.Tensor:
    """Build 3D RoPE position IDs for Ostris: target is frame 0, references are frames 1..N."""
    tgt_gh, tgt_gw = target_grid
    list_pos = []

    if txt_len > 0:
        txt_pos = torch.zeros((txt_len, 3), device=device, dtype=torch.float32)
        list_pos.append(txt_pos)

    for i, (r_gh, r_gw) in enumerate(ref_token_grids):
        frame_idx = i + 1
        grid_y = torch.arange(r_gh, device=device, dtype=torch.float32)
        grid_x = torch.arange(r_gw, device=device, dtype=torch.float32)

        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        mesh_t = torch.full_like(mesh_y, fill_value=float(frame_idx))

        ref_pos = torch.stack([mesh_t.flatten(), mesh_y.flatten(), mesh_x.flatten()], dim=-1)
        list_pos.append(ref_pos)

    tgt_y = torch.arange(tgt_gh, device=device, dtype=torch.float32)
    tgt_x = torch.arange(tgt_gw, device=device, dtype=torch.float32)
    mesh_ty, mesh_tx = torch.meshgrid(tgt_y, tgt_x, indexing="ij")
    mesh_tt = torch.zeros_like(mesh_ty)

    tgt_pos = torch.stack([mesh_tt.flatten(), mesh_ty.flatten(), mesh_tx.flatten()], dim=-1)
    list_pos.append(tgt_pos)

    seq_pos = torch.cat(list_pos, dim=0)
    return seq_pos.unsqueeze(0).repeat(batch_size, 1, 1)
