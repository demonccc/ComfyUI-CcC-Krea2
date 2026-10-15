"""Optional Krea2 Qwen Vision token weighting through ComfyUI's attn1_patch hook.

This changes only Qwen-derived conditioning rows, not VAE reference latents or their
visual_boost. Kept separate from the CcC in-context appearance runtime.
"""
import torch

def install_vision_attention(model, conditioning):
    if not conditioning:
        return model, 0
    meta = conditioning[0][1] if len(conditioning[0]) > 1 else {}
    spans = meta.get("ccc_vision_weight_spans", ())
    if not spans:
        return model, 0
    positions = {}
    for start, end, strength in spans:
        for i in range(int(start), int(end)):
            positions[i] = float(strength)
    positions = {i:v for i,v in positions.items() if abs(v-1.0)>1e-6}
    if not positions:
        return model, 0
    indexes = torch.tensor(sorted(positions),dtype=torch.long)
    factors = torch.tensor([positions[i] for i in sorted(positions)],dtype=torch.float32)
    clone = model.clone()

    def ccc_vision_token_weight(q, k, v, pe=None, attn_mask=None, extra_options=None):
        options = extra_options or {}
        # Weight positive conditioning only, never unconditional batches.
        cou = options.get("cond_or_uncond")
        batch = v.shape[0]
        if not cou:
            rows = None
        elif batch % len(cou) == 0:
            rows = torch.tensor([x==0 for x in cou], device=v.device).repeat_interleave(batch//len(cou))
        else:
            # Unknown CFG batching layout: fail safely rather than weight uncond.
            return {}
        ix = indexes.to(v.device)
        ok = ix < v.shape[2]
        ix = ix[ok]
        if not ix.numel():
            return {}
        strength = factors.to(v.device,dtype=v.dtype)[ok]
        scale = torch.ones((v.shape[2],),device=v.device,dtype=v.dtype)
        scale[ix] = strength
        scale = scale.reshape(1,1,-1,1)
        if rows is not None:
            scale = torch.where(rows.reshape(batch,1,1,1),scale,torch.ones_like(scale))
        return {"v":v*scale}
    clone.set_model_attn1_patch(ccc_vision_token_weight)
    return clone,len(positions)
