# Krea2 CcC Nodes

## Krea2 CcC Visual Reference

Declares one ordered visual reference for Krea2 Edit.

### Inputs

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | `IMAGE` | — | Visual edit reference. |
| `boost` | `0.0 .. 10.0` step `0.05` | `1.0` | Existing per-reference attention boost. |
| `rope_position` | `none`, `up`, `down`, `left`, `right` | `none` | Existing experimental per-reference RoPE position. |
| `semantic` | boolean | `true` | Controls whether the same visual reference also participates in Qwen Vision. |
| `semantic_role` | text | empty | Optional semantic name attached to this image in the Qwen prompt. |
| `instruction` | multiline text | empty | Existing per-reference Qwen instruction. |
| `grounding_px` | `0 .. 4096` step `16` | `768` | Existing Qwen grounding size. |
| `previous_references` | visual reference chain | optional | Chains this reference after the previous visual reference. |

### Semantic role rules

- `semantic = false`: `semantic_role`, `instruction`, and `grounding_px` are disabled and ignored.
- `semantic = true` + empty `semantic_role`: positional Krea2 Edit behavior; no extra semantic alias is attached.
- `semantic = true` + non-empty `semantic_role`: that exact text is attached to the corresponding Qwen vision block.

The chain order is also the Krea2 physical edit-reference order.

## Krea2 CcC Semantic Reference

Declares one semantic/style reference.

### Inputs

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | `IMAGE` | — | Source image for Qwen semantic/style conditioning. |
| `mode` | `semantic_only`, `style_direct`, `style_indirect` | `semantic_only` | Existing semantic/style mode. |
| `instruction` | multiline text | empty | Existing Qwen instruction. |
| `grounding_px` | `0 .. 4096` step `16` | `768` | Existing grounding size. |
| `processing` | `full`, `2x2`, `4x4` | `2x2` | Existing style processing mode. |
| `fidelity` | `0.0 .. 1.0` step `0.05` | `1.0` | Existing style fidelity. |
| `previous_references` | semantic reference chain | optional | Chains this reference after the previous semantic reference. |

`processing` and `fidelity` are consumed by the style modes. `semantic_only` stays on the Qwen semantic path.

## Krea2 CcC Edit

Consumes the ordered reference chains and executes Krea2 Edit.

### Required controls

| Control | Values | Default |
| --- | --- | --- |
| `model` | `MODEL` | — |
| `clip` | `CLIP` | — |
| `vae` | `VAE` | — |
| `positive_prompt` | multiline text | empty |
| `negative_prompt` | multiline text | empty |
| `aspect_ratio` | `from source`, `1:1`, `3:2`, `2:3`, `4:3`, `3:4`, `16:9`, `9:16` | `1:1` |
| `resolution` | `from source`, `0.5 MP`, `1.0 MP`, `1.5 MP`, `2.0 MP`, `2.5 MP`, `3.0 MP`, `4.0 MP` | `2.0 MP` |
| `latent_semantic` | boolean | `false` |
| `latent_semantic_instruction` | multiline text | empty |
| `latent_grounding_px` | `0 .. 4096` step `16` | `768` |
| `batch_size` | `1 .. 64` | `1` |
| `apply_krea2_edit_patch` | boolean | `true` |

### Optional sockets

| Socket | Type | Behavior |
| --- | --- | --- |
| `target_image` | `IMAGE` | Connected = image target latent. Unconnected = empty target latent. |
| `grid_size_image` | `IMAGE` | Supplies source pixel budget when `resolution = from source`. Falls back to `target_image`. |
| `grid_geometry_image` | `IMAGE` | Supplies source aspect ratio when `aspect_ratio = from source`. Falls back to `target_image`. |
| `visual_references` | visual reference chain | Ordered visual Krea2 Edit references. |
| `semantic_references` | semantic reference chain | Qwen semantic/style references. |

The Edit node always builds the final reference order as:

```text
visual references -> target semantic (when enabled) -> semantic-only references -> style references
```

This keeps style spans after edit references and preserves deterministic physical Qwen ordering.
