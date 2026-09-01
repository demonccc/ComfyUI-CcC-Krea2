# Krea2 CcC Nodes

## Krea2 CcC Visual Reference

Declares one ordered visual reference for Krea2 Edit.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | `IMAGE` | — | Visual edit reference. |
| `boost` | `0.0 .. 10.0` | `1.0` | Per-reference attention boost. |
| `rope_position` | `none`, `up`, `down`, `left`, `right` | `none` | Per-reference RoPE position. |
| `semantic` | boolean | `true` | Also participates in Qwen Vision. |
| `semantic_role` | text | empty | Optional semantic name for this image. |
| `instruction` | multiline text | empty | Per-reference Qwen instruction. |
| `grounding_px` | `0 .. 4096` | `768` | Qwen grounding size. |
| `previous_references` | visual reference chain | optional | Appends after the previous visual reference. |

Rules:

- `semantic = false`: `semantic_role`, `instruction`, and `grounding_px` are disabled and ignored.
- `semantic = true` + empty `semantic_role`: positional Krea2 Edit behavior.
- `semantic = true` + non-empty `semantic_role`: the role text is attached to the corresponding Qwen image.

## Krea2 CcC Semantic Reference

Declares one semantic/style reference.

| Control | Values | Default |
| --- | --- | --- |
| `image` | `IMAGE` | — |
| `mode` | `semantic_only`, `style_direct`, `style_indirect` | `semantic_only` |
| `instruction` | multiline text | empty |
| `grounding_px` | `0 .. 4096` | `768` |
| `processing` | `full`, `2x2`, `4x4` | `2x2` |
| `fidelity` | `0.0 .. 1.0` | `1.0` |
| `previous_references` | semantic reference chain | optional |

## Krea2 CcC Latent

Builds the target latent before Edit.

### Required controls

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `vae` | `VAE` | — | Encodes `target_image` when present. |
| `aspect_ratio` | `from source`, `1:1`, `3:2`, `2:3`, `4:3`, `3:4`, `16:9`, `9:16` | `1:1` | Target aspect ratio. |
| `resolution` | `from source`, `0.5 MP` ... `4.0 MP` | `2.0 MP` | Target pixel budget. |
| `latent_semantic` | boolean | `false` | Reuses the target image as semantic/Qwen context so the target content can be reimagined. |
| `latent_semantic_instruction` | multiline text | empty | Instruction for the target semantic reference. |
| `latent_grounding_px` | `0 .. 4096` | `768` | Grounding size for latent semantic context. |
| `batch_size` | `1 .. 64` | `1` | Latent batch size. |

### Optional image sockets

| Socket | Behavior |
| --- | --- |
| `target_image` | Connected = VAE image-init latent. Unconnected = empty latent. |
| `grid_size_image` | Supplies pixel budget when `resolution = from source`; otherwise falls back to `target_image`. |
| `grid_geometry_image` | Supplies aspect ratio when `aspect_ratio = from source`; otherwise falls back to `target_image`. |

The latent carries its semantic configuration as internal metadata so the Edit node can preserve the same ordering and behavior that existed before the split.

## Krea2 CcC Edit

Consumes a pre-built latent and executes Krea2 Edit.

### Required

- `model`
- `clip`
- `vae`
- `latent`
- `positive_prompt`
- `negative_prompt`
- `apply_krea2_edit_patch`

### Optional

- `visual_references`
- `semantic_references`

Final ordering:

```text
visual references -> latent semantic (when enabled) -> semantic-only references -> style references
```
