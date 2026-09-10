# Krea2 CcC Nodes

## Krea2 CcC Visual Reference

Declares one ordered visual reference for Krea2 Edit.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | `IMAGE` | — | Visual edit reference. |
| `boost` | `0.0 .. 10.0` | `1.0` | Per-reference attention boost. |
| `fit_to_latent` | boolean | `true` | `true` contains the reference inside target geometry; `false` preserves native scale except /16 VAE padding. |
| `rope_grid` | `inside`, `outside` | `inside` | Place RoPE coordinates inside or outside the target grid. |
| `rope_horizontal` | `center`, `left`, `right` | `center` | Horizontal RoPE alignment. |
| `rope_vertical` | `center`, `up`, `down` | `center` | Vertical RoPE alignment. |
| `semantic` | boolean | `true` | Also participates in Qwen Vision. |
| `semantic_role` | text | empty | Optional semantic name for this image. |
| `instruction` | multiline text | empty | Per-reference Qwen instruction. |
| `grounding_px` | `0 .. 4096` | `768` | Qwen grounding size. |

Rules:

- `semantic = false`: `semantic_role`, `instruction`, and `grounding_px` are disabled and ignored.
- `semantic = true` + empty `semantic_role`: positional Krea2 Edit behavior.
- `semantic = true` + non-empty `semantic_role`: the role text is attached to the corresponding Qwen image.

## Krea2 CcC Semantic Reference

Declares one Qwen-only semantic/style reference. Modes remain `semantic_only`, `style_direct`, and `style_indirect`.

## Krea2 CcC Size Resolver

Resolves a final width and height from two independent image measurements.

| Input | Behavior |
| --- | --- |
| `long_edge_image` | Supplies only `max(width, height)`. |
| `aspect_ratio_image` | Supplies only `width / height`. |

Outputs:

- `width` (`INT`)
- `height` (`INT`)

The output dimensions are not aligned. Latent performs final /16 alignment.

## Krea2 CcC Latent

Builds the target latent before Edit.

### Dimension controls

`dimensions`:

- `from_image`: use `dimensions_image` width and height.
- `fixed`: use `width` and `height`.
- `preset`: calculate from `resolution` and `aspect_ratio`.

`resolution` values for preset mode:

- `0.5 MP`
- `1.0 MP`
- `1.5 MP`
- `2.0 MP`
- `2.5 MP`

`aspect_ratio` values for preset mode:

- `1:1`
- `3:2`
- `2:3`
- `4:3`
- `3:4`
- `16:9`
- `9:16`

`width` and `height` can receive `INT` links, including the two Size Resolver outputs.

Final width and height are aligned to multiples of 16. No automatic hard cap is applied.

### Content controls

`content`:

- `empty`
- `from_image`

When `content = from_image`, connect `content_image` and choose:

- `long_edge`: preserve aspect ratio and resize so the image long edge matches the latent long edge; center and crop/pad as needed.
- `native`: preserve the image's original pixel size; center and crop/pad as needed.
- `stretch`: resize directly to the latent width and height.

`resize_method` applies only when resizing (`long_edge` or `stretch`).

Latent semantic reinterpretation remains available through `latent_semantic`, `latent_semantic_instruction`, and `latent_grounding_px`.

## Krea2 CcC Edit

Consumes the pre-built latent and executes Krea2 Edit.

Required:

- `model`
- `clip`
- `vae`
- `latent`
- `positive_prompt`
- `negative_prompt`
- `apply_krea2_edit_patch`

Optional:

- `visual_references`
- `semantic_references`

Final ordering:

```text
visual references -> latent semantic (when enabled) -> semantic-only references -> style references
```
