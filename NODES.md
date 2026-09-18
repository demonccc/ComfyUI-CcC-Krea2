# Krea2 CcC Nodes

## Krea2 CcC Visual Reference

Declares one ordered visual reference for Krea2 Edit.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | `IMAGE` | — | Visual edit reference. |
| `boost` | `0.0 .. 10.0` | `1.0` | Positive-pass target-to-reference attention boost. The grounded negative always uses `1.0`. |
| `reference_fit` | `crop`, `resize`, `native` | `native` | Pixel-space behavior before VAE encoding. |
| `placement_grid` | `inside`, `outside` | `inside` | RoPE placement grid for `resize` and `native`. Crop always forces `inside`. |
| `grid_horizontal_position` | `center`, `left`, `right` | `center` | Horizontal grid placement. In crop mode this chooses where the crop window lands on the source. |
| `grid_vertical_position` | `center`, `up`, `down` | `center` | Vertical grid placement. In crop mode this chooses where the crop window lands on the source. |
| `resize_method` | `lanczos`, `bicubic`, `bilinear`, `area` | `lanczos` | Interpolation used only when `reference_fit = resize`. |
| `semantic` | boolean | `true` | Also send this visual reference to Qwen. |
| `semantic_resize` | boolean | `true` | Enable Qwen-only downscale when the image exceeds `semantic_grounding_px`. Never upscales. |
| `semantic_grounding_px` | `16 .. 4096` | `768` | Maximum longest edge for the Qwen copy when `semantic_resize` is enabled. |
| `semantic_resize_method` | `lanczos`, `bicubic`, `bilinear`, `area` | `lanczos` | Qwen-only downscale interpolation. |
| `prompt_annotation` | multiline text | empty | Optional text appended as `Image N: <annotation>` after the complete vision prefix. |

### Visual reference geometry

The three public modes are deliberately explicit:

- `crop`: the resolved target grid is used as an **inside crop window** over the source image. The horizontal/vertical controls decide which source region survives. No intentional resize is performed. `placement_grid` is forced to `inside`.
- `resize`: always resizes, whether the source is smaller or larger. Aspect ratio is preserved and the source longest edge is mapped to the target-grid longest edge using `resize_method`.
- `native`: no intentional crop or resize. Only minimum VAE alignment is applied when required.

Qwen preparation is independent from VAE geometry:

- `semantic = false`: the reference is VAE/DiT only.
- `semantic = true, semantic_resize = false`: Qwen receives native source pixels.
- `semantic = true, semantic_resize = true`: Qwen receives a downscaled copy only when the source exceeds `semantic_grounding_px`; smaller sources are unchanged.
- `prompt_annotation` is optional. Without it, visual references remain purely positional. With it, CcC adds `Image N: ...` text while preserving physical reference order.

Configured `boost` applies to the positive conditioning pass only. The grounded negative uses the same visual references with reference boost fixed to `1.0`.

## Krea2 CcC Semantic Reference

Declares one Qwen-only semantic/style reference. It never adds a VAE reference latent and therefore does not consume an Identity Edit LoRA reference slot.

Modes remain `semantic_only`, `style_direct`, and `style_indirect`.

`semantic_only` now follows the Krea2Moodboard **subject extraction** mechanics:

- Qwen sees the complete semantic image together with the edit references and prompt.
- The semantic image uses `full` processing; crop/tile modes are intentionally disabled so pose, people, outfit, background, interactions and composition remain available.
- After Qwen encoding, only the semantic image span is transformed with Moodboard subject whitening: `(span - mean) / std`.
- The semantic image is still **not** VAE-encoded and is not added to `reference_latents`.
- A built-in content/composition directive tells Qwen to use pose, action, clothing, people, interactions, objects, background, framing and composition while leaving target identity to the edit identity references.
- `fidelity` follows Moodboard strength semantics: `1.0` keeps the raw Qwen vision rows; lower values blend progressively toward the subject/content extraction. Default is `0.5`.
- `instruction` is optional extra guidance and is appended to the built-in semantic directive.

`style_direct` and `style_indirect` keep the existing Moodboard-style style extraction behavior.

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

Execution contract:

```text
1. Resolve target latent geometry.
2. Fit every visual reference to that target with Krea2 Edit v1.2 pixel-space fit.
3. VAE-encode fitted visual references.
4. Apply optional RoPE coordinate displacement.
5. Use configured reference boosts on positive conditioning.
6. Use reference boost 1.0 on grounded negative conditioning.
```

Conditioning ordering remains:

```text
visual references -> latent semantic (when enabled) -> semantic-only references -> style references
```
