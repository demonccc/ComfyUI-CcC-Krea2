# Krea2 CcC Edit Nodes

## Krea2 CcC Visual Reference

Declares one ordered appearance reference.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | IMAGE | — | Source reference image. |
| `boost` | 0.0 .. 10.0 | 1.0 | Positive target-to-reference attention boost. |
| `reference_fit` | crop, resize, native | native | Pixel preparation before VAE encoding. |
| `placement_grid` | inside, outside | inside | RoPE placement grid. Crop always forces inside. |
| `grid_horizontal_position` | center, left, right | center | Horizontal placement. |
| `grid_vertical_position` | center, up, down | center | Vertical placement. |
| `resize_method` | lanczos, bicubic, bilinear, area | lanczos | Interpolation for resize mode. |
| `semantic` | boolean | true | Also show the source image to Qwen. |
| `semantic_resize` | boolean | true | Enable Qwen-only downscale when above the configured cap. |
| `semantic_grounding_px` | 32 .. 4096, step 32 | 768 | Maximum Qwen longest edge when semantic resize is enabled. |
| `semantic_resize_method` | lanczos, bicubic, bilinear, area | lanczos | Qwen downscale interpolation. |
| `prompt_annotation` | text | empty | Optional `Image N: ...` Qwen annotation. |

### Geometry modes

- `crop`: keep an inside source crop selected by the placement controls.
- `resize`: resize proportionally to the target-grid longest edge.
- `native`: preserve source pixels except for minimum VAE alignment.

Qwen preparation is independent from VAE geometry.

`semantic_grounding_px` uses a step of 32 because Qwen3-VL effectively aligns visual processing to a 32-pixel spatial cadence (16-pixel vision patches with merge size 2). Values such as 380 are not invalid, but they are aligned internally to the same kind of grid, so using 384 makes the effective resolution explicit and experiments easier to compare.

The same step-32 convention is used by `grounding_px` in **Krea2 CcC Semantic Reference** and `latent_grounding_px` in **Krea2 CcC Latent**.

## Krea2 CcC Semantic Reference

Adds a Qwen-only semantic/style reference.

Modes:

- `semantic_only`
- `style_direct`
- `style_indirect`

Semantic references do not create VAE reference latents.

`semantic_only` uses the full image and applies semantic extraction only to that image's Qwen span. `fidelity` controls how strongly the transformed span differs from the raw Qwen representation.

`grounding_px` is an integer with step 32 and defaults to 768. The step follows Qwen3-VL's effective 32-pixel visual grid cadence.

## Krea2 CcC Size Resolver

Inputs:

- `long_edge_image`
- `aspect_ratio_image`

Outputs:

- `width`
- `height`

It carries no image content forward.

## Krea2 CcC Latent

Builds the target latent.

Dimension modes:

- `from_image`
- `fixed`
- `preset`

Preset resolutions include 0.5, 1.0, 1.5, 2.0, 2.5 and 3.0 MP. The 3.0 MP entry is a convenience preset, not a hard limit: `fixed` and `from_image` remain free to resolve larger geometries when explicitly requested.

Content modes:

- `empty`
- `from_image`

Image fit:

- `long_edge`
- `native`
- `stretch`

Final dimensions are aligned to multiples of 16. Krea2 CcC Edit does not impose LoRA-specific megapixel limits; any narrower recommendation belongs to the particular model or adapter being used.

When latent semantic guidance is enabled, `latent_grounding_px` is an integer with step 32 and defaults to 768, matching the same Qwen3-VL visual-grid cadence used by Krea2 CcC Visual Reference and Krea2 CcC Semantic Reference.

## Krea2 CcC Edit

Required:

- `model`
- `clip`
- `vae`
- `latent`
- `positive_prompt`
- `apply_krea2_edit_patch`

Optional:

- `visual_references`
- `semantic_references`

Execution:

```text
1. Resolve ordered references.
2. Prepare Qwen images.
3. Resolve visual pixel geometry.
4. VAE-encode appearance references.
5. Attach reference latents and runtime metadata to CONDITIONING.
6. Apply the single Krea2 CcC Edit model patch.
7. Execute [text | refs | target].
```

Negative text is fixed empty by the edit contract. Appearance images remain available to the grounded negative pass with neutral reference boosts.

## Other Public Nodes

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

## Krea2 CcC Reference Cache Create

Precomputes one visual reference.

Inputs include the source image, Krea2 CLIP, VAE, target latent, appearance geometry controls, and semantic grounding controls. `semantic_grounding_px` uses the same integer step-32 convention as Krea2 CcC Visual Reference.

Outputs:

- `cache`
- `cache_info`

The cache stores the raw VAE appearance latent plus Qwen visual features. It does not store final prompt-dependent conditioning.

## Krea2 CcC Reference Cache Save / Load

Save writes portable `.safetensors` files under:

```text
ComfyUI/models/krea2_ccc_cache/
```

Load restores the cached tensors and metadata.

## Krea2 CcC Cached Visual Reference

Adds a loaded or newly created cache to the normal visual-reference chain.

Runtime controls remain editable:

- `boost`
- RoPE placement
- semantic enable/disable
- `prompt_annotation`

No VAE encode or Qwen Vision execution is required for a cached reference.
