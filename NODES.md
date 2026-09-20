# Krea2 CcC Edit Nodes

## Krea2 CcC Visual Reference

Declares one ordered appearance reference.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `image` | IMAGE | — | Source reference image. |
| `boost` | 0.0 .. 10.0 | 1.0 | Positive target-to-reference attention boost. |
| `reference_fit` | crop, resize, contain, native | native | How the visual reference fits the target latent before VAE encoding. |
| `placement_grid` | inside, outside | inside | RoPE placement grid. Crop always forces inside. |
| `grid_horizontal_position` | center, left, right | center | Horizontal placement. |
| `grid_vertical_position` | center, up, down | center | Vertical placement. |
| `resize_method` | lanczos, bicubic, bilinear, area | lanczos | Interpolation for resize and contain modes. |
| `semantic` | boolean | true | Also show the source image to Qwen. |
| `semantic_resize` | boolean | true | Enable Qwen-only downscale when above the configured cap. |
| `semantic_grounding_px` | 32 .. 4096, step 32 | 768 | Maximum Qwen longest edge when semantic resize is enabled. |
| `semantic_resize_method` | lanczos, bicubic, bilinear, area | lanczos | Qwen downscale interpolation. |
| `prompt_annotation` | text | empty | Optional `Image N: ...` Qwen annotation. |

### Reference fit modes

- `crop`: use the target as an inside window over the reference. Content outside the selected window can be discarded.
- `resize`: use the reference orientation as the anchor. Its long edge is mapped to the corresponding target axis, the other edge follows proportionally, and only the minimum centered source crop required for /16 alignment is removed.
- `contain`: calculate the maximum uniform scale that keeps the reference inside the target, then apply only the minimum centered source crop required for /16 alignment.
- `native`: keep a 1:1 source pixel scale and center-crop width and height down to /16. It can be larger than the target.

For `resize`, `contain`, and `native`, /16 alignment is crop-down rather than pad-up or non-uniform resizing. The alignment crop is only a few edge pixels needed by the VAE grid and is distinct from the explicit `crop` mode.

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

Preset sizes are explicit target geometries rather than a generated combination of megapixels plus aspect ratio.

Current presets:

| Size | Aspect ratio | Approx. pixels |
| --- | --- | --- |
| `1024 x 1024` | `1:1` | `~1.05 MP` |
| `1216 x 832` | `~3:2` | `~1.01 MP` |
| `832 x 1216` | `~2:3` | `~1.01 MP` |
| `1536 x 1024` | `3:2` | `~1.57 MP` |
| `1024 x 1536` | `2:3` | `~1.57 MP` |
| `1536 x 1152` | `4:3` | `~1.77 MP` |
| `1152 x 1536` | `3:4` | `~1.77 MP` |
| `1536 x 864` | `16:9` | `~1.33 MP` |
| `864 x 1536` | `9:16` | `~1.33 MP` |
| `2048 x 1536` | `4:3` | `~3.15 MP` |
| `1536 x 2048` | `3:4` | `~3.15 MP` |
| `2048 x 1152` | `16:9` | `~2.36 MP` |
| `1152 x 2048` | `9:16` | `~2.36 MP` |
| `2048 x 2048` | `1:1` | `~4.19 MP` |

Why explicit sizes:

- Krea 2 Turbo is documented by Krea as generating from roughly 1K to 2K, with `width` and `height` as the primary resolution controls.
- The official inference code rounds each dimension up to the model alignment, which is VAE compression multiplied by the DiT patch size; for the released model this is a 16-pixel cadence.
- Krea's official Hugging Face Space exposes concrete presets such as `1024 x 1024`, `1216 x 832`, `832 x 1216`, and `2048 x 2048`.
- Megapixels are therefore descriptive metadata for a preset, not the rule used to derive its geometry.

Official references:

- Krea 2 official repository: https://github.com/krea-ai/krea-2
- Krea 2 official sampling implementation: https://github.com/krea-ai/krea-2/blob/main/sampling.py
- Krea 2 official Hugging Face Space preset implementation: https://huggingface.co/spaces/krea/Krea-2/blob/main/app.py

The preset list in CcC is curated, not an official exhaustive Krea whitelist. `fixed` remains available for deliberate custom geometries, while `from_image` remains a separate image-derived path.

Content modes:

- `empty`
- `from_image`

Image fit:

- `long_edge`
- `native`
- `stretch`

Final dimensions are aligned to multiples of 16. Preset dimensions are already /16 and are used exactly as listed.

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


## Krea2 CcC Paint Prepare

Builds the native paint canvas without resizing the source image.

Required:

- `image`

Optional:

- `mask`

Canvas controls:

- `expand_left`
- `expand_top`
- `expand_right`
- `expand_bottom`

The source image is placed unchanged inside the expanded canvas. Final dimensions are aligned upward to multiples of 16 by adding pixels on the right and bottom.

Mask controls:

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `mask_grow` | -256 .. 256 | 0 | Positive expands the generation region; negative shrinks it. |
| `mask_blur_mode` | standard, gaussian_sigma | gaussian_sigma | Selects box-style feathering or Gaussian sigma feathering. |
| `mask_blur_amount` | 0 .. 128 | 0 | Feather radius/sigma. Zero means no blur. |
| `mask_blur_direction` | outside, inside, both | outside | Chooses where the soft transition is allowed relative to the hard mask boundary. |

Direction semantics:

- `outside`: keep the generated region at full strength and feather into the preserved surroundings.
- `inside`: keep the outside strictly protected and feather only inside the generated region.
- `both`: feather across both sides of the original boundary.

Processing order:

```text
source mask + outpaint canvas
  -> signed grow/shrink
  -> directional feather
  -> generated_mask
  -> keep_mask
  -> neutralized semantic reference
```

Outputs:

- `paint_context`
- `prepared_image`
- `semantic_reference`
- `generated_mask`
- `keep_mask`
- `paint_prepare_info`

## Krea2 CcC Paint

Consumes `KREA2_PAINT_CONTEXT` and prepares the runtime inputs for AnyPaint-style Krea 2 sampling.

It:

1. Grounds Qwen with the neutralized semantic reference.
2. VAE-encodes the semantic reference as the registered appearance reference.
3. VAE-encodes the known canvas.
4. Converts the soft generation mask to a token-aligned ComfyUI `noise_mask`.
5. Applies the registered t=0 reference runtime with optional isolated reference K/V cache.

The reference is registered over the complete target grid. Known-region preservation happens during sampling through the latent/noise-mask path rather than by a final source-image composite.

Recommended first test:

- Krea 2 Turbo
- `krea2_anypaint_rank32.safetensors`
- 8 steps
- CFG 1
- Euler
- simple scheduler
