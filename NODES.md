# Krea2 CcC Edit Nodes

## Krea2 CcC Attention Region

Declares one tagged rectangular target region and can be chained with more regions.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `tag` | text | subject | Unique tag used by Visual Reference. |
| `x` | 0 .. 100 | 0 | Left edge as percentage of the target/content image. |
| `y` | 0 .. 100 | 0 | Top edge as percentage of the target/content image. |
| `width` | 0.1 .. 100 | 100 | Region width as percentage. |
| `height` | 0.1 .. 100 | 100 | Region height as percentage. |
| `previous_regions` | KREA2_ATTENTION_REGION_CHAIN | optional | Chains another tagged region. |

Tags must be unique. Latent resolves the boxes through target pixel transforms and stores the transformed coordinates for Edit. If image-content `crop` removes even part of a box, Latent raises an error instead of clipping that box.

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
| `prompt_annotation` | text | empty | Optional role-only label inserted as `Image N: <text>` before the positive prompt. Use it only to identify the image (for example `This is the subject image.`); edit instructions belong in `positive_prompt`. It is never copied to the negative branch. |
| `attention_scope` | global, boost in region, only in region | global | Target-side spatial scope for this reference. |
| `region_tag` | text | empty | Tag of the Attention Region used by regional scopes. |

### Reference fit modes

- `crop`: use the target as an inside window over the reference. Content outside the selected window can be discarded.
- `resize`: use the reference orientation as the anchor. Its long edge is mapped to the corresponding target axis, the other edge follows proportionally, and only the minimum centered source crop required for /16 alignment is removed.
- `contain`: calculate the maximum uniform scale that keeps the reference inside the target, then apply only the minimum centered source crop required for /16 alignment.
- `native`: keep a 1:1 source pixel scale and center-crop width and height down to /16. It can be larger than the target.

For `resize`, `contain`, and `native`, /16 alignment is crop-down rather than pad-up or non-uniform resizing. The alignment crop is only a few edge pixels needed by the VAE grid and is distinct from the explicit `crop` mode.

Qwen preparation is independent from VAE geometry. Visual `prompt_annotation` text is identification-only: all physical vision blocks come first, then `Image N: ...` role labels, then the raw positive edit prompt. The negative branch keeps the appearance-image vision blocks but has no role labels or negative text.

Regional attention is target-side. `boost in region` applies the existing reference boost only to target queries inside the resolved region. `only in region` additionally blocks target queries outside the region from attending to that reference.

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

The preset list in CcC is curated, not an official exhaustive Krea whitelist.

Geometry resolution:

- `preset` uses the selected preset exactly and bypasses geometry policy.
- `fixed` starts from the requested width/height and then resolves it through `geometry_policy`.
- `from_image` starts from `dimensions_image` and then resolves it through `geometry_policy`.

Geometry policies:

- `nearest_krea_aspect`: selects the curated Krea preset geometry whose aspect ratio is closest to the source geometry; when the same aspect exists at multiple sizes, the closer size wins.
- `preserve_aspect_krea_bounds`: preserves the source aspect ratio as closely as /16 alignment allows, keeps both axes within the Krea 1024..2048 working range, and rejects aspect ratios that cannot satisfy both bounds without distortion.

Content modes:

- `empty`
- `from_image`

When `content=from_image`, `content_fit` controls how the content image fills the already-resolved target geometry:

- `crop`: preserve aspect, cover the target, then center-crop the excess.
- `contain`: preserve aspect, fit the complete image, then fill unused target pixels with white padding.
- `stretch`: resize directly to target width/height without preserving aspect ratio.

`resize_method` controls interpolation for those content transformations.

Final dimensions are aligned to multiples of 16. Preset dimensions are already /16 and are used exactly as listed.

When latent semantic guidance is enabled, `latent_grounding_px` is an integer with step 32 and defaults to 768, matching the same Qwen3-VL visual-grid cadence used by Krea2 CcC Visual Reference and Krea2 CcC Semantic Reference.

## Krea2 CcC Edit Prompt Creator

Optional multimodal prompt-generation node. It does not replace or modify **Edit**, **Latent**, **Visual Reference**, or **Semantic Reference**.

Inputs:

- `clip`: the same multimodal Krea2/Qwen3-VL `CLIP` used by Edit.
- `visual_references`: existing `KREA2_VISUAL_REFERENCE_CHAIN`; returned unchanged.
- `reference_edit_image`: optional internal image used to understand an edit situation.

Controls:

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `mode` | enhance, create_from_image, create_from_theme | enhance | Selects how the prompt is created. |
| `user_prompt` | text | empty | Existing instruction, image-guided hint, or creative theme. |
| `max_tokens` | 32 .. 4096 | 512 | Generation length passed to the same CLIP text generator used by Comfy Generate Text. |
| `temperature` | 0.01 .. 2.0 | 0.25 | Sampling temperature. |
| `top_p` | 0 .. 1 | 0.90 | Nucleus sampling threshold. |

Modes:

- `enhance`: strengthens an existing edit instruction without changing its intent.
- `create_from_image`: analyzes `reference_edit_image` for action, pose, interaction, environment, framing, and composition, then converts those details into text around the Krea visual-reference subject(s). The internal image is not inserted into the Visual Reference chain.
- `create_from_theme`: expands a high-level theme into a concrete new scene/action while preserving the referenced subject(s).

The node uses `clip.tokenize(..., images=[...])` plus `clip.generate(...)`, matching the multimodal generation path behind ComfyUI **Generate Text**. Krea visual-reference Image N numbering is preserved. `reference_edit_image` is internal to the Creator: generated prompts must describe its useful content explicitly rather than relying on Edit seeing that image.

Outputs:

- `created_prompt`
- `visual_references` (passthrough, unchanged)
- `creator_info`

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

## Krea2 CcC Character Sheet

Builds one deterministic Character Sheet image from generic portrait and body references.

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `sheet_type` | 1 portrait, 2 portraits, 3 portraits, 4 portraits, 1 body, 2 bodies, 1 portrait + 2 bodies, 3 portraits + 1 body, 4 portraits + 1 body | 1 portrait | Selects the layout and which generic inputs are required. |
| `sheet_geometry` | curated Krea preset geometries | 1024 x 1024 | Final sheet width and height. The choices are reused directly from Krea2 CcC Latent, rather than accepting arbitrary /16 sizes. |
| `resize_method` | lanczos, bicubic, bilinear, area | lanczos | Interpolation used by the fixed fit operation when resizing each source into its slot. |
| `padding` | 0 .. 128 | 8 | Gap between slots. |
| `outer_margin` | 0 .. 128 | 8 | Margin around the final sheet. |
| `background` | white, gray, black | white | Fill visible around fitted sources. |

Optional image inputs are `portrait_1` through `portrait_4` and `body_1` through `body_2`. The numbers only indicate ordering; the node does not require specific camera angles.

Every source always uses **fit** behavior: preserve aspect ratio, scale until the complete source fits inside its slot, center it, and use the sheet background for any remaining space. Character Sheet does not expose crop/cover/stretch. `resize_method` controls only the interpolation algorithm used by that fit resize.

The node is a deterministic compositor only. It does not invent missing views and does not run a model, Qwen, VAE, or latent processing.

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


## Krea2 CcC Paint Geometry

Resolves an image + mask pair to a curated Krea working geometry without resizing the source.

Controls:

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `geometry_mode` | pad, crop | pad | Pad outward to a containing Krea geometry, or crop inward to an inner Krea geometry. |
| `padding_fill` | edge, reflect, neutral, white | edge | Pixel fill used for temporary padding and explicit outpaint expansion. |
| `horizontal_position` | center, left, right | center | Chooses where padding is placed or where the crop window is anchored horizontally. |
| `vertical_position` | center, top, bottom | center | Chooses where padding is placed or where the crop window is anchored vertically. |
| `expand_left/top/right/bottom` | 0 .. 8192, step 16 | 0 | Explicit outpaint expansion. It belongs to the requested/native canvas and is preserved after restore. |

The Krea working geometry is selected from the same curated target table used by **Krea2 CcC Latent**. Feasible candidates are ranked by aspect-ratio proximity, then size proximity.

Guardrails:

- `crop` requires at least one curated Krea geometry that fits completely inside the requested canvas. Otherwise it fails with a message to use padding.
- `pad` requires at least one curated Krea geometry that completely contains the requested canvas. Otherwise it fails with a message to use crop.

Temporary padding is always marked as generable in the returned mask. Image and mask share exactly the same crop/pad transform.

Explicit outpaint expansion and temporary Krea padding are tracked separately. Both remain generable, but Paint Prepare can preserve the explicit expansion's configured `padding_fill` as reference context while still neutralizing temporary Krea padding.

Outputs:

- `paint_geometry`
- `prepared_image`
- `prepared_mask`
- `geometry_info`

## Krea2 CcC Paint Prepare

Consumes `KREA2_PAINT_GEOMETRY` plus the VAE.

Mask controls:

| Control | Values | Default | Behavior |
| --- | --- | --- | --- |
| `fill_holes` | boolean | false | Fills only background regions completely enclosed by the mask. Open gaps are not closed. |
| `mask_grow` | -256 .. 256 | 0 | Positive expands the generation region; negative shrinks it. |
| `mask_blur_mode` | standard, gaussian_sigma | gaussian_sigma | Box-style or Gaussian feathering. |
| `mask_blur_amount` | 0 .. 128 | 0 | Feather radius/sigma. |
| `mask_blur_direction` | outside, inside, both | outside | Controls which side of the hard boundary receives the soft transition. |

Processing:

```text
prepared image + prepared mask
  -> optional fill holes
  -> signed grow/shrink
  -> directional feather
  -> generated_mask / keep_mask
  -> preserve explicit outpaint padding-fill as semantic context
  -> neutralize manual/generated holes and temporary Krea padding
  -> semantic reference
  -> VAE known-image latent
  -> token-aligned noise_mask
  -> sampling LATENT
```

Paint Prepare also VAE-encodes the semantic reference for the Paint runtime. Explicit `expand_left/top/right/bottom` margins remain fully generable, but their edge/reflect/neutral/white fill is kept in the semantic/reference image so AnyPaint can see border color and lighting continuity. A normal user-painted mask is still neutralized as before.

Outputs:

- `paint_context`
- `latent`
- `prepared_image`
- `semantic_reference`
- `generated_mask`
- `keep_mask`
- `paint_prepare_info`

## Krea2 CcC Paint

Consumes `KREA2_PAINT_CONTEXT`. It no longer creates the target latent.

It:

1. Grounds Qwen with the neutralized semantic reference.
2. Attaches the appearance reference latent prepared by Paint Prepare.
3. Applies the registered t=0 reference runtime with optional isolated reference K/V cache.
4. Returns patched model plus positive/negative conditioning.

The `LATENT` used by KSampler comes directly from **Krea2 CcC Paint Prepare**.

## Krea2 CcC Paint Restore

Consumes the decoded working image plus the `KREA2_PAINT_GEOMETRY` context.

- `pad`: removes only the temporary Krea padding and preserves explicit user-requested expansion.
- `crop`: composites the generated crop back at its exact original coordinates. When `generated_mask` is connected, only that generated/feathered region replaces the preserved base canvas.

No inverse resize is performed.

Recommended first test:

- Krea 2 Turbo
- `krea2_anypaint_rank32.safetensors`
- 8 steps
- CFG 1
- Euler
- simple scheduler

