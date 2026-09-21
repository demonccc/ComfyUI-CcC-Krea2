# ComfyUI-CcC-Krea2

Krea2 CcC Edit provides ComfyUI nodes for Krea 2 generation and reference-guided editing.

## Current Edit Architecture

The edit pipeline is intentionally split into a small set of focused nodes:

- **Krea2 CcC Attention Region**
- **Krea2 CcC Visual Reference**
- **Krea2 CcC Semantic Reference**
- **Krea2 CcC Size Resolver**
- **Krea2 CcC Latent**
- **Krea2 CcC Edit**

There is one Krea2 CcC Edit runtime. Visual references, semantic references, target latent construction, conditioning and model patching all converge on that runtime.

```text
Visual Reference ----+
Visual Reference ----+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
Semantic Reference --+          ^
                                |
Size Resolver --> Latent -------+
```

### Attention Region

Use one chainable node per tagged rectangular target region. The box is expressed as percentages of the target/content image and is carried through Latent target transforms before being projected to the Krea token grid.

When Latent content is an image, `contain` and `stretch` transform the boxes with the image. `crop` is intentionally strict: if the crop removes any part of a declared Attention Region, Latent raises an error. Boxes are never silently clipped or deleted.

### Visual Reference

Use one node per ordered appearance reference.

The VAE path exposes four modes:

- `crop`: the target grid acts as an inside crop window over the source. It may intentionally discard content.
- `resize`: map the reference long edge to the corresponding target edge, preserve aspect ratio, then center-crop only the minimum pixels required to land on a /16 grid.
- `contain`: scale the reference so it fits inside the target, preserve aspect ratio, then center-crop only the minimum pixels required to land on a /16 grid.
- `native`: keep the reference at 1:1 pixel scale and center-crop each edge down to /16. Native is the only mode that may remain larger than the target.

The /16 adjustment is always crop-down. Krea2 CcC Edit does not pad or stretch a visual reference merely to satisfy VAE grid alignment.

RoPE placement is controlled independently through `placement_grid`, horizontal position and vertical position.

The Qwen path is independent from VAE geometry. `semantic` controls whether the same source image is also shown to Qwen. When `semantic_resize` is enabled, the Qwen copy is downscaled only when it exceeds `semantic_grounding_px`; smaller images are not upscaled.

`semantic_grounding_px` is exposed as an integer with a step of 32. Qwen3-VL internally aligns visual processing to a 32-pixel spatial cadence (16-pixel vision patches with merge size 2). A non-multiple such as 380 is not inherently invalid, but Qwen will align the effective visual grid to that cadence, so values such as 384 are clearer and more reproducible for experiments.

`prompt_annotation` optionally adds `Image N: <annotation>` after the physical vision prefix.

`boost` applies to the positive pass. The grounded negative uses the same appearance references with neutral boost `1.0`.

Regional attention is optional per Visual Reference:

- `global`: current behavior; the reference can influence the complete target.
- `boost in region`: the configured boost is applied only to target tokens inside the matching `region_tag`.
- `only in region`: the reference is blocked outside the matching region; inside it, the normal boost still applies.

The region tag must exist in the Attention Region chain attached to Krea2 CcC Latent.

### Semantic Reference

Semantic Reference is Qwen-only. It does not create a visual/VAE reference latent.

Modes:

- `semantic_only`
- `style_direct`
- `style_indirect`

`semantic_only` uses the complete image and transforms only its Qwen vision span. This keeps pose, action, people, clothing, objects, background, framing and composition available without adding another appearance reference.

### Size Resolver

Size Resolver combines:

- the longest edge from `long_edge_image`
- the aspect ratio from `aspect_ratio_image`

and outputs only `width` and `height`.

### Latent

Latent owns final target geometry and VAE alignment.

Dimensions:

- `from_image`
- `fixed`
- `preset`

Content:

- `empty`
- `from_image`

Geometry policies for `fixed` and `from_image`:

- `nearest_krea_aspect`
- `preserve_aspect_krea_bounds`

`preset` bypasses geometry resolution because its target size is already explicit.

Content fit modes when `content=from_image`:

- `crop`
- `contain`
- `stretch`

`contain` uses white padding for pixels not occupied by the source image. Final target dimensions are aligned to multiples of 16.

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

The preset list in CcC is curated, not an official exhaustive Krea whitelist. `fixed` remains available for deliberate custom geometries, while `from_image` remains a separate image-derived path.


### Edit

Edit consumes the prepared latent plus visual and semantic reference chains.

The current execution path is:

```text
references
  -> Qwen preparation
  -> visual pixel geometry
  -> VAE reference latents
  -> CONDITIONING metadata
  -> Krea2 CcC Edit runtime
  -> [text | refs | target]
```

Reference ordering is preserved physically.

## Reference Cache

Krea2 CcC Edit can precompute and persist one visual reference as a portable `.safetensors` cache.

The cache contains two prompt-independent payloads:

- the raw VAE appearance latent for the target geometry used during cache creation;
- Qwen3-VL visual features: `merged`, `grid`, and all DeepStack tensors.

The prompt-dependent Qwen language path is never cached. A new prompt still produces new conditioning while reusing the cached visual features.

Public cache nodes:

- **Krea2 CcC Reference Cache Create**
- **Krea2 CcC Reference Cache Save**
- **Krea2 CcC Reference Cache Load**
- **Krea2 CcC Cached Visual Reference**

Cached and normal visual references can be mixed in the same ordered reference chain. Runtime controls such as boost, RoPE placement, semantic enablement, and prompt annotation remain adjustable after loading the cache.

Appearance cache geometry is target-specific. If the output geometry changes, create another appearance cache for that target. The Qwen payload itself is prompt-independent.

The cache design was informed by **ComfyUI-Krea2IdentityMod** by ArtemKo7v:
https://github.com/ArtemKo7v/ComfyUI-Krea2IdentityMod

In particular, that project demonstrated the usefulness of persisting raw appearance latents and identified the Qwen3-VL visual cache boundary that requires `merged + grid + deepstack`, rather than caching final prompt-conditioned conditioning.

## Paint

Krea2 CcC Paint uses a temporary Krea working geometry without resizing the known source image.

The Paint path is split into four focused nodes:

- **Krea2 CcC Paint Geometry**: maps the requested native canvas to a curated Krea target using only `pad` or `crop`. The source is never resized.
- **Krea2 CcC Paint Prepare**: applies mask grow/feather, builds the semantic reference, VAE-encodes the working canvas, and returns the sampling `LATENT` with a token-aligned `noise_mask`.
- **Krea2 CcC Paint**: image-grounds Qwen, attaches the pre-encoded appearance reference, and installs the registered t=0 reference/KV-cache runtime.
- **Krea2 CcC Paint Restore**: returns the decoded result to the requested native canvas. Pad mode removes only temporary Krea padding; crop mode composites the generated crop back into the preserved native canvas.

Paint Geometry modes:

- `pad`: chooses the closest curated Krea geometry that fully contains the requested canvas. Padding can be placed left/center/right and top/center/bottom.
- `crop`: chooses the closest curated Krea geometry that fits completely inside the requested canvas. Crop origin follows the same horizontal/vertical position controls.

Padding fill options are `edge` (default), `reflect`, `neutral`, and `white`. All temporary padding is included in the generation mask.

Guardrails prevent implicit resizing:

- If no curated Krea geometry can fit inside the requested canvas, `crop` fails and asks to use padding.
- If no curated Krea geometry can contain the requested canvas, `pad` fails and asks to use crop.

Explicit `expand_left/top/right/bottom` values belong to the requested native/outpaint canvas. Restore removes only the temporary Krea normalization added after those expansions.

The source image and its mask always undergo the same geometric transform so mask coordinates cannot drift.


## Test Workflows

The repository contains:

- [`workflows/01_scene_subject.json`](workflows/01_scene_subject.json) — Scene + Subject Edit.
- [`workflows/02_anypaint_remove_people.json`](workflows/02_anypaint_remove_people.json) — AnyPaint inpaint test for removing masked people/objects.

## Other Public Nodes

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

See [NODES.md](NODES.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Acknowledgements


Krea2 CcC Edit was informed by work from the ComfyUI and Krea 2 community. The projects, commits, ideas and licenses that influenced the implementation are documented in [NOTICE](NOTICE). Those references are kept for attribution and gratitude; the active runtime and public architecture described above are the Krea2 CcC Edit implementation.
