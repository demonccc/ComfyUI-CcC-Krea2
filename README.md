# ComfyUI-CcC-Krea2

Krea2 CcC Edit provides ComfyUI nodes for Krea 2 generation and reference-guided editing.

## Current Edit Architecture

The edit pipeline is intentionally split into a small set of focused nodes:

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

Content fit modes:

- `long_edge`
- `native`
- `stretch`

Final target dimensions are aligned to multiples of 16.

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

Krea2 CcC Paint adds arbitrary-mask inpainting and outpainting without resizing the source image.

The Paint path is split into:

- **Krea2 CcC Paint Prepare**: builds the aligned native canvas, combines the inpaint mask with outpaint expansion, applies signed mask grow/shrink plus directional feathering, and creates the neutralized semantic reference.
- **Krea2 CcC Paint**: image-grounds Qwen, VAE-encodes the semantic reference, creates the known-image latent with a soft token-aligned noise mask, and installs the registered t=0 reference/KV-cache runtime.

Paint Prepare mask controls:

- `mask_grow`: positive expands the generated region; negative shrinks it.
- `mask_blur_mode`: `standard` or `gaussian_sigma`.
- `mask_blur_amount`: `0` disables feathering.
- `mask_blur_direction`: `outside`, `inside`, or `both`.

`outside` keeps the full generated region strong and feathers into the preserved surroundings, which is a useful default for removing people or objects.

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
