# ComfyUI-CcC-Krea2

CcC nodes for Krea 2 generation and reference-guided editing in ComfyUI.

## Current Edit Architecture

The edit pipeline is intentionally split into a small set of focused nodes:

- **Krea2 CcC Visual Reference**
- **Krea2 CcC Semantic Reference**
- **Krea2 CcC Size Resolver**
- **Krea2 CcC Latent**
- **Krea2 CcC Edit**

There is one CcC edit runtime. Visual references, semantic references, target latent construction, conditioning and model patching all converge on that runtime.

```text
Visual Reference ----+
Visual Reference ----+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
Semantic Reference --+          ^
                                |
Size Resolver --> Latent -------+
```

### Visual Reference

Use one node per ordered appearance reference.

The VAE path exposes three modes:

- `crop`: the target grid acts as an inside crop window over the source.
- `resize`: resize proportionally so the source longest edge matches the target-grid longest edge.
- `native`: preserve source pixels, applying only minimum VAE alignment when required.

RoPE placement is controlled independently through `placement_grid`, horizontal position and vertical position.

The Qwen path is independent from VAE geometry. `semantic` controls whether the same source image is also shown to Qwen. When `semantic_resize` is enabled, the Qwen copy is downscaled only when it exceeds `semantic_grounding_px`; smaller images are not upscaled.

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
  -> CcC Krea2 runtime
  -> [text | refs | target]
```

Reference ordering is preserved physically.

## Test Workflow

The repository contains one current edit workflow:

[`workflows/01_scene_subject.json`](workflows/01_scene_subject.json)

## Other Public Nodes

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

See [NODES.md](NODES.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Acknowledgements

CcC Krea2 was informed by work from the ComfyUI and Krea 2 community. The projects, commits, ideas and licenses that influenced the implementation are documented in [NOTICE](NOTICE). Those references are kept for attribution and gratitude; the active runtime and public architecture described above are the CcC implementation.
