# ComfyUI-CcC-Krea2

CcC nodes for Krea 2 generation and editing in ComfyUI.

## Current Edit Architecture

The Edit surface is split into focused nodes:

- **Krea2 CcC Visual Reference**
- **Krea2 CcC Semantic Reference**
- **Krea2 CcC Size Resolver**
- **Krea2 CcC Latent**
- **Krea2 CcC Edit**

### Visual Reference

Use one node per visual Krea2 Edit reference and chain them in physical reference order.

The VAE pixel path has three explicit modes:

- `crop`: use the target grid as an inside crop window over the source. `grid_horizontal_position` and `grid_vertical_position` select the retained source region. Outside placement is disabled for crop.
- `resize`: always resize up or down while preserving aspect ratio; the source longest edge is mapped to the target-grid longest edge. Interpolation is selectable.
- `native`: preserve source size and pixels, with only minimum VAE alignment when required.

RoPE placement remains explicit through `placement_grid` (`inside` / `outside`), horizontal (`center` / `left` / `right`) and vertical (`center` / `up` / `down`) controls.

Qwen is independent from the VAE path. `semantic` decides whether the same image is also sent to Qwen. When `semantic_resize` is enabled, the Qwen copy is downscaled only if it exceeds `semantic_grounding_px`; smaller images are never upscaled. `semantic_resize_method` selects the downscale method.

`prompt_annotation` is optional and is appended as `Image N: <annotation>`. This allows lightweight guidance such as `Image 1: It is the scene image, only pay attention to the buildings.` without forcing a fixed scene/subject role model.

`boost` applies to the positive pass. The grounded negative uses the same reference images with boost fixed to `1.0`.

### Size Resolver

`Krea2 CcC Size Resolver` combines two images without carrying image content forward:

- `long_edge_image` contributes only its longest edge in pixels.
- `aspect_ratio_image` contributes only its width-to-height proportions.
- outputs are only `width` and `height`.

The resolver does not align to /16. `Krea2 CcC Latent` owns final VAE alignment.

### Latent

`Krea2 CcC Latent` separates **dimensions** from **content**.

Dimensions can be:

- `from_image`: use only the connected image dimensions.
- `fixed`: use `width` and `height`, which can be linked from Size Resolver.
- `preset`: calculate from a fixed megapixel resolution (`0.5 MP` through `2.5 MP`) and aspect ratio.

Content can be:

- `empty`
- `from_image`

Image content fit modes are:

- `long_edge`: match image long edge to latent long edge, preserve aspect ratio, center, and allow overflow/cropping on the other axis.
- `native`: keep image pixel size, center it, and allow padding or overflow.
- `stretch`: resize directly to the latent width and height.

`resize_method` is used by `long_edge` and `stretch`; `native` performs no resize.

Final latent pixel dimensions are aligned to multiples of 16. There is no automatic megapixel hard cap.

The important boundary is:

```text
build target latent however needed
        ->
resolve final target geometry
        ->
fit every Krea2 Edit visual reference to that target with v1.2 fit
        ->
optionally move only its RoPE coordinates
```

### Edit

`Krea2 CcC Edit` consumes the pre-built latent, visual references, and semantic references, then builds conditioning and applies the optional Krea2 Edit patch.

Reference ordering remains:

```text
visual references -> latent semantic (when enabled) -> semantic-only references -> style references
```

## Test Workflow

The repository intentionally contains one edit workflow:

[`workflows/01_scene_subject.json`](workflows/01_scene_subject.json)

It uses Subject as the long-edge reference and Scene as the aspect-ratio reference. Size Resolver outputs feed Latent `width` and `height` in `fixed` dimensions mode. Both visual references are then fitted to the resolved latent with Krea2 Identity Edit v1.2 geometry.

## Other Public Nodes

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

See [NODES.md](NODES.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [CHANGELOG.md](CHANGELOG.md).
