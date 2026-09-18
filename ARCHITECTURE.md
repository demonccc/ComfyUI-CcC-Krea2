# Architecture

## Edit Pipeline

```text
IMAGE -> Krea2 CcC Visual Reference --+
                                       |
IMAGE -> Krea2 CcC Visual Reference --+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
                                       |          ^
IMAGE -> Krea2 CcC Semantic Reference +          |
                                                  |
Images -> Krea2 CcC Size Resolver -> width/height |
                                                  |
VAE / dimensions / content -> Krea2 CcC Latent ---+
```

## Target Geometry vs Reference Geometry

Target latent construction and visual-reference preparation remain separate.

The target latent can be created from `from_image`, fixed width/height, Size Resolver output, or a preset. Once aligned, that target geometry becomes the grid used by Visual Reference.

Each visual reference has two independent paths:

```text
raw reference image
        |
        +-> optional Qwen path
        |      semantic
        |      semantic_resize
        |      semantic_grounding_px
        |      semantic_resize_method
        |      prompt_annotation
        |
        +-> VAE / DiT appearance path
               reference_fit = crop | resize | native
               placement_grid
               grid_horizontal_position
               grid_vertical_position
               resize_method
               boost
```

### Visual-reference fit

`crop` treats the target grid as a crop window **inside the source image**. Horizontal and vertical grid position determine where that window lands. For example, `left + down` retains the lower-left source region. Crop always forces `inside`; outside placement does not apply.

`resize` always resizes, whether the source starts smaller or larger. Aspect ratio is preserved and the source longest edge is mapped to the target-grid longest edge. The interpolation method is selectable.

`native` performs no intentional crop or resize and only applies the minimum technical VAE alignment required.

### Qwen visual grounding

`semantic` only decides whether an appearance reference is also visible to Qwen.

`semantic_resize = false` sends the Qwen copy at native resolution.

`semantic_resize = true` enables a maximum longest-edge cap through `semantic_grounding_px`. It is downscale-only: sources below the cap are unchanged. `semantic_resize_method` selects the interpolation method.

`prompt_annotation` is optional. Physical vision blocks remain ordered. When present, CcC appends:

```text
Image N: <prompt_annotation>
```

after the complete vision prefix. Empty annotations preserve the positional upstream-style contract.

### Qwen-only Semantic Reference

`semantic_only` is deliberately separate from the Identity Edit appearance path.

```text
semantic image
    -> Qwen3-VL with visual refs + prompt
    -> identify semantic vision span
    -> Krea2Moodboard subject transform on that span
       (span - mean) / std, blended by fidelity
    -> final positive conditioning
```

The semantic image is never VAE-encoded and never added to `reference_latents`, so the Identity Edit LoRA still receives only the configured Visual References. Semantic-only processing always uses the full image to preserve pose, people, outfit, interactions, background and composition. `fidelity=1.0` keeps the raw Qwen span; lower values move toward Moodboard subject/content extraction.

### RoPE and boost

For `resize` and `native`, `placement_grid` can be `inside` or `outside`; horizontal and vertical controls select the coordinate placement. Crop always uses `inside` because the grid is acting as the crop window itself.

Reference attention boost remains pass-specific:

```text
positive -> configured per-reference boost
negative -> 1.0 for every reference
```

## Size Resolver

Size Resolver is intentionally stateless and image-content agnostic.

It combines:

```text
long_edge = max(width, height) from long_edge_image
aspect    = width / height from aspect_ratio_image
```

and returns only:

```text
width
height
```

It does not align dimensions or create a latent.

## Latent

Latent owns final target geometry and VAE alignment.

Dimension source is explicit:

```text
from_image -> dimensions_image
fixed      -> width + height
preset     -> resolution + aspect_ratio
```

Regardless of the source, final pixel width and height are aligned to multiples of 16.

Content source is separate:

```text
empty
from_image -> content_image
```

Image-content placement supports `long_edge`, `native`, and `stretch`. Resizing method is relevant only to the modes that resize.

There is no automatic megapixel hard cap.

When `latent_semantic` is enabled, the original `content_image`, semantic instruction, and grounding size are stored as CcC metadata. Edit consumes that metadata without passing it into the runtime latent contract.

## Edit

`Krea2 CcC Edit` receives a pre-built `LATENT`; it does not own target dimensions or latent image placement.

Conditioning order remains:

```text
visual references -> latent semantic -> semantic-only references -> style references
```
