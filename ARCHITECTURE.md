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

Target latent construction and Krea2 Edit reference preparation are intentionally separate.

The target latent can be created from:

```text
from_image
fixed width/height
Size Resolver -> fixed width/height
preset resolution + aspect ratio
```

It can also be empty or contain VAE-encoded image content, and that content can optionally participate semantically.

Once the target latent has been resolved and aligned to /16, its final geometry becomes the reference geometry contract for Krea2 Edit.

Every visual reference then follows:

```text
raw reference image
        |
        +-> Qwen semantic grounding path (when enabled)
        |
        +-> Krea2 Edit v1.2 pixel-space fit against resolved target
                |
                -> VAE encode
                |
                -> reference latent
                |
                -> optional RoPE coordinate placement/displacement
```

The visual-reference node does not expose a reference sizing mode. Krea2 Identity Edit v1.2 `fit` is mandatory for visual edit references.

## Visual References

Visual Reference nodes are ordered and append-only. The current two-reference workflow uses:

```text
scene -> subject
```

The standard RoPE placement is `inside:center:center`, matching the proven centered stride-1 behavior after v1.2 fitting.

CcC adds experimental RoPE freedom without changing reference sizing:

- grid: `inside` / `outside`
- horizontal: `center` / `left` / `right`
- vertical: `center` / `up` / `down`

This can move reference coordinates outside the target grid for experiments while preserving the same v1.2-fitted VAE reference.

Reference attention boost is pass-specific:

```text
positive -> configured per-reference boost
negative -> 1.0 for every reference
```

The negative remains grounded with the same Qwen images; only its reference attention boost is neutral, matching the proven Krea2 Identity Edit recipe.

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
