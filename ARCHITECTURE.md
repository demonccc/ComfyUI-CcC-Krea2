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

## Visual References

Visual Reference nodes are ordered and append-only. The current two-reference workflow uses:

```text
scene -> subject
```

`fit_to_latent=true` contains the VAE reference inside target geometry. `fit_to_latent=false` keeps native image scale and only pads to the VAE /16 requirement, so its RoPE grid can extend beyond the target.

RoPE placement is represented by three independent controls: grid, horizontal alignment, and vertical alignment.

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

`Krea2 CcC Edit` receives a pre-built `LATENT`; it does not own target dimensions or image placement.

Conditioning order remains:

```text
visual references -> latent semantic -> semantic-only references -> style references
```
