# Architecture

## Single Edit Runtime

Krea2 CcC Edit has one edit execution path.

```text
IMAGE -> Visual Reference ----+
IMAGE -> Visual Reference ----+--> Edit --> Krea2 CcC Edit runtime --> Krea2
IMAGE -> Semantic Reference --+      ^
                                     |
Size Resolver -> Latent -------------+
```

No alternate runtime is selected based on other installed custom nodes.

## Visual Reference: two independent representations

A visual reference can travel through two independent paths.

```text
source image
   |
   +--> Qwen path
   |      preserve aspect ratio
   |      optional downscale-only cap
   |      optional Image N annotation
   |
   +--> appearance path
          crop | resize | contain | native
          VAE encode
          reference latent
          boost
          RoPE placement
```

The Qwen image does not need to match target pixel geometry. The appearance image does use target geometry rules before VAE encoding.

### Target resolution policy

Krea2 CcC Latent does not derive preset geometry from a megapixel target. Presets are explicit width/height pairs.

Krea's official repository documents Turbo output in the roughly 1K-2K range and exposes `width` / `height` as the primary resolution controls. Its sampling implementation aligns each axis to `VAE compression x DiT patch size`, which is 16 pixels for the released model. The official Hugging Face Space also exposes concrete target sizes instead of megapixel-derived geometry.

Official references:

- https://github.com/krea-ai/krea-2
- https://github.com/krea-ai/krea-2/blob/main/sampling.py
- https://huggingface.co/spaces/krea/Krea-2/blob/main/app.py

CcC therefore treats megapixels only as descriptive metadata for a preset. The CcC preset table is curated rather than claimed as an official exhaustive Krea bucket list.

Target geometry has two resolution paths for non-preset inputs:

- `nearest_krea_aspect` maps the requested fixed/image-derived geometry to the closest curated Krea preset aspect and uses size as the tie-breaker.
- `preserve_aspect_krea_bounds` keeps the requested aspect while resolving both axes into the 1024..2048 Krea working range and /16 alignment. Geometries too extreme to satisfy both axis bounds are rejected instead of being silently distorted.

Preset dimensions bypass those policies. Content fitting is a separate stage: `crop`, `contain`, or `stretch` is applied only when an image is actually used as latent content.

### Semantic grounding cadence

Krea2 CcC Visual Reference exposes `semantic_grounding_px` as an integer with step 32. Krea2 CcC Semantic Reference and the semantic path in Krea2 CcC Latent use the same 32-pixel cadence. This matches Qwen3-VL's effective spatial alignment: 16-pixel vision patches with merge size 2.

Values outside that cadence are not intrinsically invalid. For example, 380 px can still be processed, but Qwen aligns its effective visual grid to multiples of 32, so 384 px is the clearer equivalent for controlled experiments.

## Conditioning and runtime transport

Appearance reference latents are attached to CONDITIONING together with per-reference runtime metadata:

- fit state
- positive boost
- neutral negative boost
- RoPE placement

The Krea2 CcC Edit runtime consumes that metadata and executes one in-context sequence:

```text
[text | reference 1 | reference 2 | ... | target]
```

Text occupies frame 0 coordinates, references use ordered reference frames, and target image tokens use target frame coordinates.

## Semantic Reference

Semantic references are Qwen-only and do not create appearance latents.

`semantic_only` operates on the complete Qwen vision span. Style modes may split the source into multiple physical Qwen images before span processing.

## Target geometry

Target construction and reference preparation are separate.

Latent determines the final target width and height. Visual Reference then resolves its own pixel preparation against that target.

Public visual modes:

- `crop`
- `resize`
- `contain`
- `native`

`resize` maps the reference long edge to the corresponding target axis. `contain` instead calculates the maximum uniform scale that fits inside both target axes. Both preserve aspect ratio and use minimal centered crop-down alignment to /16. `native` keeps a 1:1 pixel scale and also crops down to /16 rather than padding.

RoPE placement is independent from pixel preparation.

## Size Resolver

Size Resolver returns only width and height. It does not resize image content and does not create a latent.

## Latent

Latent owns:

- target dimensions
- /16 alignment
- optional target content image
- target image placement
- optional latent semantic metadata

## Ownership

The current public edit architecture is implemented by Krea2 CcC Edit modules only. External projects that influenced individual ideas or techniques are acknowledged separately in [NOTICE](NOTICE).

## Reference cache

Reference caching is an optimization layer, not a second runtime.

```text
source image
   |
   +--> appearance preparation --> VAE --> appearance_latent ----+
   |                                                             |
   +--> Qwen visual path --> merged + grid + deepstack ----------+--> cache.safetensors
                                                                 |
runtime                                                          |
   prompt --> Qwen language path <--- cached Qwen visual --------+
   target --> Krea2 CcC Edit runtime <--- cached appearance latent ---+
```

The final Qwen conditioning is always recomputed because it depends on the current prompt.

The appearance latent is validated against the target geometry used to create it. Cached and uncached references converge on the same CONDITIONING/reference-latent transport and the same Krea2 CcC Edit runtime.
