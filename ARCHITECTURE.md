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
          crop | resize | native
          VAE encode
          reference latent
          boost
          RoPE placement
```

The Qwen image does not need to match target pixel geometry. The appearance image does use target geometry rules before VAE encoding.

### Target megapixel policy

Krea2 CcC Latent exposes convenience presets through 3.0 MP, but Krea2 CcC Edit does not treat 3.0 MP as a global model limit. Fixed dimensions and image-derived dimensions may resolve above that value when the user explicitly requests them.

Megapixel limits that come from a specific LoRA, adapter or checkpoint must be treated as model-specific guidance, not enforced as a generic Krea2 CcC Edit restriction.

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
- `native`

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
