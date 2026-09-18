# Architecture

## Single Edit Runtime

CcC Krea2 has one edit execution path.

```text
IMAGE -> Visual Reference ----+
IMAGE -> Visual Reference ----+--> Edit --> CcC runtime --> Krea2
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

## Conditioning and runtime transport

Appearance reference latents are attached to CONDITIONING together with per-reference runtime metadata:

- fit state
- positive boost
- neutral negative boost
- RoPE placement

The CcC runtime consumes that metadata and executes one in-context sequence:

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

The current public edit architecture is implemented by CcC modules only. External projects that influenced individual ideas or techniques are acknowledged separately in [NOTICE](NOTICE).
