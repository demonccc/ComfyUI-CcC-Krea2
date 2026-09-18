# Changelog

## Unreleased

### Changed
- `Semantic Reference / semantic_only` now uses Krea2Moodboard-style `subject` extraction on the Qwen vision span, full-image processing, a content/composition directive, and no VAE/LoRA reference latent. Default semantic fidelity/strength is `0.5`.
- Removed Easy Edit and the previous simple/Advanced Edit public surfaces.
- Split Edit responsibilities into Visual Reference, Semantic Reference, Size Resolver, Latent, and Edit nodes.
- Visual Reference exposes independent RoPE grid/horizontal/vertical controls.
- Removed the `fit_to_latent` visual-reference sizing toggle. All visual Krea2 Edit references now always use the Identity Edit v1.2 pixel-space `fit` geometry against the resolved target latent before VAE encoding.
- Kept CcC RoPE displacement independent from reference sizing, including optional positions outside the target grid.
- Reference attention boost is now conditioning-specific: configured boosts apply to positive conditioning and grounded negative conditioning always uses `1.0`.
- Visual edit reference latents, fit flags, boosts, and CcC RoPE placement now travel through CONDITIONING instead of being captured inside a MODEL-wrapper closure, matching the Krea2 Identity Edit transport contract.
- Grounded negative conditioning uses the same appearance images and fitted reference latents but excludes positive `semantic_role` and per-reference `instruction` text.
- Corrected genuine v1.2 `fit` mismatches to preserve the complete source image before resampling; /16 snapping no longer back-computes a hidden source crop.
- Added a regression contract for a `1719 x 1164` visual reference against a `992 x 992` target (`992 x 656` VAE input, `124 x 82` reference latent).
- Standard centered RoPE placement now uses the integer stride-1 centered offset used by the proven Krea2 Identity Edit implementation.
- Replaced the intermediate Geometry object with `Krea2 CcC Size Resolver`, which outputs only `width` and `height`.
- Size Resolver combines the long edge from one image with the aspect ratio from another image.
- Latent now has explicit `dimensions` modes: `from_image`, `fixed`, and `preset`.
- Latent preset resolution is limited to the selectable range `0.5 MP` through `2.5 MP`; no automatic hard cap is applied to other dimension modes.
- Latent content is now explicit (`empty` or `from_image`) and uses `content_image`.
- Image content fit modes are `long_edge`, `native`, and `stretch`, with selectable resize method for resizing modes.
- Final Latent width and height are aligned to multiples of 16.
- Updated the scene + subject workflow so Subject supplies long-edge scale, Scene supplies aspect ratio, Size Resolver feeds fixed Latent width/height, and visual references are always v1.2-fitted to that resolved target.
