# Changelog

## Unreleased

### Changed
- Removed Easy Edit and the previous simple/Advanced Edit public surfaces.
- Split Edit responsibilities into Visual Reference, Semantic Reference, Size Resolver, Latent, and Edit nodes.
- Visual Reference now exposes `fit_to_latent` and independent RoPE grid/horizontal/vertical controls.
- `fit_to_latent=false` preserves native reference scale apart from /16 VAE padding.
- Replaced the intermediate Geometry object with `Krea2 CcC Size Resolver`, which outputs only `width` and `height`.
- Size Resolver combines the long edge from one image with the aspect ratio from another image.
- Latent now has explicit `dimensions` modes: `from_image`, `fixed`, and `preset`.
- Latent preset resolution is limited to the selectable range `0.5 MP` through `2.5 MP`; no automatic hard cap is applied to other dimension modes.
- Latent content is now explicit (`empty` or `from_image`) and uses `content_image`.
- Image content fit modes are `long_edge`, `native`, and `stretch`, with selectable resize method for resizing modes.
- Final Latent width and height are aligned to multiples of 16.
- Updated the scene + subject workflow so Subject supplies long-edge scale, Scene supplies aspect ratio, and Size Resolver feeds fixed Latent width/height.
