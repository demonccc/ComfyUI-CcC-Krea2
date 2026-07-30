# Changelog

All notable changes to the `ComfyUI-CcC-Krea2` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2026-07-30

### Added
- Experimental Alpha release of `ComfyUI-CcC-Krea2` suite.
- 4 general editing nodes (`Subject`, `Subject + Outfit`, `Subject + Scene`, `Subject + Scene + Outfit`).
- 3 inpainting nodes (`Inpaint`, `Inpaint Subject + Outfit`, `Inpaint Subject + Scene`).
- Standard ComfyUI `DIFFUSION_MODEL` wrapper signature `(executor, x, timesteps, context, *wargs, **kwargs)` with closure transport.
- Applied `model.model.process_latent_in(...)` to every reference VAE latent.
- Native pixel-path reference geometry (`reference_fit_mode="fit"`) aligned to multiples of 16 without black canvas padding.
- Explicit inpaint base role selection per node.
- Pure PyTorch mask dilation via `F.max_pool2d` removing `scipy` dependency.
- Model-driven empty latent generation honoring `batch_size`.
- Mandatory VAE input across all 7 custom node definitions.
- Unit test suite in `tests/` covering mock integration, geometry, masks, dynamic prompt templates, and patch isolation.
