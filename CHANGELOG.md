# Changelog

All notable changes to the `ComfyUI-CcC-Krea2` custom node package will be documented in this file.

## [0.1.0] - 2026-07-30

### Added
- Initial production release of `ComfyUI-CcC-Krea2`.
- Seven explicit, reference-guided Krea 2 editing nodes under category `CcC/Krea2`:
  - `CcC Krea2 - Subject`
  - `CcC Krea2 - Subject + Outfit`
  - `CcC Krea2 - Subject + Scene`
  - `CcC Krea2 - Subject + Scene + Outfit`
  - `CcC Krea2 - Inpaint`
  - `CcC Krea2 - Inpaint Subject + Outfit`
  - `CcC Krea2 - Inpaint Subject + Scene`
- Per-instance model patching via ComfyUI `ModelPatcher` wrapper extensions (zero global monkey-patching).
- Dual image path architecture: independent Qwen3-VL grounding resize and pixel-space VAE reference latent encoding.
- Configurable Qwen3-VL grounding resize modes (`none`, `downscale_only`, `normalize`, `clamp`) for each reference role.
- Pixel-space reference image transformations (`fit`, `crop`, `stretch`) matching VAE encoding and attention mask alignment.
- Per-role reference attention boost dials (`subject_boost`, `scene_boost`, `outfit_boost`) and hard/soft attention masks.
- Selectable single latent outputs (`empty`, `subject`, `scene`) with strict refusal of outfit latent options.
- Inpainting support with base image latent preparation, noise mask attached, grow, blur, and invert controls.
- Comprehensive CPU-safe unit test suite covering grounding, geometry, masks, latent selection, reference ordering, and patch isolation.
- Example ComfyUI workflow JSON files for all seven nodes.
