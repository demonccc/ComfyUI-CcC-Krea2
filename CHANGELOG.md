# Changelog

All notable changes to the `ComfyUI-CcC-Krea2` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CcC Krea2 - Text to Image` node (`CcCKrea2TextToImage`) for native text-to-image generation using CLIP tokenization, scheduled text encoding, and Empty SD3 Latent generation.
- Curated text-to-image example workflow `workflows/text_to_image.json`.
- `CcC Krea2 - LoRA Stack` node (`CcCKrea2LoRAStack`) for model-only LoRA stacking (up to 4 slots) with global/slot strengths, persistent slot loader instances, and prompt augmentation accumulators.
- `CcC Krea2 - LoRA Prompt Settings` node (`CcCKrea2LoRAPromptSettings`) for slot-level positive and negative prompt fragment configuration with `prepend` and `append` positioning.
- Immutable prompt augmentation module (`ccc_krea2/prompt_augmentation.py`) supporting deterministic text merging via double newline `\n\n` separators without mutating input strings.
- Optional `prompt_augmentation` socket on all 7 Krea 2 main nodes.
- Full LoRA Stack integration across production example workflow JSON files in `workflows/`.
- Unit test suites `tests/test_t2i_node.py`, `tests/test_prompt_augmentation.py`, and `tests/test_lora_nodes.py`, as well as updated interface and workflow tests.

### Fixed
- Standardized default `boost` widget value to `2.5` for `subject` role in `CcCKrea2ImageAdvancedSettings` across example workflows.
- Corrected documentation option lists for `grounding_resize_mode` and `reference_fit_mode` in `NODES.md`.

## [0.1.0-alpha] - 2026-07-30

### Added
- Experimental Alpha release of `ComfyUI-CcC-Krea2` suite.
- 4 general editing nodes (`Subject`, `Subject + Outfit`, `Subject + Scene`, `Subject + Scene + Outfit`).
- 3 inpainting nodes (`Inpaint`, `Inpaint Subject + Outfit`, `Inpaint Subject + Scene`).
- 2 advanced settings nodes (`Image Advanced Settings`, `Edit Advanced Settings`).
- `role_resolution_limit_mode` and `role_resolution_max_megapixels` controls in `Edit Advanced Settings` for clamping high-resolution role images while preserving aspect ratio.
- 6 curated example workflows in `workflows/` covering single-, dual-, and triple-reference editing, as well as Qwen-VL Vision Language Model integrations.
- Standard ComfyUI `DIFFUSION_MODEL` wrapper signature `(executor, x, timesteps, context, *wargs, **kwargs)` with closure transport.
- Applied `model.model.process_latent_in(...)` to every reference VAE latent.
- Native pixel-path reference geometry (`reference_fit_mode="fit"`) aligned to multiples of 16 without black canvas padding.
- Explicit inpaint base role selection per node.
- Pure PyTorch mask dilation via `F.max_pool2d` removing `scipy` dependency.
- Model-driven empty latent generation honoring `batch_size`.
- Mandatory VAE input across all 7 custom node definitions.
- Unit test suite in `tests/` covering mock integration, geometry, masks, dynamic prompt templates, resize method wiring, role resolution clamping, workflow files, and patch isolation.
