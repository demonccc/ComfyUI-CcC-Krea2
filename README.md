# ComfyUI-CcC-Krea2

CcC nodes for Krea 2 generation and editing in ComfyUI.

## Current Edit Architecture

The Edit surface is split into four nodes:

- **Krea2 CcC Visual Reference**
- **Krea2 CcC Semantic Reference**
- **Krea2 CcC Latent**
- **Krea2 CcC Edit**

The split keeps visual references, semantic/style references, target latent construction, and final edit orchestration independent.

### Visual Reference

Use one node per visual Krea2 Edit reference and chain them in the exact physical reference order expected by the model. For the bundled scene + subject test:

```text
scene -> subject
```

`semantic_role` is optional. With `semantic = true` and an empty role, Qwen keeps positional Krea2 Edit behavior. With a value, that text identifies the corresponding image semantically. With `semantic = false`, semantic role/instruction/grounding are ignored.

### Semantic Reference

Use one node per Qwen-only semantic or style reference. Existing modes remain `semantic_only`, `style_direct`, and `style_indirect`.

### Latent

`Krea2 CcC Latent` now owns everything that previously built the target latent inside Edit:

- empty latent generation
- image-initialized latent through VAE
- `aspect_ratio`
- `resolution`
- independent `grid_size_image` and `grid_geometry_image`
- `batch_size`
- latent semantic reinterpretation (`latent_semantic`, instruction, grounding size)

When latent semantic is enabled, the target image and its semantic instruction travel with the LATENT metadata and are consumed by Edit in the same reference ordering position used before the split.

### Edit

`Krea2 CcC Edit` now receives a pre-built `LATENT` and owns only the final edit contract:

- `MODEL`
- `CLIP`
- `VAE`
- `LATENT`
- positive/negative prompts
- Krea2 Edit patch toggle
- optional visual reference chain
- optional semantic reference chain

Reference ordering remains:

```text
visual references -> latent semantic (when enabled) -> semantic-only references -> style references
```

## Test Workflow

For now the repository intentionally contains a single edit workflow:

[`workflows/01_scene_subject.json`](workflows/01_scene_subject.json)

It uses scene -> subject visual references, Subject for grid size, Scene for grid geometry, an empty latent, and passes the generated latent into Edit. KSampler remains at 8 steps.

## Other Public Nodes

The package also keeps:

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

See [NODES.md](NODES.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [CHANGELOG.md](CHANGELOG.md).
