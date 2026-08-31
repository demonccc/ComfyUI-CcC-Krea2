# ComfyUI-CcC-Krea2

CcC nodes for Krea 2 generation and editing in ComfyUI.

## Current Edit Architecture

The Edit surface is intentionally split into three nodes:

- **Krea2 CcC Visual Reference**
- **Krea2 CcC Semantic Reference**
- **Krea2 CcC Edit**

The split keeps visual reference transport, semantic/style conditioning, and final edit orchestration independent while preserving the behavior already tested in the former Advanced Edit laboratory.

### Visual Reference

Use one node per visual Krea2 Edit reference and chain them in the exact physical reference order expected by the model.

For the bundled scene + subject test:

```text
scene -> subject
```

Each Visual Reference exposes:

- `boost`
- `rope_position`
- `semantic`
- `semantic_role`
- `instruction`
- `grounding_px`

`semantic_role` is optional. When `semantic = true` and `semantic_role` is empty, the reference behaves like the original positional Krea2 Edit conditioning: Qwen receives the image in order, without an extra semantic alias. When a value is provided, that exact text is attached to the corresponding vision block so the prompt can reference it semantically.

When `semantic = false`, `semantic_role`, `instruction`, and `grounding_px` are disabled in the UI and the reference is transported only through the visual/edit path.

### Semantic Reference

Use one node per Qwen-only semantic or style reference.

Available modes:

- `semantic_only`
- `style_direct`
- `style_indirect`

Existing controls are preserved:

- `instruction`
- `grounding_px`
- `processing`
- `fidelity`

### Edit

The Edit node owns the global edit contract:

- `positive_prompt`
- `negative_prompt`
- `aspect_ratio`
- `resolution`
- target latent semantic participation
- `batch_size`
- `apply_krea2_edit_patch`

Optional image sockets replace the old source selectors:

- `target_image`
- `grid_size_image`
- `grid_geometry_image`

Optional reference sockets:

- `visual_references`
- `semantic_references`

Visual references are resolved first. Semantic-only references follow them. Style references are kept last because they may expand into multiple physical Qwen image spans.

## Test Workflow

For now the repository intentionally contains a single edit workflow:

[`workflows/01_scene_subject.json`](workflows/01_scene_subject.json)

It reproduces the current scene + subject experiment:

- empty target latent
- scene as Visual Reference 1
- subject as Visual Reference 2
- scene boost `1.0`
- subject boost `4.0`
- both references participate in Qwen semantic conditioning
- semantic roles are `scene image` and `subject image`
- Subject supplies grid size
- Scene supplies grid geometry
- KSampler uses 8 steps

## Other Public Nodes

The package also keeps:

- `CcC Krea2 - LoRA Prompt Settings`
- `CcC Krea2 - LoRA Stack`
- `CcC Krea2 - Text to Image`

See [NODES.md](NODES.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [CHANGELOG.md](CHANGELOG.md).
