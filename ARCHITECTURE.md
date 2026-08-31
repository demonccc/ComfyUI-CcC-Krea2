# Architecture

## Edit Pipeline

The editing surface is split into three responsibilities.

```text
IMAGE -> Krea2 CcC Visual Reference --+
                                       |
IMAGE -> Krea2 CcC Visual Reference --+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
                                       |
IMAGE -> Krea2 CcC Semantic Reference +
```

## Visual References

Visual Reference nodes are declarative. They keep the raw image plus the controls that previously lived in `reference_1_*` and `reference_2_*`.

Chaining is append-only and order-sensitive. For the two-reference identity/body edit contract used by the current test workflow:

```text
scene -> subject
```

That order becomes both:

```text
Qwen physical image order: scene, subject
Krea2 visual reference order: scene, subject
```

The visual path therefore does not infer `scene` or `subject` from socket names.

### Semantic naming

A Visual Reference may also participate in Qwen Vision.

- With an empty `semantic_role`, the image is represented positionally, matching the original Krea2 Edit behavior.
- With a non-empty `semantic_role`, the exact text is attached to that image's Qwen vision block.
- With `semantic = false`, the image remains a visual edit reference but is omitted from Qwen Vision.

This separates physical reference ordering from optional semantic naming.

## Semantic References

Semantic Reference nodes carry the controls previously exposed as `semantic_1_*` and `semantic_2_*`.

`semantic_only` produces a non-appearance edit-path reference for Qwen.

`style_direct` and `style_indirect` produce style references using the existing `processing` and `fidelity` behavior.

Style references are appended after every visual and semantic-only reference because one logical style reference may expand into multiple physical Qwen images.

## Edit

Edit owns target creation and final orchestration.

The old `subject/scene/outfit/style` source selectors are no longer required. The source is represented directly by the connected image socket:

- `target_image`
- `grid_size_image`
- `grid_geometry_image`

When `target_image` is not connected, the target latent is empty.

`grid_size_image` and `grid_geometry_image` remain independent. This preserves the experiment where Subject provides the target pixel budget while Scene provides the target aspect ratio.

The final orchestration still uses:

```text
reference_method = krea2_edit
```

and `apply_krea2_edit_patch` controls whether the model patch is applied.

## Repository Workflow Policy

During this refactor only one workflow is kept:

```text
workflows/01_scene_subject.json
```

It is the canonical test bed for the split nodes until their contracts are stable.
