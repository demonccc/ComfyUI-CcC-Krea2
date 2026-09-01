# Architecture

## Edit Pipeline

The editing surface is split into four responsibilities:

```text
IMAGE -> Krea2 CcC Visual Reference --+
                                       |
IMAGE -> Krea2 CcC Visual Reference --+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
                                       |          ^
IMAGE -> Krea2 CcC Semantic Reference +          |
                                                  |
Images / VAE -> Krea2 CcC Latent ----------------+
```

## Visual References

Visual Reference nodes are ordered and append-only. For the current two-reference test contract the order is:

```text
scene -> subject
```

That order becomes both the Qwen physical image order for visual references and the Krea2 visual-reference order. Optional `semantic_role` metadata names a visual reference for Qwen without changing its physical order.

## Semantic References

Semantic Reference nodes add Qwen-only semantic or style context. Semantic-only references stay before style references. Style references remain last because a logical style reference may expand into several physical Qwen images.

## Latent

`Krea2 CcC Latent` owns target construction independently from the Edit node. It preserves the former Edit behavior for:

- empty target latent
- VAE image-init target latent
- explicit aspect ratio and megapixel resolution
- `from source` grid size
- `from source` grid geometry
- batch size
- optional semantic reinterpretation of the target image

`grid_size_image` and `grid_geometry_image` remain independent. One image can define the target pixel budget while another defines the target aspect ratio.

When `latent_semantic` is enabled, the target image, instruction, and grounding size are stored with the latent as CcC metadata. Edit consumes that metadata and inserts the target semantic reference after visual references, exactly where the former integrated Edit path inserted it.

## Edit

`Krea2 CcC Edit` receives a pre-built `LATENT`; it no longer receives target/grid images or geometry controls.

The final conditioning order is:

```text
visual references -> latent semantic -> semantic-only references -> style references
```

The latent is then passed unchanged into the Krea2 Edit orchestrator, which derives target geometry from its tensor shape, prepares visual reference VAE latents, builds Qwen conditioning, and applies the optional Krea2 Edit model patch.
