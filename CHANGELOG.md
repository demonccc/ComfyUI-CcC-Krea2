# Changelog

## Unreleased

### Changed
- Removed Easy Edit and the previous simple/Advanced Edit public surfaces.
- Split the former Edit Advanced laboratory into `Krea2 CcC Visual Reference`, `Krea2 CcC Semantic Reference`, and `Krea2 CcC Edit`.
- Visual references now keep physical Krea2 reference order separately from optional Qwen semantic naming.
- Added optional `semantic_role` to Visual Reference. Empty keeps positional Krea2 Edit behavior; non-empty attaches the role to the corresponding Qwen vision block.
- `semantic_role`, `instruction`, and `grounding_px` are disabled and ignored when Visual Reference semantic participation is disabled.
- Replaced the old subject/scene/outfit/style source selectors for target geometry with direct `target_image`, `grid_size_image`, and `grid_geometry_image` sockets.
- Removed old Easy Edit workflows and kept only `workflows/01_scene_subject.json` as the current split-node test workflow.
