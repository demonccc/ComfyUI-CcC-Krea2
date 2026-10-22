"""Shared behavioral contracts for Vision modes used by Edit and T2I.

Mode names describe what the reference contributes, independent of the consumer.
The consumer remains responsible for how that information is conditioned.
"""

SEMANTIC_DIRECTIVE = (
    "Use the reference image as a subject/content guide for the SAME subject requested by the user's prompt. "
    "Preserve semantic content such as subject type, attributes, pose, action, objects, composition, and scene meaning, "
    "while the text prompt controls visual style and rendering. Do not add the reference subject as a second subject."
)

VISUAL_DIRECTIVE = (
    "Use this reference as direct visual guidance for the subject/result requested by the user's prompt. "
    "Preserve visible appearance and details, but do not duplicate the reference subject as an additional subject."
)

STYLE_DIRECTIVE = (
    "Use the reference image only as a style guide: adopt its color palette, lighting, texture, line work, shading, "
    "medium, tone, and overall mood, but do not copy or introduce its subjects, people, identities, clothing, objects, "
    "layout, pose, or composition."
)

CONTEXT_DIRECTIVE = (
    "Use this reference only for situation, action, framing, environment, spatial relationships, and scene context. "
    "Do not copy identities, clothing, exact appearance, or style, and do not introduce a reference person as an "
    "additional subject unless the user's prompt explicitly requires that extra role."
)


VISION_DIRECTIVES = {
    "semantic": SEMANTIC_DIRECTIVE,
    "visual": VISUAL_DIRECTIVE,
    "style": STYLE_DIRECTIVE,
    "context": CONTEXT_DIRECTIVE,
    "original": "",
}
