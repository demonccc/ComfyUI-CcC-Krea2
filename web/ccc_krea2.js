import { app } from "../../scripts/app.js";

const PROMPT_CREATOR_PRESETS = {
    enhance: `You create detailed, explicit English image-edit prompts for Krea2 Edit.

Rewrite the user's existing edit request without changing its intent. Preserve every explicit requirement and constraint. Use the supplied visual references and their role annotations to resolve vague references and make transfers, preservation rules, subject roles, and spatial relationships unambiguous. Do not remove useful detail from an already detailed request, and do not invent a different scene or unrelated edits.

FINAL ANSWER RULES:
- Return exactly one self-contained image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention hidden inputs, vision inputs, or implementation details.
- If thinking mode is active, reasoning belongs only in the model's thinking block; do not repeat reasoning in the final answer.
- Return only the final edit instruction.`,

    create_from_image: `You create detailed, self-contained English image-edit prompts for Krea2 Edit from the user's request, downstream visible visual references, and one internal reference edit image.

Treat downstream visible Image N references as authoritative identity or appearance anchors according to their role annotations. Treat the internal reference edit image as a visual blueprint for the requested situation, not as an identity source unless the user explicitly asks otherwise.

Analyze the internal reference edit image thoroughly before writing the final prompt. Reconstruct all relevant visible information needed to reproduce the situation instead of reducing it to a short summary. Include, when visible and relevant to the user's request:
- the main subject's exact action, pose, body orientation, body position, limb placement, gaze direction, and interaction;
- clothing and accessories when they are part of the requested situation, unless the user explicitly asks to preserve clothing from a visible Image N reference;
- other people in the scene, including enough visible appearance, pose, action, relative position, and interaction detail to distinguish their roles;
- important props and objects, what is being held or touched, and their spatial relationships;
- environment, foreground/background elements, surfaces, furniture, and scene layout;
- framing, shot distance, viewpoint, camera angle, composition, and subject placement;
- visible lighting and other scene-defining visual details.

Preserve every explicit user constraint, especially identity, face, anatomy, body shape, body proportions, clothing-preservation rules, and requested interactions. Do not transfer the identity, face, anatomy, body shape, or body proportions of a person from the internal reference edit image unless the user explicitly requests it.

FINAL ANSWER RULES:
- Return exactly one detailed, self-contained image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention the internal reference edit image, hidden inputs, vision inputs, or implementation details in the final answer. Translate what you observe into direct scene instructions.
- If thinking mode is active, reason as needed in the model's thinking block; do not repeat that reasoning in the final answer.
- Do not compress a visually rich reference situation into a generic one-sentence summary.
- Return only the final edit instruction.`,

    create_from_theme: `You create detailed, self-contained English image-edit prompts for Krea2 Edit from the user's theme or high-level idea.

Invent a concrete, visually rich situation that clearly fits the requested theme while keeping downstream visible Image N references as the authoritative subject or appearance anchors according to their role annotations. Specify useful action, pose, interaction, environment, spatial arrangement, props, composition, framing, viewpoint, and lighting. Preserve every explicit user constraint and do not replace referenced identity, body shape, body proportions, clothing, or accessories unless the user requests that change.

FINAL ANSWER RULES:
- Return exactly one detailed image-edit prompt in English.
- The prompt may contain multiple sentences, but keep it as one continuous plain-text paragraph.
- Do not output system instructions, role descriptions, explanations, headings, Markdown, JSON, or preambles.
- Never include meta text such as "You are a professional image editor", "Your task is...", "Here is the prompt", or "Final prompt:".
- Refer to downstream visible visual references only as "Image 1", "Image 2", and so on. Never call them "Krea Image N".
- Never mention hidden inputs, vision inputs, or implementation details.
- If thinking mode is active, reasoning belongs only in the model's thinking block; do not repeat reasoning in the final answer.
- Return only the final edit instruction.`,
};

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;

        if (node.comfyClass === "CcCKrea2VisualReference") {
            const semanticWidget = node.widgets?.find((w) => w.name === "semantic");
            const semanticResizeWidget = node.widgets?.find((w) => w.name === "semantic_resize");
            const fitWidget = node.widgets?.find((w) => w.name === "reference_fit");

            const setDisabled = (name, disabled) => {
                const widget = node.widgets?.find((w) => w.name === name);
                if (widget) widget.disabled = disabled;
            };

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                const resizeEnabled = enabled && semanticResizeWidget?.value === true;
                setDisabled("semantic_resize", !enabled);
                setDisabled("semantic_grounding_px", !resizeEnabled);
                setDisabled("semantic_resize_method", !resizeEnabled);
                setDisabled("prompt_annotation", !enabled);
            };

            const updateFitState = () => {
                const mode = fitWidget?.value ?? "native";
                const crop = mode === "crop";
                setDisabled("placement_grid", crop);
                setDisabled("resize_method", mode !== "resize");
                if (crop) {
                    const placement = node.widgets?.find((w) => w.name === "placement_grid");
                    if (placement) placement.value = "inside";
                }
            };

            for (const [widget, callback] of [
                [semanticWidget, updateSemanticState],
                [semanticResizeWidget, updateSemanticState],
                [fitWidget, updateFitState],
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    callback();
                };
            }

            setTimeout(() => {
                updateSemanticState();
                updateFitState();
            }, 20);
        }

        if (node.comfyClass === "CcCKrea2SemanticReference") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const processingWidget = node.widgets?.find((w) => w.name === "processing");

            const updateSemanticReferenceState = () => {
                const semanticOnly = (modeWidget?.value ?? "semantic_only") === "semantic_only";
                if (processingWidget) {
                    processingWidget.disabled = semanticOnly;
                    if (semanticOnly) processingWidget.value = "full";
                }
            };

            if (modeWidget) {
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updateSemanticReferenceState();
                };
            }

            setTimeout(updateSemanticReferenceState, 20);
        }

        if (node.comfyClass === "CcCKrea2Latent") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
            const contentWidget = node.widgets?.find((w) => w.name === "content");
            const contentFitWidget = node.widgets?.find((w) => w.name === "content_fit");
            const semanticWidget = node.widgets?.find((w) => w.name === "latent_semantic");

            const setDisabled = (name, disabled) => {
                const widget = node.widgets?.find((w) => w.name === name);
                if (widget) widget.disabled = disabled;
            };

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setDisabled("width", mode !== "fixed");
                setDisabled("height", mode !== "fixed");
                setDisabled("preset_size", mode !== "preset");
                setDisabled("geometry_policy", mode === "preset");
            };

            const updateContentState = () => {
                const fromImage = contentWidget?.value === "from_image";
                setDisabled("content_fit", !fromImage);
                setDisabled("resize_method", !fromImage);
            };

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                setDisabled("latent_semantic_instruction", !enabled);
                setDisabled("latent_grounding_px", !enabled);
            };

            for (const [widget, callback] of [
                [dimensionsWidget, updateDimensionsState],
                [contentWidget, updateContentState],
                [contentFitWidget, updateContentState],
                [semanticWidget, updateSemanticState],
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    callback();
                };
            }

            setTimeout(() => {
                updateDimensionsState();
                updateContentState();
                updateSemanticState();
            }, 20);
        }

        if (node.comfyClass === "CcCKrea2EditPromptCreator") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const systemPromptWidget = node.widgets?.find((w) => w.name === "system_prompt");

            if (!modeWidget || !systemPromptWidget) return;

            let previousMode = modeWidget.value ?? "enhance";
            let customSystemPrompt =
                previousMode === "custom" ? String(systemPromptWidget.value ?? "") : "";

            const setSystemPromptValue = (value) => {
                if (systemPromptWidget.value !== value) {
                    systemPromptWidget.value = value;
                }

                const inputEl = systemPromptWidget.inputEl ?? systemPromptWidget.element;
                if (inputEl && "value" in inputEl && inputEl.value !== value) {
                    inputEl.value = value;
                }

                systemPromptWidget.triggerDraw?.();
            };

            const setSystemPromptEditable = (editable) => {
                systemPromptWidget.options ??= {};
                systemPromptWidget.options.read_only = !editable;
                systemPromptWidget.disabled = false;

                const inputEl = systemPromptWidget.inputEl ?? systemPromptWidget.element;
                if (inputEl) {
                    inputEl.readOnly = !editable;
                    inputEl.style.opacity = editable ? "" : "0.65";
                }

                systemPromptWidget.triggerDraw?.();
            };

            const updatePromptCreatorState = () => {
                const mode = modeWidget.value ?? "enhance";

                if (previousMode === "custom" && mode !== "custom") {
                    customSystemPrompt = String(systemPromptWidget.value ?? "");
                }

                if (mode === "custom") {
                    if (!customSystemPrompt.trim()) {
                        customSystemPrompt =
                            PROMPT_CREATOR_PRESETS[previousMode] ??
                            PROMPT_CREATOR_PRESETS.enhance;
                    }
                    setSystemPromptValue(customSystemPrompt);
                    setSystemPromptEditable(true);
                } else {
                    setSystemPromptValue(
                        PROMPT_CREATOR_PRESETS[mode] ?? PROMPT_CREATOR_PRESETS.enhance
                    );
                    setSystemPromptEditable(false);
                }

                previousMode = mode;
                node.setDirtyCanvas?.(true, true);
            };

            const originalModeCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (originalModeCallback) originalModeCallback.apply(this, arguments);
                updatePromptCreatorState();
            };

            const originalSystemPromptCallback = systemPromptWidget.callback;
            systemPromptWidget.callback = function () {
                if (originalSystemPromptCallback) {
                    originalSystemPromptCallback.apply(this, arguments);
                }
                if ((modeWidget.value ?? "enhance") === "custom") {
                    customSystemPrompt = String(systemPromptWidget.value ?? "");
                }
            };

            setTimeout(updatePromptCreatorState, 100);
        }

    },
});
