import { app } from "../../scripts/app.js";

const PROMPT_CREATOR_PRESETS = {
    enhance: `Analyze the user's instructions and the available visual context, then create a clear English I2I prompt for Krea2 Edit.

Treat the user's instructions as the authoritative guide for the desired result. Use the available visual context to understand and describe that result accurately.

Preserve the user's intent and constraints while improving clarity. Do not add unnecessary assumptions or unrelated details.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,

    create_from_image: `Analyze the reference edit image and the user's instructions, then create a clear English I2I prompt for Krea2 Edit.

Treat the reference edit image as the authoritative visual blueprint for the desired result. Build the prompt from what is visibly present in that image.

Use the user's instructions to determine what should be preserved, changed, emphasized, or adapted.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,

    create_from_theme: `Analyze the user's theme or idea and the available visual context, then create a clear English I2I prompt for Krea2 Edit.

Treat the user's theme or idea as the authoritative creative direction. Use the available visual context to build a coherent desired result that fits it.

Keep the prompt focused on the intended final image. Add only the visual detail needed to make the result clear and useful.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,
};


// Consolidated from the existing compactLoader frontend behavior.
function setWidgetDisabled(node, name, disabled) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (widget) widget.disabled = disabled;
}

// Use the stock ComfyUI widget renderers and interaction handling.
// Only toggle the same 'disabled' property used by the existing reference nodes.
function setupNativeLoraWidgets(node) {
    if (node.__cccNativeLoraReady) return;
    node.__cccNativeLoraReady = true;
    const slotCount = node.comfyClass === "CcCKrea2LoRAStackWithPrompt" ? 5 : 10;
    const byName = (name) => node.widgets?.find((widget) => widget.name === name);

    const refresh = () => {
        for (let i = 1; i <= slotCount; i++) {
            const enabled = byName(`lora_${i}_enabled`)?.value === true;
            for (const field of ["name", "strength", "prompt_position", "prompt"]) {
                setWidgetDisabled(node, `lora_${i}_${field}`, !enabled);
            }
        }
        node.setDirtyCanvas?.(true, true);
    };
    for (let i = 1; i <= slotCount; i++) {
        const widget = byName(`lora_${i}_enabled`);
        if (!widget) continue;
        const original = widget.callback;
        widget.callback = function (...args) {
            const result = original?.apply(this, args);
            refresh();
            return result;
        };
    }
    setTimeout(refresh, 20);
}

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;
        if (["CcCKrea2LoRALoader", "CcCKrea2LoRAStack", "CcCKrea2LoRAStackWithPrompt"].includes(node.comfyClass)) {
            setupNativeLoraWidgets(node);
        }

        if (node.comfyClass === "CcCKrea2VisualReference") {
            const semanticWidget = node.widgets?.find((w) => w.name === "semantic");
            const semanticResizeWidget = node.widgets?.find((w) => w.name === "semantic_resize");
            const fitWidget = node.widgets?.find((w) => w.name === "reference_fit");

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                const resizeEnabled = enabled && semanticResizeWidget?.value === true;
                setWidgetDisabled(node, "semantic_resize", !enabled);
                setWidgetDisabled(node, "semantic_grounding_px", !resizeEnabled);
                setWidgetDisabled(node, "semantic_resize_method", !resizeEnabled);
                setWidgetDisabled(node, "prompt_annotation", !enabled);
            };

            const updateFitState = () => {
                const mode = fitWidget?.value ?? "native";
                const crop = mode === "crop";
                setWidgetDisabled(node, "placement_grid", crop);
                setWidgetDisabled(node, "resize_method", mode !== "resize");
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

        if (node.comfyClass === "CcCKrea2VisionReference") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const processingWidget = node.widgets?.find((w) => w.name === "style_processing");
            const seedModeWidget = node.widgets?.find((w) => w.name === "guidance_seed_mode");

            const updateVisionReferenceState = () => {
                const styleMode = (modeWidget?.value ?? "semantic") === "style";
                if (processingWidget) {
                    processingWidget.disabled = !styleMode;
                    if (!styleMode) processingWidget.value = "full";
                }
                const seedMode = seedModeWidget?.value ?? "fixed";
                setWidgetDisabled(node, "guidance_seed", seedMode !== "fixed");
            };

            for (const widget of [modeWidget, seedModeWidget]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    updateVisionReferenceState();
                    return result;
                };
            }

            setTimeout(updateVisionReferenceState, 20);
        }

        if (node.comfyClass === "CcCKrea2TextToImage") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setWidgetDisabled(node, "preset_size", mode !== "preset");
                setWidgetDisabled(node, "width", mode !== "fixed");
                setWidgetDisabled(node, "height", mode !== "fixed");
                setWidgetDisabled(node, "geometry_policy", mode !== "fixed");
            };

            if (dimensionsWidget) {
                const originalCallback = dimensionsWidget.callback;
                dimensionsWidget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    updateDimensionsState();
                    return result;
                };
            }

            setTimeout(updateDimensionsState, 20);
        }

        if (node.comfyClass === "CcCKrea2Latent") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
            const contentWidget = node.widgets?.find((w) => w.name === "content");
            const contentFitWidget = node.widgets?.find((w) => w.name === "content_fit");
            const semanticWidget = node.widgets?.find((w) => w.name === "latent_semantic");

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setWidgetDisabled(node, "width", mode !== "fixed");
                setWidgetDisabled(node, "height", mode !== "fixed");
                setWidgetDisabled(node, "preset_size", mode !== "preset");
                setWidgetDisabled(node, "geometry_policy", mode === "preset");
            };

            const updateContentState = () => {
                const fromImage = contentWidget?.value === "from_image";
                setWidgetDisabled(node, "content_fit", !fromImage);
                setWidgetDisabled(node, "resize_method", !fromImage);
            };

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                setWidgetDisabled(node, "latent_semantic_instruction", !enabled);
                setWidgetDisabled(node, "latent_grounding_px", !enabled);
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
            // Existing prompt-creator title behavior consolidated here.
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            if (modeWidget) {
                const updateTitle = () => {
                    const mode = modeWidget.value ?? "enhance";
                    const current = String(node.title ?? "");
                    if (!current || current === "Krea2 CcC Edit Prompt Creator" ||
                        current.startsWith("Prompt Creator - ")) {
                        node.title = `Prompt Creator - ${mode}`;
                    }
                    node.setDirtyCanvas?.(true, true);
                };
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function (...args) {
                    const result = originalCallback?.apply(this, args);
                    updateTitle();
                    return result;
                };
                setTimeout(updateTitle, 120);
            }
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
