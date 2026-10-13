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
// LoRA stacks use ComfyUI's native BOOLEAN, COMBO, FLOAT and STRING widgets.
// Keep widget types, callbacks and workflow serialization untouched.
const NATIVE_LORA_NODES = new Set([
    "CcCKrea2LoRALoader",
    "CcCKrea2LoRAStack",
    "CcCKrea2LoRAStackWithPrompt",
]);

const T2I_LEGACY_ASPECT_RATIOS = new Set([
    "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "custom",
]);

const T2I_KREA_PRESETS = [
    ["1024 x 1024 | 1:1 | ~1.05 MP", 1024, 1024],
    ["1216 x 832 | ~3:2 | ~1.01 MP", 1216, 832],
    ["832 x 1216 | ~2:3 | ~1.01 MP", 832, 1216],
    ["1536 x 1024 | 3:2 | ~1.57 MP", 1536, 1024],
    ["1024 x 1536 | 2:3 | ~1.57 MP", 1024, 1536],
    ["1536 x 1152 | 4:3 | ~1.77 MP", 1536, 1152],
    ["1152 x 1536 | 3:4 | ~1.77 MP", 1152, 1536],
    ["2048 x 1536 | 4:3 | ~3.15 MP", 2048, 1536],
    ["1536 x 2048 | 3:4 | ~3.15 MP", 1536, 2048],
    ["2048 x 1152 | 16:9 | ~2.36 MP", 2048, 1152],
    ["1152 x 2048 | 9:16 | ~2.36 MP", 1152, 2048],
    ["2048 x 2048 | 1:1 | ~4.19 MP", 2048, 2048],
];

function setWidgetDisabled(node, name, disabled) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget) return;
    widget.disabled = disabled;
    widget.triggerDraw?.();
}

function t2iPresetDimensions(label) {
    const match = String(label ?? "").match(/^(\d+)\s*x\s*(\d+)/i);
    if (!match) return null;
    const width = Number(match[1]);
    const height = Number(match[2]);
    if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
    return [width, height];
}

function normalizeT2ICurrentWidgets(node) {
    const find = (name) => node.widgets?.find((w) => w.name === name);
    const dimensions = find("dimensions");
    const preset = find("preset_size");
    const width = find("width");
    const height = find("height");
    const policy = find("geometry_policy");
    const batch = find("batch_size");

    if (dimensions && !["preset", "fixed"].includes(String(dimensions.value ?? ""))) {
        dimensions.value = "preset";
    }

    const mode = String(dimensions?.value ?? "preset");
    if (mode === "preset") {
        const resolved = t2iPresetDimensions(preset?.value);
        if (resolved) {
            if (width) width.value = resolved[0];
            if (height) height.value = resolved[1];
        } else {
            if (preset) preset.value = T2I_KREA_PRESETS[0][0];
            if (width) width.value = T2I_KREA_PRESETS[0][1];
            if (height) height.value = T2I_KREA_PRESETS[0][2];
        }
    } else {
        if (width && (!Number.isFinite(Number(width.value)) || Number(width.value) < 16)) {
            width.value = 1024;
        }
        if (height && (!Number.isFinite(Number(height.value)) || Number(height.value) < 16)) {
            height.value = 1024;
        }
    }

    if (policy && !["nearest_krea_aspect", "preserve_aspect_krea_bounds"].includes(String(policy.value ?? ""))) {
        policy.value = "preserve_aspect_krea_bounds";
    }

    const batchValue = Number(batch?.value);
    if (batch && (!Number.isFinite(batchValue) || batchValue < 1 || batchValue > 64)) {
        batch.value = 1;
    }
}

function legacyT2IRatio(values) {
    const aspect = String(values?.[1] ?? "");
    if (!T2I_LEGACY_ASPECT_RATIOS.has(aspect)) return null;
    if (aspect !== "custom") {
        const [w, h] = aspect.split(":").map(Number);
        return w > 0 && h > 0 ? w / h : null;
    }

    // Old schemas:
    // v1: prompt, aspect, MP, batch, custom_w, custom_h, negative
    // v2: prompt, aspect, MP, batch, seed_mode, seed, detail, vision, custom_w, custom_h, negative
    const customIndex = values.length >= 11 ? 8 : 4;
    const w = Number(values[customIndex]);
    const h = Number(values[customIndex + 1]);
    return w > 0 && h > 0 ? w / h : 1.0;
}

function nearestT2IKreaPreset(values) {
    const ratio = legacyT2IRatio(values) ?? 1.0;
    const megapixels = Math.max(0.01, Number(values?.[2]) || 1.0);

    return T2I_KREA_PRESETS.reduce((best, candidate) => {
        const [, w, h] = candidate;
        const candidateRatio = w / h;
        const ratioError = Math.abs((candidateRatio / ratio) - 1.0);
        const mpError = Math.abs(((w * h) / 1_000_000) - megapixels);
        const score = [ratioError, mpError];

        if (!best) return { candidate, score };
        if (score[0] < best.score[0]) return { candidate, score };
        if (score[0] === best.score[0] && score[1] < best.score[1]) {
            return { candidate, score };
        }
        return best;
    }, null)?.candidate ?? T2I_KREA_PRESETS[0];
}

function migrateLegacyT2IWidgets(node, values) {
    if (!Array.isArray(values) || !T2I_LEGACY_ASPECT_RATIOS.has(String(values[1] ?? ""))) {
        return false;
    }

    const [presetLabel, presetWidth, presetHeight] = nearestT2IKreaPreset(values);
    const batchSize = Math.min(64, Math.max(1, Number(values[3]) || 1));
    const negativeIndex = values.length >= 11 ? 10 : 6;
    const negativePrompt = String(values[negativeIndex] ?? "");

    const assign = (name, value) => {
        const widget = node.widgets?.find((w) => w.name === name);
        if (widget) widget.value = value;
    };

    assign("dimensions", "preset");
    assign("preset_size", presetLabel);
    assign("width", presetWidth);
    assign("height", presetHeight);
    assign("geometry_policy", "preserve_aspect_krea_bounds");
    assign("batch_size", batchSize);
    assign("negative_prompt", negativePrompt);

    node.setDirtyCanvas?.(true, true);
    return true;
}

function setupLoraDisabledState(node, attempts = 0) {
    if (node.__cccLoraDisabledInstalled) return;
    const count = node.comfyClass === "CcCKrea2LoRAStackWithPrompt" ? 5 : 10;
    const find = (name) => node.widgets?.find((widget) => widget.name === name);

    if (!find("lora_1_enabled")) {
        if (attempts < 20) requestAnimationFrame(() => setupLoraDisabledState(node, attempts + 1));
        return;
    }

    node.__cccLoraDisabledInstalled = true;
    const refresh = () => {
        for (let i = 1; i <= count; i++) {
            const enabled = find(`lora_${i}_enabled`)?.value === true;
            for (const suffix of ["name", "strength", "prompt_position", "prompt"]) {
                const widget = find(`lora_${i}_${suffix}`);
                if (widget) widget.disabled = !enabled;
            }
        }
        node.setDirtyCanvas?.(true, true);
    };

    for (let i = 1; i <= count; i++) {
        const toggle = find(`lora_${i}_enabled`);
        if (!toggle) continue;
        const originalCallback = toggle.callback;
        toggle.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            refresh();
            return result;
        };
    }

    const originalConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = originalConfigure?.apply(this, args);
        refresh();
        return result;
    };
    refresh();
}

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;
        if (NATIVE_LORA_NODES.has(node.comfyClass)) setupLoraDisabledState(node);

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

        if (node.comfyClass === "CcCKrea2VisionReference") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const processingWidget = node.widgets?.find((w) => w.name === "style_processing");
            const seedModeWidget = node.widgets?.find((w) => w.name === "guidance_seed_mode");

            const updateVisionReferenceState = () => {
                const mode = modeWidget?.value ?? "semantic";
                const styleMode = mode === "style";
                const contextMode = mode === "context";
                if (processingWidget) {
                    processingWidget.disabled = !styleMode;
                    if (!styleMode) processingWidget.value = "full";
                }

                const detailWidget = node.widgets?.find((w) => w.name === "guidance_detail");
                const visionWidget = node.widgets?.find((w) => w.name === "vision_guidance");
                if (contextMode) {
                    if (detailWidget) detailWidget.value = "balanced";
                    if (visionWidget) visionWidget.value = "balanced";
                }

                setWidgetDisabled(node, "guidance_seed_mode", false);
                setWidgetDisabled(node, "guidance_detail", contextMode);
                setWidgetDisabled(node, "vision_guidance", contextMode);

                const seedMode = seedModeWidget?.value ?? "fixed";
                setWidgetDisabled(node, "guidance_seed", seedMode !== "fixed");
                node.setDirtyCanvas?.(true, true);
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
            const presetWidget = node.widgets?.find((w) => w.name === "preset_size");

            const updateDimensionsState = () => {
                normalizeT2ICurrentWidgets(node);
                const mode = dimensionsWidget?.value ?? "preset";
                setWidgetDisabled(node, "preset_size", mode !== "preset");
                setWidgetDisabled(node, "width", mode !== "fixed");
                setWidgetDisabled(node, "height", mode !== "fixed");
                setWidgetDisabled(node, "geometry_policy", mode !== "fixed");
                node.setDirtyCanvas?.(true, true);
            };

            for (const widget of [dimensionsWidget, presetWidget]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    updateDimensionsState();
                    return result;
                };
            }

            // Workflows saved before the shared Krea geometry refactor used:
            // prompt + aspect_ratio + megapixels + ...
            // Migrate raw legacy values when available, then always normalize
            // the current schema because ComfyUI validates disabled widgets too.
            const originalConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const legacyValues = Array.isArray(info?.widgets_values)
                    ? [...info.widgets_values]
                    : null;
                const result = originalConfigure?.apply(this, arguments);
                if (legacyValues) migrateLegacyT2IWidgets(node, legacyValues);
                normalizeT2ICurrentWidgets(node);
                updateDimensionsState();
                return result;
            };

            setTimeout(() => {
                const currentMode = String(dimensionsWidget?.value ?? "");
                if (!["preset", "fixed"].includes(currentMode)) {
                    const syntheticLegacy = node.widgets?.map((w) => w.value) ?? [];
                    migrateLegacyT2IWidgets(node, syntheticLegacy);
                }
                normalizeT2ICurrentWidgets(node);
                updateDimensionsState();
            }, 20);
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

    }
});
