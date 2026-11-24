import { app } from "../../scripts/app.js";

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

// The Python nodes expose only Visual and Semantic. Saved workflows can carry
// obsolete combo choices, so constrain both the values and the current selection.
function normalizeVisionMode(widget) {
    if (!widget) return;
    if (widget.options) widget.options.values = ["visual", "semantic"];
    if (widget.value === "reference") widget.value = "visual";
    else if (!["visual", "semantic"].includes(widget.value)) widget.value = "visual";
}

function setWidgetDisabled(node, name, disabled) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget) return;
    const next = Boolean(disabled);
    widget.disabled = next;
    // DOM-backed STRING/textarea widgets are not canvas widgets. Keep their
    // actual input element in sync with LiteGraph/Nodes 2.0 disabled state.
    if (widget.element) {
        widget.element.disabled = next;
        widget.element.readOnly = next;
        widget.element.querySelectorAll?.("input, textarea, select, button")
            .forEach((element) => { element.disabled = next; });
    }
    widget.triggerDraw?.();
}

// A fixed 20ms timer is not a reliable widget-initialization barrier, especially
// when a saved workflow creates nodes before their widgets or DOM controls exist.
// Refresh immediately, on the next frames, and after ComfyUI restores node values.
function scheduleWidgetRefresh(node, refresh) {
    refresh();
    requestAnimationFrame(() => {
        refresh();
        requestAnimationFrame(refresh);
    });
    // ComfyUI may attach restored widgets after nodeCreated. Refresh after
    // graph insertion too; avoid relying solely on transient timers.
    const originalAdded = node.onAdded;
    node.onAdded = function (...args) {
        const result = originalAdded?.apply(this, args);
        refresh();
        requestAnimationFrame(refresh);
        return result;
    };
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
    const count = 5;
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
                if (widget) setWidgetDisabled(node, widget.name, !enabled);
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

        if (node.comfyClass === "CcCKrea2LayoutCell") {
            const find = (name) => node.widgets?.find(w => w.name === name);
            const refresh = () => {
                const placement = find("placement")?.value ?? "cell";
                const fit = find("fit")?.value ?? "contain";
                setWidgetDisabled(node, "row", placement === "full_column");
                setWidgetDisabled(node, "column", placement === "full_row");
                const hasPosition = fit === "crop" || fit === "fit";
                setWidgetDisabled(node, "resize_percent", fit !== "crop");
                for (const name of ["horizontal_position", "vertical_position", "offset_x", "offset_y"]) {
                    setWidgetDisabled(node, name, !hasPosition);
                }
                node.setDirtyCanvas?.(true, true);
            };
            for (const name of ["placement", "fit"]) {
                const widget = find(name);
                if (!widget) continue;
                const old = widget.callback;
                widget.callback = function (...args) {
                    const result = old?.apply(this, args);
                    refresh();
                    return result;
                };
            }
            const oldConfigure = node.onConfigure;
            node.onConfigure = function (...args) {
                const result = oldConfigure?.apply(this, args);
                refresh();
                return result;
            };
            scheduleWidgetRefresh(node, refresh);
        }

        if (node.comfyClass === "CcCKrea2PaintPrepare") {
            // Saved Paint Prepare workflows used to start with geometry_mode.
            // Geometry is now always padding; shift the saved values by one.
            const originalConfigure = node.onConfigure;
            node.onConfigure = function (info, ...rest) {
                const result = originalConfigure?.call(this, info, ...rest);
                const saved = info?.widgets_values;
                if (Array.isArray(saved) && (saved[0] === "pad" || saved[0] === "crop")) {
                    const values = saved.slice(1);
                    for (let i = 0; i < (node.widgets?.length ?? 0) && i < values.length; i++) {
                        node.widgets[i].value = values[i];
                    }
                    node.setDirtyCanvas?.(true, true);
                }
                return result;
            };
        }

        if (node.comfyClass === "CcCKrea2Mask") {
            const find = (name) => node.widgets?.find((w) => w.name === name);
            const refresh = () => {
                const usingMask = find("source")?.value === "mask";
                setWidgetDisabled(node, "prompt", usingMask);
                setWidgetDisabled(node, "confidence", usingMask);
                setWidgetDisabled(node, "unload_sam_model", usingMask);
                node.setDirtyCanvas?.(true, true);
            };
            const source = find("source");
            if (source) {
                const original = source.callback;
                source.callback = function (...args) {
                    const result = original?.apply(this, args);
                    refresh();
                    return result;
                };
            }
            const originalConfigure = node.onConfigure;
            node.onConfigure = function (info, ...rest) {
                const result = originalConfigure?.call(this, info, ...rest);
                const saved = info?.widgets_values;
                if (Array.isArray(saved) && typeof saved[4] === "boolean" &&
                    typeof saved[5] === "string") {
                    // Previous order: unload, composite_color, [invert].
                    // Current order: composite_color, invert, unload (last).
                    const color = find("composite_color");
                    const invert = find("invert");
                    const unload = find("unload_sam_model");
                    if (color) color.value = saved[5];
                    if (invert) invert.value = typeof saved[6] === "boolean" ? saved[6] : false;
                    if (unload) unload.value = saved[4];
                }
                refresh();
                return result;
            };
            scheduleWidgetRefresh(node, refresh);
        }

        if (node.comfyClass === "CcCKrea2Reference") {
            const find = (name) => node.widgets?.find((w) => w.name === name);
            const refresh = () => {
                normalizeVisionMode(find("vision_mode"));
                const visual = find("vae_rope_enable")?.value === true;
                const vision = find("vision_enable")?.value === true;
                for (const widget of node.widgets ?? []) {
                    if (widget.name.startsWith("vae_rope_") && widget.name !== "vae_rope_enable") {
                        setWidgetDisabled(node, widget.name, !visual);
                    } else if (widget.name.startsWith("vision_") && widget.name !== "vision_enable") {
                        setWidgetDisabled(node, widget.name, !vision);
                    }
                }
                const fit = find("vae_rope_reference_fit")?.value ?? "native";
                setWidgetDisabled(node, "vae_rope_placement_grid", !visual || fit === "crop");
                setWidgetDisabled(node, "vae_rope_rope_geometry", !visual || fit === "native");
                setWidgetDisabled(node, "vision_max_edge", !vision || find("vision_resize")?.value !== true);
                setWidgetDisabled(node, "vision_instruction", !vision);
                setWidgetDisabled(node, "vae_rope_prompt_annotation", !visual);
                node.setDirtyCanvas?.(true, true);
            };
            for (const name of ["vae_rope_enable", "vae_rope_reference_fit",
                                "vision_enable", "vision_resize", "vision_mode", "vision_guidance_seed_mode"]) {
                const widget = find(name);
                if (!widget) continue;
                const original = widget.callback;
                widget.callback = function (...args) {
                    const result = original?.apply(this, args);
                    refresh();
                    return result;
                };
            }
            const configure = node.onConfigure;
            node.onConfigure = function (...args) {
                const result = configure?.apply(this, args);
                refresh();
                return result;
            };
            scheduleWidgetRefresh(node, refresh);
        }

        if (node.comfyClass === "CcCKrea2VisualReference") {
            const semanticWidget = node.widgets?.find((w) => w.name === "semantic");
            const semanticResizeWidget = node.widgets?.find((w) => w.name === "semantic_resize");
            const fitWidget = node.widgets?.find((w) => w.name === "reference_fit");

            const setDisabled = (name, disabled) => setWidgetDisabled(node, name, disabled);

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                const resizeEnabled = enabled && semanticResizeWidget?.value === true;
                setDisabled("semantic_resize", !enabled);
                setDisabled("semantic_grounding_px", !resizeEnabled);
                setDisabled("prompt_annotation", !enabled);
            };

            const updateFitState = () => {
                const mode = fitWidget?.value ?? "native";
                const crop = mode === "crop";
                setDisabled("placement_grid", crop);
                setDisabled("rope_geometry", mode === "native");
                if (mode === "native") {
                    const geometry = node.widgets?.find((w) => w.name === "rope_geometry");
                    if (geometry) geometry.value = "source";
                }
                if (crop) {
                    const placement = node.widgets?.find((w) => w.name === "placement_grid");
                    if (placement) placement.value = "inside";
                }
            };

            for (const [widget, callback] of [
                [semanticWidget, updateSemanticState],
                ...["latent_vision_resize", "latent_vision_mode", "latent_vision_guidance_seed_mode"]
                    .map(name => [node.widgets?.find(w => w.name === name), updateSemanticState]),
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

            scheduleWidgetRefresh(node, () => {
                updateSemanticState();
                updateFitState();
                node.setDirtyCanvas?.(true, true);
            });
        }

        if (node.comfyClass === "CcCKrea2VisionReference") {
            const refresh = () => {
                normalizeVisionMode(node.widgets?.find(w => w.name === "mode"));
                node.setDirtyCanvas?.(true, true);
            };
            const originalConfigure = node.onConfigure;
            node.onConfigure = function (...args) {
                const result = originalConfigure?.apply(this, args);
                refresh();
                return result;
            };
            scheduleWidgetRefresh(node, refresh);
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

            scheduleWidgetRefresh(node, () => {
                const currentMode = String(dimensionsWidget?.value ?? "");
                if (!["preset", "fixed"].includes(currentMode)) {
                    const syntheticLegacy = node.widgets?.map((w) => w.value) ?? [];
                    migrateLegacyT2IWidgets(node, syntheticLegacy);
                }
                normalizeT2ICurrentWidgets(node);
                updateDimensionsState();
            });
        }

        if (node.comfyClass === "CcCKrea2Latent") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
            const contentWidget = node.widgets?.find((w) => w.name === "content");
            const contentFitWidget = node.widgets?.find((w) => w.name === "content_fit");
            const visionEnableWidget = node.widgets?.find((w) => w.name === "latent_vision_enable");

            const setDisabled = (name, disabled) => setWidgetDisabled(node, name, disabled);

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
            };

            const updateVisionState = () => {
                // Empty target latents have no source image for Vision analysis.
                // Clear the switch, rather than just greying it out while true.
                if (contentWidget?.value !== "from_image" && visionEnableWidget) {
                    visionEnableWidget.value = false;
                }
                normalizeVisionMode(node.widgets?.find(w => w.name === "latent_vision_mode"));
                const enabled = visionEnableWidget?.value === true && contentWidget?.value === "from_image";
                // The enable switch must remain interactive when content comes
                // from an image. Do not disable it with the dependent controls.
                for (const widget of node.widgets ?? []) {
                    if (widget.name.startsWith("latent_vision_") &&
                        widget.name !== "latent_vision_enable") {
                        setWidgetDisabled(node, widget.name, !enabled);
                    }
                }
                setWidgetDisabled(node, "latent_vision_enable",
                    contentWidget?.value !== "from_image");
                setWidgetDisabled(node, "latent_vision_max_edge", !enabled ||
                    node.widgets?.find(w => w.name === "latent_vision_resize")?.value !== true);

                node.setDirtyCanvas?.(true, true);
            };

            for (const [widget, callback] of [
                [dimensionsWidget, updateDimensionsState],
                [contentWidget, () => { updateContentState(); updateVisionState(); }],
                [contentFitWidget, updateContentState],
                [visionEnableWidget, updateVisionState],
                ...["latent_vision_resize", "latent_vision_mode", "latent_vision_guidance_seed_mode"]
                    .map(name => [node.widgets?.find(w => w.name === name), updateVisionState]),
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    callback();
                };
            }

            const originalConfigure = node.onConfigure;
            node.onConfigure = function (...args) {
                const result = originalConfigure?.apply(this, args);
                updateDimensionsState();
                updateContentState();
                updateVisionState();
                return result;
            };

            scheduleWidgetRefresh(node, () => {
                updateDimensionsState();
                updateContentState();
                updateVisionState();
                node.setDirtyCanvas?.(true, true);
            });
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


    }
});
