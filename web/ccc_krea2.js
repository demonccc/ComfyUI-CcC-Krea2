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
            setTimeout(refresh, 20);
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
            node.onConfigure = function (...args) {
                const result = originalConfigure?.apply(this, args);
                refresh();
                return result;
            };
            setTimeout(refresh, 20);
        }

        if (node.comfyClass === "CcCKrea2Reference") {
            const find = (name) => node.widgets?.find((w) => w.name === name);
            const restoreReferenceWidgets = (saved) => {
                const values = saved?.widgets_values;
                if (!Array.isArray(values)) return;
                const widgets = (node.widgets ?? []).filter(w => w.serialize !== false);
                const current = widgets.map(w => w.name);
                const old = current.filter(n => n !== "vae_rope_prompt_annotation");
                const annotationIndex = old.indexOf("vision_vision_fidelity");
                if (annotationIndex < 0) return;
                old.splice(annotationIndex, 0, "vision_reference_annotation");

                // Saved values are positional, not keyed. The old schema moved
                // the annotation from Vision to VAE/RoPE without retaining its
                // original position; detect the layout BEFORE Comfy restores it.
                const modes = ["reference", "semantic", "visual", "style", "context",
                    "native", "none", "guided"];
                const valid = (name, value) => {
                    if (name === "vision_mode") return modes.includes(value);
                    if (name === "vision_enable" || name === "vae_rope_enable" ||
                        name === "vision_resize") return typeof value === "boolean";
                    if (name === "vision_max_edge") return Number.isInteger(value) && value >= 384 && value <= 2048;
                    if (name === "vision_guidance_seed") return Number.isInteger(value) && value >= 0;
                    if (name === "vae_rope_boost") return typeof value === "number" && Number.isFinite(value);
                    if (name === "vision_guidance_seed_mode") return ["fixed", "random", "increment", "decrement"].includes(value);
                    return true;
                };
                const score = (names) => {
                    if (values.length !== names.length) return -1;
                    return ["vision_mode", "vision_enable", "vae_rope_enable", "vision_resize",
                        "vision_max_edge", "vision_guidance_seed", "vae_rope_boost",
                        "vision_guidance_seed_mode"].reduce((sum, name) => {
                            const index = names.indexOf(name);
                            return sum + (index < 0 ? 0 : valid(name, values[index]) ? 2 : -3);
                        }, 0);
                };
                const layout = score(old) > score(current) ? old : current;
                if (layout === current) return;
                const remapped = current.map((name, index) => {
                    const source = layout.indexOf(
                        name === "vae_rope_prompt_annotation" ? "vision_reference_annotation" : name
                    );
                    return source < 0 ? widgets[index].value : values[source];
                });
                return remapped;
            };
            const refresh = () => {
                const currentMode = find("vision_mode");
                if (currentMode) {
                    // Saved workflow combo options can outlive a backend node update.
                    // Keep the visible modes synchronized with the Python schema.
                    const modes = ["reference", "semantic", "visual", "style", "context"];
                    if (currentMode.options) currentMode.options.values = modes;
                    if (!modes.includes(currentMode.value)) {
                        currentMode.value = "reference";
                    }
                }
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
                if (visual && fit === "crop" && find("vae_rope_placement_grid")) {
                    find("vae_rope_placement_grid").value = "inside";
                }
                if (visual && fit === "native" && find("vae_rope_rope_geometry")) {
                    find("vae_rope_rope_geometry").value = "source";
                }
                setWidgetDisabled(node, "vision_max_edge", !vision || find("vision_resize")?.value !== true);
                const visionMode = find("vision_mode")?.value ?? "reference";
                setWidgetDisabled(node, "vision_style_processing", !vision || visionMode !== "style");
                setWidgetDisabled(node, "vision_instruction", !vision);
                setWidgetDisabled(node, "vae_rope_prompt_annotation", !visual);
                setWidgetDisabled(node, "vision_vision_fidelity", !vision || !["semantic", "style"].includes(visionMode));
                if (vision && visionMode !== "style" && find("vision_style_processing")) {
                    find("vision_style_processing").value = "full";
                }
                setWidgetDisabled(node, "vision_guidance_seed",
                    !vision || (find("vision_guidance_seed_mode")?.value ?? "fixed") !== "fixed");
                node.setDirtyCanvas?.(true, true);
            };
            for (const name of ["vae_rope_enable", "vae_rope_reference_fit",
                                "vision_enable", "vision_resize", "vision_mode", "vision_guidance_seed_mode"]) {
                const widget = find(name);
                if (!widget) continue;
                const original = widget.callback;
                widget.callback = function (...args) {
                    if (name === "vision_mode") {
                        const mode = find("vision_mode")?.value;
                        const fidelity = find("vision_vision_fidelity");
                        if (fidelity && mode === "style") fidelity.value = 0.0;
                        else if (fidelity && mode === "semantic") fidelity.value = 0.25;
                    }
                    const result = original?.apply(this, args);
                    refresh();
                    return result;
                };
            }
            const configure = node.onConfigure;
            node.onConfigure = function (...args) {
                const remapped = restoreReferenceWidgets(args[0]);
                const result = configure?.apply(this, args);
                // LiteGraph restores widget values BEFORE invoking onConfigure.
                // Replacing args[0] alone cannot repair the already-restored UI.
                if (remapped) {
                    const widgets = (node.widgets ?? []).filter(w => w.serialize !== false);
                    widgets.forEach((widget, i) => {
                        if (i < remapped.length) widget.value = remapped[i];
                    });
                }
                const mode = find("vision_mode");
                if (mode && !["reference", "semantic", "visual", "style", "context",
                    "native", "none", "guided"].includes(mode.value)) {
                    mode.value = "reference";
                }
                const edge = find("vision_max_edge");
                if (edge && (!Number.isInteger(edge.value) || edge.value < 384 || edge.value > 2048)) {
                    edge.value = 768;
                }
                const seed = find("vision_guidance_seed");
                if (seed && (!Number.isInteger(seed.value) || seed.value < 0)) seed.value = 0;
                refresh();
                return result;
            };
            setTimeout(refresh, 20);
        }

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

                if (processingWidget) {
                    processingWidget.disabled = !styleMode;
                    if (!styleMode) processingWidget.value = "full";
                }

                // All Vision modes use Qwen guidance generation in the current
                // T2I runtime, so these controls must stay available for
                // Semantic, Visual, Style, and Context.
                setWidgetDisabled(node, "guidance_seed_mode", false);
                setWidgetDisabled(node, "guidance_detail", false);
                setWidgetDisabled(node, "vision_guidance", false);

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
            const visionEnableWidget = node.widgets?.find((w) => w.name === "latent_vision_enable");

            const restoreLatentWidgets = (saved) => {
                const values = saved?.widgets_values;
                if (!Array.isArray(values)) return;
                const widgets = (node.widgets ?? []).filter(w => w.serialize !== false);
                const current = widgets.map(w => w.name);
                const withoutAnnotation = current.filter(n => n !== "latent_vision_annotation");
                // Earlier versions placed annotation at the end of the Vision
                // group, or included two removed guidance controls.
                const legacy = [...withoutAnnotation];
                const insertAt = legacy.indexOf("latent_vision_vision_fidelity");
                if (insertAt < 0) return;
                legacy.splice(insertAt, 0, "latent_vision_reference_annotation");
                const original = [...legacy];
                const beforeAnnotation = original.indexOf("latent_vision_reference_annotation");
                original.splice(beforeAnnotation, 0,
                    "latent_vision_guidance_detail", "latent_vision_vision_guidance");
                const contentAnnotation = [...withoutAnnotation];
                contentAnnotation.splice(contentAnnotation.indexOf("latent_vision_enable"), 0,
                    "content_prompt_annotation");

                const modes = ["reference", "semantic", "visual", "style", "context",
                    "native", "none", "guided"];
                const controls = ["latent_vision_mode", "latent_vision_enable",
                    "latent_vision_resize", "latent_vision_max_edge",
                    "latent_vision_guidance_seed", "latent_vision_guidance_seed_mode",
                    "content", "batch_size"];
                const valid = (name, value) => {
                    if (name === "latent_vision_mode") return modes.includes(value);
                    if (name === "latent_vision_enable" || name === "latent_vision_resize") {
                        return typeof value === "boolean";
                    }
                    if (name === "latent_vision_max_edge") return Number.isInteger(value) && value >= 384 && value <= 2048;
                    if (name === "latent_vision_guidance_seed") return Number.isInteger(value) && value >= 0;
                    if (name === "latent_vision_guidance_seed_mode") return ["fixed", "random", "increment", "decrement"].includes(value);
                    if (name === "content") return ["empty", "from_image"].includes(value);
                    if (name === "batch_size") return Number.isInteger(value) && value >= 1 && value <= 64;
                    return true;
                };
                const layouts = [current, withoutAnnotation, legacy, original, contentAnnotation];
                let selected = current;
                let best = -Infinity;
                for (const names of layouts) {
                    if (names.length !== values.length) continue;
                    const score = controls.reduce((sum, name) => {
                        const index = names.indexOf(name);
                        return sum + (index < 0 ? 0 : valid(name, values[index]) ? 2 : -3);
                    }, 0);
                    if (score > best) { best = score; selected = names; }
                }
                if (selected === current) return;
                return current.map((name, i) => {
                    const sourceName = name === "latent_vision_annotation"
                        ? selected.includes("latent_vision_reference_annotation")
                            ? "latent_vision_reference_annotation"
                            : "content_prompt_annotation"
                        : name;
                    const source = selected.indexOf(sourceName);
                    return source >= 0 ? values[source] : widgets[i].value;
                });
            };

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
            };

            const updateVisionState = () => {
                // Empty target latents have no source image for Vision analysis.
                // Clear the switch, rather than just greying it out while true.
                if (contentWidget?.value !== "from_image" && visionEnableWidget) {
                    visionEnableWidget.value = false;
                }
                const modeWidget = node.widgets?.find(w => w.name === "latent_vision_mode");
                if (modeWidget) {
                    const modes = ["reference", "semantic", "visual", "style", "context"];
                    if (modeWidget.options) modeWidget.options.values = modes;
                    if (["native", "none", "guided"].includes(modeWidget.value)) {
                        modeWidget.value = "reference";
                    } else if (!modes.includes(modeWidget.value)) {
                        // Saved workflows can restore an obsolete numeric combo index.
                        // Do not silently infer a semantic mode from that index.
                        modeWidget.value = "reference";
                    }
                }
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
                const mode = node.widgets?.find(w => w.name === "latent_vision_mode")?.value ?? "reference";
                setWidgetDisabled(node, "latent_vision_style_processing", !enabled || mode !== "style");
                const seedMode = node.widgets?.find(w => w.name === "latent_vision_guidance_seed_mode")?.value ?? "fixed";
                setWidgetDisabled(node, "latent_vision_guidance_seed", !enabled || seedMode !== "fixed");
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
                const remapped = restoreLatentWidgets(args[0]);
                const result = originalConfigure?.apply(this, args);
                // LiteGraph already assigned the positional values at this stage.
                if (remapped) {
                    const widgets = (node.widgets ?? []).filter(w => w.serialize !== false);
                    widgets.forEach((widget, i) => {
                        if (i < remapped.length) widget.value = remapped[i];
                    });
                }
                updateDimensionsState();
                updateContentState();
                updateVisionState();
                return result;
            };

            setTimeout(() => {
                updateDimensionsState();
                updateContentState();
                updateVisionState();
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


    }
});
