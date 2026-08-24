import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;

        // Dynamic widget management for CcCKrea2QwenVisionImagePrep
        if (node.comfyClass === "CcCKrea2QwenVisionImagePrep") {
            const modeWidget = node.widgets?.find(w => w.name === "mode");
            if (modeWidget) {
                const updatePrepState = () => {
                    const mode = modeWidget.value;
                    const minMpWidget = node.widgets?.find(w => w.name === "min_mp");
                    const maxMpWidget = node.widgets?.find(w => w.name === "max_mp");
                    const fixedMpWidget = node.widgets?.find(w => w.name === "fixed_mp");
                    const downscaleWidget = node.widgets?.find(w => w.name === "downscale_method");
                    const upscaleWidget = node.widgets?.find(w => w.name === "upscale_method");

                    if (mode === "native") {
                        if (minMpWidget) minMpWidget.disabled = true;
                        if (maxMpWidget) maxMpWidget.disabled = true;
                        if (fixedMpWidget) fixedMpWidget.disabled = true;
                        if (downscaleWidget) downscaleWidget.disabled = true;
                        if (upscaleWidget) upscaleWidget.disabled = true;
                    } else if (mode === "fixed") {
                        if (minMpWidget) minMpWidget.disabled = true;
                        if (maxMpWidget) maxMpWidget.disabled = true;
                        if (fixedMpWidget) fixedMpWidget.disabled = false;
                        if (downscaleWidget) downscaleWidget.disabled = false;
                        if (upscaleWidget) upscaleWidget.disabled = false;
                    } else {
                        // adaptive (default)
                        if (minMpWidget) minMpWidget.disabled = false;
                        if (maxMpWidget) maxMpWidget.disabled = false;
                        if (fixedMpWidget) fixedMpWidget.disabled = true;
                        if (downscaleWidget) downscaleWidget.disabled = false;
                        if (upscaleWidget) upscaleWidget.disabled = false;
                    }
                };

                const origCb = modeWidget.callback;
                modeWidget.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    updatePrepState();
                };
                setTimeout(updatePrepState, 20);
            }
        }

        // Dynamic widget management for CcCKrea2ReferenceImage
        if (node.comfyClass === "CcCKrea2ReferenceImage") {
            const refPathWidget = node.widgets?.find(w => w.name === "reference_path");
            if (refPathWidget) {
                const updateRefPathState = () => {
                    const mode = refPathWidget.value;
                    const isEdit = (mode === "edit");

                    const editWidgets = ["attention_boost", "masked_attention_boost", "visual_reference_fit"];
                    const styleWidgets = ["style_fidelity", "style_processing", "indirect_style_transfer"];

                    node.widgets?.forEach(w => {
                        if (editWidgets.includes(w.name)) {
                            w.disabled = !isEdit;
                        }
                        if (styleWidgets.includes(w.name)) {
                            w.disabled = isEdit;
                        }
                    });
                };

                const origCallback = refPathWidget.callback;
                refPathWidget.callback = function (val) {
                    if (origCallback) origCallback.apply(this, arguments);
                    updateRefPathState();
                };
                setTimeout(updateRefPathState, 20);
            }
        }

        // Dynamic widget management for CcCKrea2TargetLatent
        if (node.comfyClass === "CcCKrea2TargetLatent") {
            const contentWidget = node.widgets?.find(w => w.name === "target_content");
            const geomWidget = node.widgets?.find(w => w.name === "geometry_mode");
            const visionWidget = node.widgets?.find(w => w.name === "include_in_vision");

            const updateTargetState = () => {
                const isGeomFixed = (geomWidget?.value === "fixed");
                const isVisionIncluded = (visionWidget?.value !== "no");

                const targetMpWidget = node.widgets?.find(w => w.name === "target_megapixels");
                const fixedMpWidget = node.widgets?.find(w => w.name === "fixed_megapixels");
                const aspectWidget = node.widgets?.find(w => w.name === "aspect_ratio");

                const slotWidget = node.widgets?.find(w => w.name === "target_vision_slot");
                const aliasWidget = node.widgets?.find(w => w.name === "target_alias");
                const instructionWidget = node.widgets?.find(w => w.name === "target_vision_instruction");

                if (fixedMpWidget) fixedMpWidget.disabled = !isGeomFixed;
                if (targetMpWidget) targetMpWidget.disabled = isGeomFixed;
                if (aspectWidget) aspectWidget.disabled = !isGeomFixed;

                if (slotWidget) slotWidget.disabled = !isVisionIncluded;
                if (aliasWidget) aliasWidget.disabled = !isVisionIncluded;
                if (instructionWidget) instructionWidget.disabled = !isVisionIncluded;
            };

            [contentWidget, geomWidget, visionWidget].forEach(w => {
                if (w) {
                    const origCb = w.callback;
                    w.callback = function () {
                        if (origCb) origCb.apply(this, arguments);
                        updateTargetState();
                    };
                }
            });
            setTimeout(updateTargetState, 20);
        }

        // Dynamic widget management for Edit and Easy Edit nodes
        if (["CcCKrea2Edit", "CcCKrea2EasyEdit", "CcCKrea2EasyEditOstris"].includes(node.comfyClass)) {
            const methodWidget = node.widgets?.find(w => w.name === "reference_method");
            const patchWidget = node.widgets?.find(w => w.name === "apply_model_patch" || w.name === "apply_krea2_edit_patch" || w.name === "apply_ostris_edit_patch");
            const kvCacheWidget = node.widgets?.find(w => w.name === "ostris_kv_cache");

            const updateEditState = () => {
                const method = methodWidget?.value || (node.comfyClass === "CcCKrea2EasyEditOstris" ? "ostris_edit" : "krea2_edit");

                // Native does not support model patching
                if (method === "native") {
                    if (patchWidget) patchWidget.disabled = true;
                } else {
                    if (patchWidget) patchWidget.disabled = false;
                }

                // ostris_kv_cache is currently unsupported in runtime environment
                if (kvCacheWidget) {
                    kvCacheWidget.disabled = true;
                    kvCacheWidget.tooltip = "Currently unavailable in the CcC Ostris backend. Intended only for LoRAs trained with ai-toolkit kv_cache.";
                }
            };

            if (methodWidget) {
                const origCb = methodWidget.callback;
                methodWidget.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    updateEditState();
                };
            }
            setTimeout(updateEditState, 20);

            // Dynamic default prompt management for Easy Edit nodes
            if (["CcCKrea2EasyEdit", "CcCKrea2EasyEditOstris"].includes(node.comfyClass)) {
                const useDefaultWidget = node.widgets?.find(w => w.name === "use_default_prompt");
                const posPromptWidget = node.widgets?.find(w => w.name === "positive_prompt");
                const presetWidget = node.widgets?.find(w => w.name === "preset");
                const refSubjWidget = node.widgets?.find(w => w.name === "reference_subject");
                const subjDescWidget = node.widgets?.find(w => w.name === "subject_description");
                const outfitSourceWidget = node.widgets?.find(w => w.name === "outfit_source");
                const styleSourceWidget = node.widgets?.find(w => w.name === "style_source");

                const EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER =
                    "Replace only the identity of the {reference_subject} of the scene image with the identity of the {subject_description} from the subject image.\n\n" +
                    "Transfer the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject_description} from the subject image to the {reference_subject} of the scene image.\n\n" +
                    "Preserve the position, action, pose, role, interaction, clothing, and accessories of the {reference_subject} from the scene image.";

                const EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT =
                    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n" +
                    "Transfer the complete {subject} from the {subject_source}, including the exact facial identity, facial features, hair, anatomy, body shape, body proportions, clothing, and accessories.\n\n" +
                    "Preserve the face, body shape, body proportions, clothing, and accessories of the {subject} from the {subject_source}.\n\n" +
                    "Adapt the transferred {subject} naturally to the target scene while preserving the rest of the scene.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER =
                    "Replace only the {reference_subject} of the {scene_source} with the {subject} from the {subject_source}.\n\n" +
                    "Transfer the exact facial identity, facial features, hair, anatomy, body shape, body proportions, clothing, and accessories of the {subject} from the {subject_source}.\n\n" +
                    "Place the transferred {subject} in the same position and pose as the {reference_subject}. Make the transferred {subject} perform the same action, fulfill the same role, and interact with every person and object in the same way as the {reference_subject}.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT =
                    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n" +
                    "Preserve the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n" +
                    "Dress the transferred {subject} using {outfit_reference}.\n\n" +
                    "Do not preserve the clothing or accessories of the {subject} from the {subject_source} when an explicit outfit source is selected. Use {outfit_reference} instead.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER =
                    "Transfer only {outfit_reference} to the {subject}.\n\n" +
                    "Preserve the {subject} identity, body, pose, framing, and composition.\n\n" +
                    "Do not preserve the {subject} clothing.\n\n" +
                    "Fit the transferred outfit and accessories naturally to the {subject}.\n\n" +
                    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n" +
                    "Do not duplicate accessories.";

                const EASY_DEFAULT_PROMPT_SUBJECT_SCENE =
                    "Place the {subject} from the {subject_source} naturally into the {scene_source}.\n\n" +
                    "Preserve the {subject} identity, body shape, and body proportions.\n\n" +
                    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n" +
                    "Adapt the {subject} naturally to the {scene_source} lighting and environment.";

                const EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT =
                    "Place the {subject} from the {subject_source} naturally into the {scene_source} wearing {outfit_reference}.\n\n" +
                    "Preserve the {subject} identity, body shape, and body proportions.\n\n" +
                    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n" +
                    "Do not preserve the {subject} clothing.\n\n" +
                    "Fit the transferred outfit and accessories naturally to the {subject} and the scene.\n\n" +
                    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n" +
                    "Do not duplicate accessories.";

                const EASY_DEFAULT_PROMPT_STYLE =
                    "Use the {style_source} only as a visual style reference.\n\n" +
                    "Apply its color palette, lighting character, contrast, texture, rendering treatment, photographic treatment, and overall visual mood.\n\n" +
                    "Do not transfer subjects, identities, facial features, hair, anatomy, body shapes, clothing, accessories, poses, objects, environment, layout, framing, or scene composition from the {style_source}.";

                const EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION =
                    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n" +
                    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n" +
                    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n" +
                    "Adapt the {subject} naturally to the referenced action and environment.\n\n" +
                    "Creatively reinterpret the clothing and accessories worn by the {reference_subject} in the {scene_source} so they are appropriate for the {subject} and the newly generated image. Do not copy the original scene outfit literally.\n\n" +
                    "Generate a coherent new image rather than recreating the source scene exactly.";

                const EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION_WITH_OUTFIT =
                    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n" +
                    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n" +
                    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n" +
                    "Dress the {subject} using {outfit_reference}.\n\n" +
                    "Adapt the {subject} and the selected outfit naturally to the referenced action and environment.\n\n" +
                    "Generate a coherent new image rather than recreating the source scene exactly.";

                const IDENTITY_TEST_PRESETS = [
                    "identity_transfer",
                ];

                const GROUP_A_PRESETS = [];
                const GROUP_B_PRESETS = [];
                const GROUP_C_PRESETS = [];
                const GROUP_D_PRESETS = [];

                const PRESET_DISPLAY_LABELS = {
                    "flexible": "Flexible",
                    "balanced": "Balanced",
                    "consistent": "Consistent",
                    "preserve_identity": "Preserve Identity",
                    "max_identity": "Max Identity",
                    "identity_transfer": "Identity Transfer",
                    "subject_transfer_1": "Subject Transfer 1",
                    "subject_transfer_2": "Subject Transfer 2",
                    "flexible_subject_transfer_1": "Flexible Subject Transfer 1",
                    "flexible_subject_transfer_2": "Flexible Subject Transfer 2",
                    "preserve_scene": "Preserve Scene",
                    "outfit_transfer": "Outfit Transfer",
                    "style_transfer": "Style Transfer",
                    "scene_reinterpretation": "Scene Reinterpretation",
                };

                const DISPLAY_TO_PRESET_ID = Object.fromEntries(
                    Object.entries(PRESET_DISPLAY_LABELS).map(([k, v]) => [v, k])
                );

                const STABLE_PRESET_IDS = [
                    "flexible",
                    "balanced",
                    "consistent",
                    "preserve_identity",
                    "max_identity",
                    "identity_transfer",
                    "subject_transfer_1",
                    "subject_transfer_2",
                    "flexible_subject_transfer_1",
                    "flexible_subject_transfer_2",
                    "preserve_scene",
                    "outfit_transfer",
                    "style_transfer",
                    "scene_reinterpretation",
                ];

                const LEGACY_PRESETS = [
                    ...STABLE_PRESET_IDS,
                ];

                const migrateLegacyEasyEditWidgets = (info) => {
                    if (!info || !Array.isArray(info.widgets_values)) return;
                    const vals = info.widgets_values;
                    if (vals.length < 2) return;

                    // 1. If index 1 is a preset string (older schema 1 without use_default_prompt at index 1), splice use_default_prompt boolean
                    const valAtIndex1 = vals[1];
                    if (typeof valAtIndex1 === "string" && (
                        LEGACY_PRESETS.includes(valAtIndex1) ||
                        valAtIndex1.startsWith("transfer_identity_test_") ||
                        DISPLAY_TO_PRESET_ID[valAtIndex1] ||
                        Object.keys(PRESET_DISPLAY_LABELS).includes(valAtIndex1)
                    )) {
                        vals.splice(1, 0, true);
                        node._isPromptSystemManaged = true;
                    }

                    // 2. Schema check for reference_subject and subject_description fields (index 3 and 4)
                    // If index 3 is an outfit_source value, splice subject defaults.
                    if (typeof vals[3] === "string" && ["none", "subject image", "outfit image", "scene image", "style image"].includes(vals[3])) {
                        vals.splice(3, 0, "main subject", "main subject");
                    }

                    if (node.widgets && Array.isArray(node.widgets)) {
                        for (let i = 0; i < vals.length && i < node.widgets.length; i++) {
                            if (node.widgets[i]) {
                                node.widgets[i].value = vals[i];
                            }
                        }
                    }
                };

                let updateTimer = null;
                let updateRaf = null;

                const schedulePromptStateUpdate = () => {
                    updatePromptState();

                    if (updateTimer) clearTimeout(updateTimer);
                    if (updateRaf && typeof cancelAnimationFrame === "function") {
                        cancelAnimationFrame(updateRaf);
                        updateRaf = null;
                    }

                    const runDeferredPasses = () => {
                        updatePromptState();
                        updateTimer = setTimeout(() => {
                            updatePromptState();
                            node._isRestoring = false;
                            updateTimer = null;
                        }, 50);
                    };

                    if (typeof requestAnimationFrame === "function") {
                        updateRaf = requestAnimationFrame(() => {
                            updateRaf = null;
                            runDeferredPasses();
                        });
                    } else {
                        updateTimer = setTimeout(runDeferredPasses, 20);
                    }
                };

                const origOnConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    node._isRestoring = true;
                    migrateLegacyEasyEditWidgets(info);
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    schedulePromptStateUpdate();
                };

                const renderJsEasyPrompt = (template, context) => {
                    if (!template) return "";
                    const refSubj = (context.reference_subject && context.reference_subject.trim()) ? context.reference_subject.trim() : "main subject";
                    const subj = (context.subject && context.subject.trim()) ? context.subject.trim() : "main subject";
                    const subjDesc = (context.subject_description && context.subject_description.trim()) ? context.subject_description.trim() : (
                        (context.subject && context.subject.trim()) ? context.subject.trim() : "main subject"
                    );
                    const sceneSrc = context.scene_source || "scene image";
                    const subjSrc = context.subject_source || "subject image";
                    const outfitSrc = context.outfit_source || "outfit image";
                    const styleSrc = context.style_source || "style image";
                    let outfitReference = "the principal outfit identified in the outfit image";
                    if (outfitSrc === "scene image") {
                        outfitReference = `the clothing, footwear, and accessories worn by the ${refSubj} in the scene image`;
                    } else if (outfitSrc === "subject image") {
                        outfitReference = `the clothing, footwear, and accessories worn by the ${subjDesc} in the subject image`;
                    } else if (outfitSrc === "style image") {
                        outfitReference = "the relevant clothing, footwear, and accessories interpreted from the style image";
                    }

                    return template
                        .replace(/\{reference_subject\}/g, refSubj)
                        .replace(/\{subject_description\}/g, subjDesc)
                        .replace(/\{subject\}/g, subj)
                        .replace(/\{scene_source\}/g, sceneSrc)
                        .replace(/\{subject_source\}/g, subjSrc)
                        .replace(/\{outfit_source\}/g, outfitSrc)
                        .replace(/\{outfit_reference\}/g, outfitReference)
                        .replace(/\{style_source\}/g, styleSrc);
                };

                const resolveJsDefaultPrompt = (preset, hasS, hasSc, hasO, hasSt, outfitSource, styleSource, refSubjVal, subjDescVal) => {
                    const context = {
                        reference_subject: refSubjVal,
                        subject: subjDescVal,
                        subject_description: subjDescVal,
                        scene_source: "scene image",
                        subject_source: "subject image",
                        outfit_source: outfitSource,
                        style_source: styleSource,
                    };

                    if (IDENTITY_TEST_PRESETS.includes(preset)) {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        return { hasDefault: true, text: renderJsEasyPrompt(EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER, context) };
                    }

                    if (["subject_transfer_1", "subject_transfer_2"].includes(preset)) {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        return { hasDefault: true, text: renderJsEasyPrompt(EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER, context) };
                    }

                    if (["flexible_subject_transfer_1", "flexible_subject_transfer_2"].includes(preset)) {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        const usesSubjectOutfit = hasO && outfitSource === "subject image";
                        const template = (hasO && !usesSubjectOutfit)
                            ? EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT
                            : EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT;
                        return { hasDefault: true, text: renderJsEasyPrompt(template, context) };
                    }

                    if (preset === "scene_reinterpretation") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        const template = hasO
                            ? EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION_WITH_OUTFIT
                            : EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION;
                        return { hasDefault: true, text: renderJsEasyPrompt(template, context) };
                    }

                    let baseTemplate = "";
                    if (preset === "outfit_transfer") {
                        if (hasO) baseTemplate = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        else if (hasSc) baseTemplate = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                    } else if (preset === "preserve_scene") {
                        if (hasSc) {
                            baseTemplate = hasO ? EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT : EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                        } else if (hasO) {
                            baseTemplate = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        }
                    } else if (preset === "style_transfer") {
                        if (hasSt) baseTemplate = EASY_DEFAULT_PROMPT_STYLE;
                        else {
                            if (hasSc && hasO) baseTemplate = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT;
                            else if (hasSc) baseTemplate = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                            else if (hasO) baseTemplate = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        }
                    } else {
                        if (hasSc && hasO) baseTemplate = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT;
                        else if (hasSc) baseTemplate = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                        else if (hasO) baseTemplate = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                    }

                    if (hasSt && preset !== "style_transfer") {
                        const renderedStyle = renderJsEasyPrompt(EASY_DEFAULT_PROMPT_STYLE, context);
                        if (baseTemplate) {
                            const renderedBase = renderJsEasyPrompt(baseTemplate, context);
                            return { hasDefault: true, text: `${renderedBase}\n\n${renderedStyle}` };
                        }
                        return { hasDefault: true, text: renderedStyle };
                    }

                    if (baseTemplate) {
                        return { hasDefault: true, text: renderJsEasyPrompt(baseTemplate, context) };
                    }
                    return { hasDefault: false, text: "" };
                };

                const findPromptWidgetElement = (widget, targetNode) => {
                    if (!widget) return null;
                    if (widget.inputEl) {
                        if (widget.inputEl.tagName === "TEXTAREA" || widget.inputEl.tagName === "INPUT") {
                            return widget.inputEl;
                        }
                        const inner = widget.inputEl.querySelector?.("textarea, input");
                        if (inner) return inner;
                        return widget.inputEl;
                    }
                    if (widget.element) {
                        if (widget.element.tagName === "TEXTAREA" || widget.element.tagName === "INPUT") {
                            return widget.element;
                        }
                        const inner = widget.element.querySelector?.("textarea, input");
                        if (inner) return inner;
                        return widget.element;
                    }
                    if (targetNode && targetNode.element) {
                        const nodeInner = targetNode.element.querySelector?.("textarea, input");
                        if (nodeInner) return nodeInner;
                    }
                    return null;
                };

                const setPromptWidgetValue = (widget, value, targetNode) => {
                    if (!widget) return;
                    targetNode._updatingManagedPrompt = true;
                    try {
                        const prevValue = widget.value;
                        widget.value = value;

                        if (widget.valueStore && typeof widget.valueStore.set === "function") {
                            widget.valueStore.set(value);
                        } else if (widget.component && typeof widget.component.setValue === "function") {
                            widget.component.setValue(value);
                        }

                        if (typeof targetNode.onWidgetChanged === "function" && prevValue !== value) {
                            targetNode.onWidgetChanged(widget.name, value, prevValue, widget);
                        }

                        const el = findPromptWidgetElement(widget, targetNode);
                        if (el && el.value !== value) {
                            el.value = value;
                        }
                    } finally {
                        targetNode._updatingManagedPrompt = false;
                    }
                    if (typeof targetNode.setDirtyCanvas === "function") {
                        targetNode.setDirtyCanvas(true, true);
                    }
                };

                const setPromptWidgetManaged = (widget, managed, text, targetNode) => {
                    if (!widget) return;
                    if (managed) {
                        targetNode._isPromptSystemManaged = true;
                        targetNode._resolvedManagedPrompt = text;
                        widget.disabled = false;
                        widget.readOnly = true;
                        if (widget.options) {
                            widget.options.readOnly = true;
                        }
                        setPromptWidgetValue(widget, text, targetNode);

                        const applyElementProps = () => {
                            const el = findPromptWidgetElement(widget, targetNode);
                            if (el) {
                                el.readOnly = true;
                                el.disabled = false;
                                el.title = "Resolved preset prompt. Disable Use Default Prompt to edit manually.";
                                if (el.style) el.style.opacity = "0.85";
                            }
                        };
                        applyElementProps();

                        const el = findPromptWidgetElement(widget, targetNode);
                        if (!el && !targetNode._renderRetryScheduled) {
                            targetNode._renderRetryScheduled = true;
                            if (typeof globalThis.requestAnimationFrame === "function") {
                                globalThis.requestAnimationFrame(() => {
                                    targetNode._renderRetryScheduled = false;
                                    applyElementProps();
                                });
                            }
                        }
                    } else {
                        targetNode._isPromptSystemManaged = false;
                        targetNode._resolvedManagedPrompt = null;
                        widget.disabled = false;
                        widget.readOnly = false;
                        if (widget.options) {
                            widget.options.readOnly = false;
                        }
                        const el = findPromptWidgetElement(widget, targetNode);
                        if (el) {
                            el.readOnly = false;
                            el.disabled = false;
                            el.title = "";
                            if (el.style) el.style.opacity = "";
                        }
                    }
                };

                const updatePromptState = () => {
                    if (!useDefaultWidget || !posPromptWidget) return;

                    if (node._isPromptSystemManaged === undefined) {
                        node._isPromptSystemManaged = !!useDefaultWidget.value;
                    }

                    const preset = DISPLAY_TO_PRESET_ID[presetWidget?.value] || presetWidget?.value || "balanced";
                    const isGroupD = GROUP_D_PRESETS.includes(preset);
                    const isGroupC = GROUP_C_PRESETS.includes(preset);
                    const isFlexibleSubjectTransfer = ["flexible_subject_transfer_1", "flexible_subject_transfer_2"].includes(preset);
                    const isSceneReinterpretation = preset === "scene_reinterpretation";
                    const usesOutfit = !["preserve_scene", "style_transfer", "subject_transfer_1", "subject_transfer_2", ...IDENTITY_TEST_PRESETS].includes(preset);
                    const isStyleDisabled = isGroupD;
                    const isSceneOutfitStyle = GROUP_C_PRESETS.includes(preset);
                    const isSceneAutoStyle = ["subject_transfer_1", "subject_transfer_2", "identity_transfer", "scene_reinterpretation", ...GROUP_B_PRESETS].includes(preset);
                    const isSceneAutoOutfit = isGroupD;

                    const subjectInput = node.inputs?.find(i => i.name === "subject");
                    const sceneInput = node.inputs?.find(i => i.name === "scene");
                    const outfitInput = node.inputs?.find(i => i.name === "outfit");
                    const styleInput = node.inputs?.find(i => i.name === "style");

                    const enteringFlexibleSubjectTransfer = isFlexibleSubjectTransfer
                        && node._lastEasyPreset !== undefined
                        && node._lastEasyPreset !== preset
                        && !node._isRestoring;
                    const enteringSceneReinterpretation = isSceneReinterpretation
                        && node._lastEasyPreset !== undefined
                        && node._lastEasyPreset !== preset
                        && !node._isRestoring;

                    if (outfitSourceWidget?.options) {
                        outfitSourceWidget.options.values = isFlexibleSubjectTransfer
                            ? ["none", "subject image", "scene image"]
                            : ["none", "subject image", "scene image", "outfit image", "style image"];
                    }
                    if (isFlexibleSubjectTransfer && !["none", "subject image", "scene image"].includes(outfitSourceWidget?.value)) {
                        outfitSourceWidget.value = "subject image";
                    }
                    if (enteringFlexibleSubjectTransfer) {
                        if (outfitSourceWidget) outfitSourceWidget.value = "subject image";
                        if (styleSourceWidget) styleSourceWidget.value = "none";
                    }
                    if (enteringSceneReinterpretation && outfitSourceWidget) {
                        outfitSourceWidget.value = "scene image";
                    }
                    node._lastEasyPreset = preset;

                    const outfitSource = outfitSourceWidget?.value || (isFlexibleSubjectTransfer ? "subject image" : "outfit image");
                    const styleSource = styleSourceWidget?.value || "none";
                    const refSubjVal = refSubjWidget?.value || "main subject";
                    const subjDescVal = subjDescWidget?.value || "main subject";

                    const styleSocketIsUsedAsOutfit = usesOutfit && !isFlexibleSubjectTransfer && outfitSource === "style image";

                    if (outfitSourceWidget) {
                        if (isSceneAutoOutfit) {
                            outfitSourceWidget.disabled = true;
                            outfitSourceWidget.label = "Outfit Source [Auto: Scene]";
                            outfitSourceWidget.tooltip = "Automatically uses the Scene image as the outfit reference.";
                        } else if (!usesOutfit) {
                            outfitSourceWidget.disabled = true;
                            delete outfitSourceWidget.label;
                            outfitSourceWidget.tooltip = "Outfit source selector is disabled for this preset.";
                        } else {
                            outfitSourceWidget.disabled = false;
                            delete outfitSourceWidget.label;
                            outfitSourceWidget.tooltip = isFlexibleSubjectTransfer
                                ? "Semantic Outfit source. Defaults to Subject and uses direct semantic conditioning."
                                : "Source image socket to use for outfit conditioning.";
                        }
                    }
                    if (outfitInput) {
                        outfitInput.disabled = isSceneAutoOutfit || !usesOutfit || isFlexibleSubjectTransfer;
                    }

                    if (styleSourceWidget) {
                        if (isStyleDisabled) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Disabled]";
                            styleSourceWidget.tooltip = "This preset disables style conditioning.";
                        } else if (isSceneOutfitStyle) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Auto: Scene Outfit]";
                            styleSourceWidget.tooltip = "Uses the Scene image as outfit-focused style conditioning to reinforce the target subject's clothing and accessories.";
                        } else if (isSceneAutoStyle) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Auto: Scene]";
                            let tooltipText = "Subject Transfer automatically uses its custom indirect Scene guidance. The stored Style Source value is preserved for other presets.";
                            if (preset === "scene_reinterpretation") {
                                tooltipText = "Scene Reinterpretation automatically uses Scene as a direct custom Style reference to recreate the scene while excluding the replaced subject and outfit.";
                            } else if (IDENTITY_TEST_PRESETS.includes(preset)) {
                                tooltipText = "Identity Transfer automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            }
                            styleSourceWidget.tooltip = tooltipText;
                        } else {
                            styleSourceWidget.disabled = false;
                            delete styleSourceWidget.label;
                            styleSourceWidget.tooltip = "Source image socket to use for style conditioning.";
                        }
                    }

                    if (styleInput) {
                        if (isStyleDisabled || isSceneOutfitStyle) {
                            styleInput.disabled = true;
                        } else if (isSceneAutoStyle) {
                            styleInput.disabled = !styleSocketIsUsedAsOutfit;
                        } else {
                            styleInput.disabled = false;
                        }
                    }

                    const hasS = !!(subjectInput && subjectInput.link != null);
                    const hasSc = !!(sceneInput && sceneInput.link != null);
                    const hasRawO = !!(outfitInput && outfitInput.link != null);
                    const hasRawSt = !!(styleInput && styleInput.link != null);

                    const effectiveOutfitSource = isSceneAutoOutfit ? "scene image" : (usesOutfit ? outfitSource : "none");
                    const effectiveStyleSource = isStyleDisabled ? "none" : ((isSceneAutoStyle || isSceneOutfitStyle) ? "scene image" : styleSource);

                    const hasO = isSceneAutoOutfit ? hasSc : (usesOutfit && (
                        (effectiveOutfitSource === "outfit image" && hasRawO) ||
                        (effectiveOutfitSource === "subject image" && hasS) ||
                        (effectiveOutfitSource === "scene image" && hasSc) ||
                        (effectiveOutfitSource === "style image" && hasRawSt)
                    ));

                    const hasSt = isStyleDisabled ? false : ((isSceneAutoStyle || isSceneOutfitStyle) ? hasSc : (
                        (effectiveStyleSource === "style image" && hasRawSt) ||
                        (effectiveStyleSource === "scene image" && hasSc) ||
                        (effectiveStyleSource === "subject image" && hasS)
                    ));

                    const isSubjectOnly = hasS && !hasSc && !hasO && !hasSt;
                    const { hasDefault, text } = resolveJsDefaultPrompt(
                        preset, hasS, hasSc, hasO, hasSt, effectiveOutfitSource, effectiveStyleSource, refSubjVal, subjDescVal
                    );

                    // Subject description widgets are enabled ONLY when use_default_prompt is ON and preset supports subject fields
                    const supportsSubjectFields = [...IDENTITY_TEST_PRESETS, "subject_transfer_1", "subject_transfer_2", "flexible_subject_transfer_1", "flexible_subject_transfer_2", "outfit_transfer", "scene_reinterpretation"].includes(preset);
                    const enableSubjectFields = useDefaultWidget.value && supportsSubjectFields;
                    if (refSubjWidget) refSubjWidget.disabled = !enableSubjectFields;
                    if (subjDescWidget) subjDescWidget.disabled = !enableSubjectFields;

                    if (isSubjectOnly || !hasDefault) {
                        useDefaultWidget.disabled = true;
                        setPromptWidgetManaged(posPromptWidget, false, "", node);

                        if (!node._isRestoring) {
                            if (node._isPromptSystemManaged && isSubjectOnly) {
                                setPromptWidgetValue(posPromptWidget, "", node);
                                node._isPromptSystemManaged = false;
                                node._resolvedManagedPrompt = null;
                            }
                        }
                    } else {
                        useDefaultWidget.disabled = false;

                        if (useDefaultWidget.value && hasDefault) {
                            setPromptWidgetManaged(posPromptWidget, true, text, node);
                            node._isRestoring = false;
                        } else {
                            setPromptWidgetManaged(posPromptWidget, false, "", node);
                            node._isRestoring = false;
                        }
                    }

                    if (typeof node.setDirtyCanvas === "function") {
                        node.setDirtyCanvas(true, true);
                    }
                };

                if (useDefaultWidget) {
                    const origCb = useDefaultWidget.callback;
                    useDefaultWidget.callback = function () {
                        if (origCb) origCb.apply(this, arguments);
                        if (!useDefaultWidget.value) {
                            node._isPromptSystemManaged = false;
                            node._resolvedManagedPrompt = null;
                        } else {
                            node._isPromptSystemManaged = true;
                        }
                        schedulePromptStateUpdate();
                    };
                }

                if (presetWidget) {
                    presetWidget.formatValue = (val) => PRESET_DISPLAY_LABELS[val] || val;
                    const origValues = presetWidget.options?.values;
                    if (presetWidget.options) {
                        presetWidget.options.values = () => {
                            const raw = typeof origValues === "function" ? origValues() : origValues;
                            const rawList = (Array.isArray(raw) && raw.length > 0) ? raw : Object.keys(PRESET_DISPLAY_LABELS);
                            const stableInRaw = STABLE_PRESET_IDS.filter(id => rawList.includes(id));
                            const remainingInRaw = rawList.filter(id => !STABLE_PRESET_IDS.includes(id));
                            const ordered = [...stableInRaw, ...remainingInRaw];
                            return ordered.map(v => PRESET_DISPLAY_LABELS[v] || v);
                        };
                    }
                }

                [presetWidget, refSubjWidget, subjDescWidget, outfitSourceWidget, styleSourceWidget].forEach(w => {
                    if (w) {
                        const orig = w.callback;
                        w.callback = function (val) {
                            if (w === presetWidget && DISPLAY_TO_PRESET_ID[val]) {
                                this.value = DISPLAY_TO_PRESET_ID[val];
                                val = this.value;
                            }
                            if (orig) orig.apply(this, arguments);
                            schedulePromptStateUpdate();
                        };
                    }
                });

                if (posPromptWidget) {
                    const origPosCb = posPromptWidget.callback;
                    posPromptWidget.callback = function (val) {
                        if (node._updatingManagedPrompt) return;

                        if (useDefaultWidget?.value && node._isPromptSystemManaged && node._resolvedManagedPrompt != null) {
                            if (val !== node._resolvedManagedPrompt) {
                                setPromptWidgetValue(posPromptWidget, node._resolvedManagedPrompt, node);
                                return;
                            }
                        }

                        if (origPosCb) origPosCb.apply(this, arguments);
                        if (!useDefaultWidget?.value) {
                            node._isPromptSystemManaged = false;
                            node._resolvedManagedPrompt = null;
                        }
                    };
                }

                const origConnChange = node.onConnectionsChange;
                node.onConnectionsChange = function () {
                    if (origConnChange) origConnChange.apply(this, arguments);
                    schedulePromptStateUpdate();
                };

                schedulePromptStateUpdate();
            }
        }
    }
});
