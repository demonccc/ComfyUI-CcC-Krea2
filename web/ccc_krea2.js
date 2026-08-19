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
                const outfitSourceWidget = node.widgets?.find(w => w.name === "outfit_source");
                const styleSourceWidget = node.widgets?.find(w => w.name === "style_source");

                const EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER =
                    "Transfer only the outfit and accessories from the outfit reference to the subject. " +
                    "Preserve the subject identity, body, pose, framing, and composition. " +
                    "Do not preserve the subject clothing. " +
                    "Fit the transferred outfit and accessories naturally to the subject. " +
                    "Keep accessories physically attached to the subject in a natural way and never floating. " +
                    "Do not duplicate accessories.";

                const EASY_DEFAULT_PROMPT_SUBJECT_SCENE =
                    "Place the subject from the subject reference naturally into the scene reference. " +
                    "Preserve the subject identity, body shape, and body proportions. " +
                    "Preserve the scene composition, environment, framing, perspective, and spatial layout. " +
                    "Adapt the subject naturally to the scene lighting and environment.";

                const EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT =
                    "Place the subject from the subject reference naturally into the scene reference wearing the outfit and accessories from the outfit reference. " +
                    "Preserve the subject identity, body shape, and body proportions. " +
                    "Preserve the scene composition, environment, framing, perspective, and spatial layout. " +
                    "Do not preserve the subject clothing. " +
                    "Fit the transferred outfit and accessories naturally to the subject and the scene. " +
                    "Keep accessories physically attached to the subject in a natural way and never floating. " +
                    "Do not duplicate accessories.";

                const EASY_DEFAULT_PROMPT_STYLE =
                    "Apply the visual style from the style reference while preserving the subject identity, content, geometry, framing, and composition. " +
                    "Transfer only the visual style, including its color palette, texture, lighting character, and overall visual mood. " +
                    "Do not copy subjects, objects, or scene content from the style reference.";

                const EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_BASE =
                    "Replace only the target subject in the scene reference with the subject from the subject reference.\n\n" +
                    "Preserve the identity of the subject from the subject reference.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER =
                    "Replace only the identity of the target subject in the scene reference with the identity of the subject from the subject reference.\n\n" +
                    "Preserve the facial identity, facial features, hair, body identity, anatomy, body shape, and body proportions of the subject reference.\n\n" +
                    "Preserve the target subject's scene role, position, action, pose, clothing, interaction, and surrounding scene.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION =
                    "Create a new image of the subject from the subject reference performing the main action or activity shown by the main subject in the scene reference.\n\n" +
                    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the subject reference.\n\n" +
                    "Use the scene reference as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n" +
                    "Adapt the subject naturally to the referenced action and environment.\n\n" +
                    "Creatively reinterpret the clothing and accessories worn by the main subject in the scene so they are appropriate for the subject and the newly generated image. Do not copy the original scene outfit literally.\n\n" +
                    "Generate a coherent new image rather than recreating the source scene exactly.";

                const LEGACY_PRESETS = [
                    "flexible",
                    "balanced",
                    "consistent",
                    "preserve_identity",
                    "max_identity",
                    "preserve_scene",
                    "outfit_transfer",
                    "style_transfer",
                ];

                const migrateLegacyEasyEditWidgets = (info) => {
                    if (!info || !Array.isArray(info.widgets_values)) return;
                    const vals = info.widgets_values;
                    if (vals.length < 2) return;

                    const valAtIndex1 = vals[1];

                    // If index 1 is already a boolean, the workflow uses the new schema
                    if (typeof valAtIndex1 === "boolean") return;

                    // If index 1 is a valid preset string, this is a legacy workflow layout
                    if (typeof valAtIndex1 === "string" && LEGACY_PRESETS.includes(valAtIndex1)) {
                        vals.splice(1, 0, false);
                        node._isPromptSystemManaged = false;

                        if (node.widgets && Array.isArray(node.widgets)) {
                            for (let i = 0; i < vals.length && i < node.widgets.length; i++) {
                                if (node.widgets[i]) {
                                    node.widgets[i].value = vals[i];
                                }
                            }
                        }
                    } else {
                        console.warn(`[CcC.Krea2] Unknown value at use_default_prompt position for node ${node.comfyClass}:`, valAtIndex1);
                    }
                };

                const origOnConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    migrateLegacyEasyEditWidgets(info);
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    updatePromptState();
                };

                const resolveJsDefaultPrompt = (preset, hasS, hasSc, hasO, hasSt, outfitSource, styleSource) => {
                    if (preset === "identity_transfer") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        return { hasDefault: true, text: EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER };
                    }

                    if (preset === "subject_transfer") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        let outfitClause = "Keep the clothing and accessories of the subject reference.";
                        if (hasO) {
                            if (outfitSource === "outfit image") {
                                outfitClause = "Use the clothing and accessories from the outfit reference.";
                            } else if (outfitSource === "scene image") {
                                outfitClause = "Use the clothing and accessories of the target subject from the scene reference.";
                            } else if (outfitSource === "style image") {
                                outfitClause = "Use the clothing and accessories from the outfit reference.";
                            }
                        }
                        return { hasDefault: true, text: `${EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_BASE}\n\n${outfitClause}` };
                    }

                    if (preset === "scene_reinterpretation") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        return { hasDefault: true, text: EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION };
                    }

                    if ((hasS && !hasSc && !hasO && !hasSt) || (!hasS && !hasSc && !hasO && !hasSt)) {
                        return { hasDefault: false, text: "" };
                    }
                    let basePrompt = "";
                    if (preset === "outfit_transfer") {
                        if (hasO) basePrompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        else if (hasSc) basePrompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                    } else if (preset === "preserve_scene") {
                        if (hasSc) {
                            basePrompt = hasO ? EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT : EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                        } else if (hasO) {
                            basePrompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        }
                    } else if (preset === "style_transfer") {
                        if (hasSt) basePrompt = EASY_DEFAULT_PROMPT_STYLE;
                        else {
                            if (hasSc && hasO) basePrompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT;
                            else if (hasSc) basePrompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                            else if (hasO) basePrompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                        }
                    } else {
                        if (hasSc && hasO) basePrompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT;
                        else if (hasSc) basePrompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE;
                        else if (hasO) basePrompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER;
                    }

                    if (hasSt && preset !== "style_transfer") {
                        if (basePrompt) return { hasDefault: true, text: basePrompt + "\n\n" + EASY_DEFAULT_PROMPT_STYLE };
                        return { hasDefault: true, text: EASY_DEFAULT_PROMPT_STYLE };
                    }

                    if (basePrompt) return { hasDefault: true, text: basePrompt };
                    return { hasDefault: false, text: "" };
                };

                const updatePromptState = () => {
                    if (!useDefaultWidget || !posPromptWidget) return;

                    if (node._isPromptSystemManaged === undefined) {
                        node._isPromptSystemManaged = !!useDefaultWidget.value;
                    }

                    const preset = presetWidget?.value || "balanced";
                    const usesOutfit = !["preserve_scene", "style_transfer", "scene_reinterpretation", "identity_transfer"].includes(preset);
                    const isSceneAutoStyle = ["subject_transfer", "identity_transfer", "scene_reinterpretation"].includes(preset);

                    const subjectInput = node.inputs?.find(i => i.name === "subject");
                    const sceneInput = node.inputs?.find(i => i.name === "scene");
                    const outfitInput = node.inputs?.find(i => i.name === "outfit");
                    const styleInput = node.inputs?.find(i => i.name === "style");

                    const outfitSource = outfitSourceWidget?.value || "outfit image";
                    const styleSource = styleSourceWidget?.value || "style image";

                    const styleSocketIsUsedAsOutfit = usesOutfit && outfitSource === "style image";

                    if (outfitSourceWidget) {
                        outfitSourceWidget.disabled = !usesOutfit;
                    }
                    if (outfitInput) outfitInput.disabled = !usesOutfit;

                    if (styleSourceWidget) {
                        styleSourceWidget.disabled = isSceneAutoStyle;
                        if (isSceneAutoStyle) {
                            styleSourceWidget.label = "Style Source [Auto: Scene]";
                            let tooltipText = "Scene Reinterpretation automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            if (preset === "identity_transfer") {
                                tooltipText = "Identity Transfer automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            } else if (preset === "subject_transfer") {
                                tooltipText = "Subject Transfer automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            }
                            styleSourceWidget.tooltip = tooltipText;
                        } else {
                            delete styleSourceWidget.label;
                            styleSourceWidget.tooltip = "Source image socket to use for style conditioning.";
                        }
                    }

                    if (styleInput) {
                        if (isSceneAutoStyle) {
                            styleInput.disabled = !styleSocketIsUsedAsOutfit;
                        } else {
                            styleInput.disabled = false;
                        }
                    }

                    const hasS = !!(subjectInput && subjectInput.link != null);
                    const hasSc = !!(sceneInput && sceneInput.link != null);
                    const hasRawO = !!(outfitInput && outfitInput.link != null);
                    const hasRawSt = !!(styleInput && styleInput.link != null);

                    const effectiveOutfitSource = usesOutfit ? outfitSource : "none";
                    const effectiveStyleSource = isSceneAutoStyle ? "scene image" : styleSource;

                    const hasO = usesOutfit && (
                        (effectiveOutfitSource === "outfit image" && hasRawO) ||
                        (effectiveOutfitSource === "scene image" && hasSc) ||
                        (effectiveOutfitSource === "style image" && hasRawSt)
                    );

                    const hasSt = isSceneAutoStyle ? hasSc : (
                        (effectiveStyleSource === "style image" && hasRawSt) ||
                        (effectiveStyleSource === "scene image" && hasSc) ||
                        (effectiveStyleSource === "subject image" && hasS)
                    );

                    const isSubjectOnly = hasS && !hasSc && !hasO && !hasSt;
                    const { hasDefault, text } = resolveJsDefaultPrompt(preset, hasS, hasSc, hasO, hasSt, effectiveOutfitSource, effectiveStyleSource);

                    if (isSubjectOnly || !hasDefault) {
                        useDefaultWidget.value = false;
                        useDefaultWidget.disabled = true;
                        posPromptWidget.disabled = false;
                        if (posPromptWidget.inputEl) posPromptWidget.inputEl.readOnly = false;

                        if (node._isPromptSystemManaged && isSubjectOnly) {
                            posPromptWidget.value = "";
                        }
                        node._isPromptSystemManaged = false;
                    } else {
                        useDefaultWidget.disabled = false;

                        if (useDefaultWidget.value && hasDefault) {
                            posPromptWidget.value = text;
                            posPromptWidget.disabled = true;
                            if (posPromptWidget.inputEl) posPromptWidget.inputEl.readOnly = true;
                            node._isPromptSystemManaged = true;
                        } else {
                            posPromptWidget.disabled = false;
                            if (posPromptWidget.inputEl) posPromptWidget.inputEl.readOnly = false;
                            node._isPromptSystemManaged = false;
                        }
                    }
                };

                // Button widget: "Use Preset as Custom"
                if (!node.widgets?.find(w => w.name === "use_preset_as_custom")) {
                    const btn = node.addWidget("button", "Use Preset as Custom", "use_preset_as_custom", () => {
                        if (useDefaultWidget && posPromptWidget) {
                            const preset = presetWidget?.value || "balanced";
                            const subjectInput = node.inputs?.find(i => i.name === "subject");
                            const sceneInput = node.inputs?.find(i => i.name === "scene");
                            const outfitInput = node.inputs?.find(i => i.name === "outfit");
                            const styleInput = node.inputs?.find(i => i.name === "style");

                            const hasS = !!(subjectInput && subjectInput.link != null);
                            const hasSc = !!(sceneInput && sceneInput.link != null);
                            const hasRawO = !!(outfitInput && outfitInput.link != null);
                            const hasRawSt = !!(styleInput && styleInput.link != null);

                            const outfitSource = outfitSourceWidget?.value || "outfit image";
                            const styleSource = styleSourceWidget?.value || "style image";

                            const usesOutfit = !["preserve_scene", "style_transfer", "scene_reinterpretation", "identity_transfer"].includes(preset);
                            const isSceneAutoStyle = ["subject_transfer", "identity_transfer", "scene_reinterpretation"].includes(preset);

                            const effectiveOutfitSource = usesOutfit ? outfitSource : "none";
                            const effectiveStyleSource = isSceneAutoStyle ? "scene image" : styleSource;

                            const hasO = usesOutfit && (
                                (effectiveOutfitSource === "outfit image" && hasRawO) ||
                                (effectiveOutfitSource === "scene image" && hasSc) ||
                                (effectiveOutfitSource === "style image" && hasRawSt)
                            );

                            const hasSt = isSceneAutoStyle ? hasSc : (
                                (effectiveStyleSource === "style image" && hasRawSt) ||
                                (effectiveStyleSource === "scene image" && hasSc) ||
                                (effectiveStyleSource === "subject image" && hasS)
                            );

                            const { hasDefault, text } = resolveJsDefaultPrompt(preset, hasS, hasSc, hasO, hasSt, effectiveOutfitSource, effectiveStyleSource);
                            if (hasDefault && text) {
                                posPromptWidget.value = text;
                            }
                            useDefaultWidget.value = false;
                            node._isPromptSystemManaged = false;
                            posPromptWidget.disabled = false;
                            if (posPromptWidget.inputEl) posPromptWidget.inputEl.readOnly = false;
                            updatePromptState();
                            if (app.graph) app.graph.setDirtyCanvas(true, true);
                        }
                    });
                    if (btn) btn.serialize = false;
                }

                if (useDefaultWidget) {
                    const origCb = useDefaultWidget.callback;
                    useDefaultWidget.callback = function () {
                        if (origCb) origCb.apply(this, arguments);
                        if (!useDefaultWidget.value) {
                            node._isPromptSystemManaged = false;
                        } else {
                            node._isPromptSystemManaged = true;
                        }
                        updatePromptState();
                    };
                }

                [presetWidget, outfitSourceWidget, styleSourceWidget].forEach(w => {
                    if (w) {
                        const orig = w.callback;
                        w.callback = function () {
                            if (orig) orig.apply(this, arguments);
                            updatePromptState();
                        };
                    }
                });

                if (posPromptWidget) {
                    const origPosCb = posPromptWidget.callback;
                    posPromptWidget.callback = function () {
                        if (origPosCb) origPosCb.apply(this, arguments);
                        if (!useDefaultWidget?.value) {
                            node._isPromptSystemManaged = false;
                        }
                    };
                }

                const origConnChange = node.onConnectionsChange;
                node.onConnectionsChange = function () {
                    if (origConnChange) origConnChange.apply(this, arguments);
                    updatePromptState();
                };

                setTimeout(updatePromptState, 20);
            }
        }
    }
});

