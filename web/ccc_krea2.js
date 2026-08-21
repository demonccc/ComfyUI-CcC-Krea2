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

                const EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT =
                    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n" +
                    "Preserve the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n" +
                    "Dress the transferred {subject} using the clothing and accessories from the {outfit_source}.\n\n" +
                    "Do not preserve the clothing or accessories of the {subject} from the {subject_source} when an explicit outfit source is selected. Use the clothing and accessories from the {outfit_source} instead.\n\n" +
                    "Keep every other person and the rest of the scene unchanged.";

                const EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER =
                    "Transfer only the outfit and accessories from the {outfit_source} to the {subject}.\n\n" +
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
                    "Place the {subject} from the {subject_source} naturally into the {scene_source} wearing the outfit and accessories from the {outfit_source}.\n\n" +
                    "Preserve the {subject} identity, body shape, and body proportions.\n\n" +
                    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n" +
                    "Do not preserve the {subject} clothing.\n\n" +
                    "Fit the transferred outfit and accessories naturally to the {subject} and the scene.\n\n" +
                    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n" +
                    "Do not duplicate accessories.";

                const EASY_DEFAULT_PROMPT_STYLE =
                    "Apply the visual style from the {style_source} while preserving the {subject} identity, content, geometry, framing, and composition.\n\n" +
                    "Transfer only the visual style, including its color palette, texture, lighting character, and overall visual mood.\n\n" +
                    "Do not copy subjects, objects, or scene content from the {style_source}.";

                const EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION =
                    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n" +
                    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n" +
                    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n" +
                    "Adapt the {subject} naturally to the referenced action and environment.\n\n" +
                    "Creatively reinterpret the clothing and accessories worn by the {reference_subject} in the {scene_source} so they are appropriate for the {subject} and the newly generated image. Do not copy the original scene outfit literally.\n\n" +
                    "Generate a coherent new image rather than recreating the source scene exactly.";

                const IDENTITY_TEST_PRESETS = [
                    "identity_transfer",
                    "transfer_identity_test_2",
                    "transfer_identity_test_3",
                    "transfer_identity_test_4",
                    "transfer_identity_test_5",
                    "transfer_identity_test_6",
                ];

                const LEGACY_PRESETS = [
                    "flexible",
                    "balanced",
                    "consistent",
                    "preserve_identity",
                    "max_identity",
                    "identity_transfer",
                    "transfer_identity_test_2",
                    "transfer_identity_test_3",
                    "transfer_identity_test_4",
                    "transfer_identity_test_5",
                    "transfer_identity_test_6",
                    "subject_transfer",
                    "preserve_scene",
                    "outfit_transfer",
                    "style_transfer",
                    "scene_reinterpretation",
                ];

                const migrateLegacyEasyEditWidgets = (info) => {
                    if (!info || !Array.isArray(info.widgets_values)) return;
                    const vals = info.widgets_values;
                    if (vals.length < 2) return;

                    let valAtIndex1 = vals[1];

                    // 1. If index 1 is a preset string (older schema 1 without use_default_prompt), splice use_default_prompt boolean
                    if (typeof valAtIndex1 === "string" && LEGACY_PRESETS.includes(valAtIndex1)) {
                        vals.splice(1, 0, false);
                        node._isPromptSystemManaged = false;
                        valAtIndex1 = false;
                    }

                    // 2. Schema check for reference_subject and subject_description fields (index 3 and 4)
                    // If index 3 is outfit_source (e.g. "outfit image", "scene image", "style image", "none"), splice subject defaults
                    if (typeof vals[3] === "string" && ["none", "outfit image", "scene image", "style image"].includes(vals[3])) {
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

                const origOnConfigure = node.onConfigure;
                node.onConfigure = function (info) {
                    migrateLegacyEasyEditWidgets(info);
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    updatePromptState();
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

                    return template
                        .replace(/\{reference_subject\}/g, refSubj)
                        .replace(/\{subject_description\}/g, subjDesc)
                        .replace(/\{subject\}/g, subj)
                        .replace(/\{scene_source\}/g, sceneSrc)
                        .replace(/\{subject_source\}/g, subjSrc)
                        .replace(/\{outfit_source\}/g, outfitSrc)
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

                    if (preset === "subject_transfer") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        const template = hasO ? EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT : EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT;
                        return { hasDefault: true, text: renderJsEasyPrompt(template, context) };
                    }

                    if (preset === "scene_reinterpretation") {
                        if (!hasS || !hasSc) return { hasDefault: false, text: "" };
                        return { hasDefault: true, text: renderJsEasyPrompt(EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION, context) };
                    }

                    if ((hasS && !hasSc && !hasO && !hasSt) || (!hasS && !hasSc && !hasO && !hasSt)) {
                        return { hasDefault: false, text: "" };
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

                const updatePromptState = () => {
                    if (!useDefaultWidget || !posPromptWidget) return;

                    if (node._isPromptSystemManaged === undefined) {
                        node._isPromptSystemManaged = !!useDefaultWidget.value;
                    }

                    const preset = presetWidget?.value || "balanced";
                    const usesOutfit = !["preserve_scene", "style_transfer", "scene_reinterpretation", ...IDENTITY_TEST_PRESETS].includes(preset);
                    const isStyleDisabled = preset === "transfer_identity_test_5";
                    const isSceneAutoStyle = ["subject_transfer", "scene_reinterpretation", "identity_transfer", "transfer_identity_test_2", "transfer_identity_test_3", "transfer_identity_test_4", "transfer_identity_test_6"].includes(preset);

                    const subjectInput = node.inputs?.find(i => i.name === "subject");
                    const sceneInput = node.inputs?.find(i => i.name === "scene");
                    const outfitInput = node.inputs?.find(i => i.name === "outfit");
                    const styleInput = node.inputs?.find(i => i.name === "style");

                    const outfitSource = outfitSourceWidget?.value || "outfit image";
                    const styleSource = styleSourceWidget?.value || "style image";
                    const refSubjVal = refSubjWidget?.value || "main subject";
                    const subjDescVal = subjDescWidget?.value || "main subject";

                    const styleSocketIsUsedAsOutfit = usesOutfit && outfitSource === "style image";

                    if (outfitSourceWidget) {
                        outfitSourceWidget.disabled = !usesOutfit;
                    }
                    if (outfitInput) outfitInput.disabled = !usesOutfit;

                    if (styleSourceWidget) {
                        if (isStyleDisabled) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Disabled]";
                            styleSourceWidget.tooltip = "Transfer Identity Test 5 intentionally disables Scene style reinforcement.";
                        } else if (isSceneAutoStyle) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Auto: Scene]";
                            let tooltipText = "Scene Reinterpretation automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            if (IDENTITY_TEST_PRESETS.includes(preset)) {
                                tooltipText = "Identity Transfer automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            } else if (preset === "subject_transfer") {
                                tooltipText = "Subject Transfer automatically uses the Scene reference for style integration. The stored Style Source value is preserved for other presets.";
                            }
                            styleSourceWidget.tooltip = tooltipText;
                        } else {
                            styleSourceWidget.disabled = false;
                            delete styleSourceWidget.label;
                            styleSourceWidget.tooltip = "Source image socket to use for style conditioning.";
                        }
                    }

                    if (styleInput) {
                        if (isStyleDisabled) {
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

                    const effectiveOutfitSource = usesOutfit ? outfitSource : "none";
                    const effectiveStyleSource = isStyleDisabled ? "none" : (isSceneAutoStyle ? "scene image" : styleSource);

                    const hasO = usesOutfit && (
                        (effectiveOutfitSource === "outfit image" && hasRawO) ||
                        (effectiveOutfitSource === "scene image" && hasSc) ||
                        (effectiveOutfitSource === "style image" && hasRawSt)
                    );

                    const hasSt = isStyleDisabled ? false : (isSceneAutoStyle ? hasSc : (
                        (effectiveStyleSource === "style image" && hasRawSt) ||
                        (effectiveStyleSource === "scene image" && hasSc) ||
                        (effectiveStyleSource === "subject image" && hasS)
                    ));

                    const isSubjectOnly = hasS && !hasSc && !hasO && !hasSt;
                    const { hasDefault, text } = resolveJsDefaultPrompt(
                        preset, hasS, hasSc, hasO, hasSt, effectiveOutfitSource, effectiveStyleSource, refSubjVal, subjDescVal
                    );

                    // Subject description widgets are enabled ONLY when use_default_prompt is ON and preset supports subject fields
                    const supportsSubjectFields = [...IDENTITY_TEST_PRESETS, "subject_transfer", "outfit_transfer"].includes(preset);
                    const enableSubjectFields = useDefaultWidget.value && supportsSubjectFields;
                    if (refSubjWidget) refSubjWidget.disabled = !enableSubjectFields;
                    if (subjDescWidget) subjDescWidget.disabled = !enableSubjectFields;

                    if (isSubjectOnly || !hasDefault) {
                        useDefaultWidget.value = false;
                        useDefaultWidget.disabled = true;
                        posPromptWidget.disabled = false;
                        if (posPromptWidget.inputEl) {
                            posPromptWidget.inputEl.readOnly = false;
                            posPromptWidget.inputEl.title = "";
                        }

                        if (node._isPromptSystemManaged && isSubjectOnly) {
                            posPromptWidget.value = "";
                        }
                        node._isPromptSystemManaged = false;
                    } else {
                        useDefaultWidget.disabled = false;

                        if (useDefaultWidget.value && hasDefault) {
                            posPromptWidget.value = text;
                            posPromptWidget.disabled = false;
                            if (posPromptWidget.inputEl) {
                                posPromptWidget.inputEl.readOnly = true;
                                posPromptWidget.inputEl.title = "Resolved preset prompt. Read-only while Use Default Prompt is enabled.";
                            }
                            node._isPromptSystemManaged = true;
                        } else {
                            posPromptWidget.disabled = false;
                            if (posPromptWidget.inputEl) {
                                posPromptWidget.inputEl.readOnly = false;
                                posPromptWidget.inputEl.title = "";
                            }
                            node._isPromptSystemManaged = false;
                        }
                    }
                };



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

                [presetWidget, refSubjWidget, subjDescWidget, outfitSourceWidget, styleSourceWidget].forEach(w => {
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

