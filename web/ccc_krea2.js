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
                    // Group A
                    "transfer_identity_test_a_4_4",
                    "transfer_identity_test_a_4_5",
                    "transfer_identity_test_a_4_6",
                    // Group B
                    "transfer_identity_test_b_2_5_4",
                    "transfer_identity_test_b_2_5_5",
                    "transfer_identity_test_b_2_5_6",
                    // Group C
                    "transfer_identity_test_c_4_4",
                    "transfer_identity_test_c_4_5",
                    "transfer_identity_test_c_4_6",
                    "transfer_identity_test_c_4_7",
                    "transfer_identity_test_c_2_5_4",
                    "transfer_identity_test_c_2_5_5",
                    "transfer_identity_test_c_2_5_6",
                    "transfer_identity_test_c_2_5_9",
                    // Group D
                    "transfer_identity_test_d_s2_5_o2_5",
                    "transfer_identity_test_d_s2_5_o4",
                    "transfer_identity_test_d_s4_o4",
                    "transfer_identity_test_d_s5_o4",
                    "transfer_identity_test_d_s6_o4",
                    "transfer_identity_test_d_s7_o4",
                ];

                const GROUP_A_PRESETS = [
                    "transfer_identity_test_a_4_4",
                    "transfer_identity_test_a_4_5",
                    "transfer_identity_test_a_4_6",
                ];
                const GROUP_B_PRESETS = [
                    "transfer_identity_test_b_2_5_4",
                    "transfer_identity_test_b_2_5_5",
                    "transfer_identity_test_b_2_5_6",
                ];
                const GROUP_C_PRESETS = [
                    "transfer_identity_test_c_4_4",
                    "transfer_identity_test_c_4_5",
                    "transfer_identity_test_c_4_6",
                    "transfer_identity_test_c_4_7",
                    "transfer_identity_test_c_2_5_4",
                    "transfer_identity_test_c_2_5_5",
                    "transfer_identity_test_c_2_5_6",
                    "transfer_identity_test_c_2_5_9",
                ];
                const GROUP_D_PRESETS = [
                    "transfer_identity_test_d_s2_5_o2_5",
                    "transfer_identity_test_d_s2_5_o4",
                    "transfer_identity_test_d_s4_o4",
                    "transfer_identity_test_d_s5_o4",
                    "transfer_identity_test_d_s6_o4",
                    "transfer_identity_test_d_s7_o4",
                ];

                const PRESET_DISPLAY_LABELS = {
                    "flexible": "Flexible",
                    "balanced": "Balanced",
                    "consistent": "Consistent",
                    "preserve_identity": "Preserve Identity",
                    "max_identity": "Max Identity",
                    "identity_transfer": "Identity Transfer",
                    "subject_transfer": "Subject Transfer",
                    "preserve_scene": "Preserve Scene",
                    "outfit_transfer": "Outfit Transfer",
                    "style_transfer": "Style Transfer",
                    "scene_reinterpretation": "Scene Reinterpretation",
                    // Experimental calibration presets
                    "transfer_identity_test_2": "[Experimental] Identity Test 2",
                    "transfer_identity_test_3": "[Experimental] Identity Test 3 — Scene 4 / Subject 7",
                    "transfer_identity_test_4": "[Experimental] Identity Test 4 — Scene 2.5 / Subject 9",
                    "transfer_identity_test_5": "[Experimental] Identity Test 5",
                    "transfer_identity_test_6": "[Experimental] Identity Test 6",
                    // Group A
                    "transfer_identity_test_a_4_4": "[Experimental A] Scene 4 / Subject 4",
                    "transfer_identity_test_a_4_5": "[Experimental A] Scene 4 / Subject 5",
                    "transfer_identity_test_a_4_6": "[Experimental A] Scene 4 / Subject 6",
                    // Group B
                    "transfer_identity_test_b_2_5_4": "[Experimental B] Scene 2.5 / Subject 4",
                    "transfer_identity_test_b_2_5_5": "[Experimental B] Scene 2.5 / Subject 5",
                    "transfer_identity_test_b_2_5_6": "[Experimental B] Scene 2.5 / Subject 6",
                    // Group C
                    "transfer_identity_test_c_4_4": "[Experimental C] Scene 4 / Subject 4 + Outfit Style",
                    "transfer_identity_test_c_4_5": "[Experimental C] Scene 4 / Subject 5 + Outfit Style",
                    "transfer_identity_test_c_4_6": "[Experimental C] Scene 4 / Subject 6 + Outfit Style",
                    "transfer_identity_test_c_4_7": "[Experimental C] Scene 4 / Subject 7 + Outfit Style",
                    "transfer_identity_test_c_2_5_4": "[Experimental C] Scene 2.5 / Subject 4 + Outfit Style",
                    "transfer_identity_test_c_2_5_5": "[Experimental C] Scene 2.5 / Subject 5 + Outfit Style",
                    "transfer_identity_test_c_2_5_6": "[Experimental C] Scene 2.5 / Subject 6 + Outfit Style",
                    "transfer_identity_test_c_2_5_9": "[Experimental C] Scene 2.5 / Subject 9 + Outfit Style",
                    // Group D
                    "transfer_identity_test_d_s2_5_o2_5": "[Experimental D] Subject 2.5 / Outfit 2.5",
                    "transfer_identity_test_d_s2_5_o4": "[Experimental D] Subject 2.5 / Outfit 4",
                    "transfer_identity_test_d_s4_o4": "[Experimental D] Subject 4 / Outfit 4",
                    "transfer_identity_test_d_s5_o4": "[Experimental D] Subject 5 / Outfit 4",
                    "transfer_identity_test_d_s6_o4": "[Experimental D] Subject 6 / Outfit 4",
                    "transfer_identity_test_d_s7_o4": "[Experimental D] Subject 7 / Outfit 4",
                };

                const DISPLAY_TO_PRESET_ID = Object.fromEntries(
                    Object.entries(PRESET_DISPLAY_LABELS).map(([k, v]) => [v, k])
                );

                const LEGACY_PRESETS = [
                    "flexible",
                    "balanced",
                    "consistent",
                    "preserve_identity",
                    "max_identity",
                    "identity_transfer",
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
                    if (typeof valAtIndex1 === "string" && (LEGACY_PRESETS.includes(valAtIndex1) || valAtIndex1.startsWith("transfer_identity_test_"))) {
                        vals.splice(1, 0, true);
                        node._isPromptSystemManaged = true;
                        valAtIndex1 = true;
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

                let updateTimer = null;
                let updateRaf = null;

                const schedulePromptStateUpdate = () => {
                    updatePromptState();

                    if (updateTimer) clearTimeout(updateTimer);
                    if (updateRaf && typeof requestAnimationFrame === "function") {
                        cancelAnimationFrame(updateRaf);
                        updateRaf = null;
                    }

                    const runDeferredPasses = () => {
                        updatePromptState();
                        updateTimer = setTimeout(() => {
                            updatePromptState();
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

                    if (!baseTemplate) return { hasDefault: false, text: "" };
                    return { hasDefault: true, text: renderJsEasyPrompt(baseTemplate, context) };
                };

                const updatePromptState = () => {
                    const preset = presetWidget ? (DISPLAY_TO_PRESET_ID[presetWidget.value] || presetWidget.value) : "flexible";
                    const isIdentityTransfer = IDENTITY_TEST_PRESETS.includes(preset);
                    const isSubjectTransfer = preset === "subject_transfer";
                    const isPreserveScene = preset === "preserve_scene";
                    const isOutfitTransfer = preset === "outfit_transfer";
                    const isStyleTransfer = preset === "style_transfer";
                    const isSceneReinterpretation = preset === "scene_reinterpretation";

                    const usesOutfit = isSubjectTransfer || isPreserveScene || isOutfitTransfer || isStyleTransfer || isSceneReinterpretation;
                    const usesStyle = isStyleTransfer;

                    if (outfitSourceWidget) outfitSourceWidget.disabled = !usesOutfit;
                    if (styleSourceWidget) styleSourceWidget.disabled = !usesStyle;

                    const outfitSource = outfitSourceWidget ? outfitSourceWidget.value : "outfit image";
                    const styleSource = styleSourceWidget ? styleSourceWidget.value : "style image";

                    const isSceneAutoOutfit = (isSubjectTransfer || isPreserveScene || isSceneReinterpretation) && outfitSource === "scene image";
                    const isSceneOutfitStyle = GROUP_C_PRESETS.includes(preset);
                    const isSceneAutoStyle = isPreserveScene || isSceneReinterpretation || (isStyleTransfer && styleSource === "scene image");
                    const isStyleDisabled = isIdentityTransfer && !isSceneOutfitStyle;

                    const refSubjVal = refSubjWidget ? refSubjWidget.value : "main subject";
                    const subjDescVal = subjDescWidget ? subjDescWidget.value : "main subject";

                    const subjectInput = node.inputs?.find(i => i.name === "subject");
                    const sceneInput = node.inputs?.find(i => i.name === "scene");
                    const outfitInput = node.inputs?.find(i => i.name === "outfit");
                    const styleInput = node.inputs?.find(i => i.name === "style");

                    if (styleSourceWidget) {
                        if (isStyleDisabled) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Disabled]";
                            styleSourceWidget.tooltip = "Style conditioning is disabled for this preset.";
                        } else if (isSceneOutfitStyle) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Auto: Scene Outfit]";
                            styleSourceWidget.tooltip = "Uses the Scene image as outfit-focused style conditioning to reinforce the target subject's clothing and accessories.";
                        } else if (isSceneAutoStyle) {
                            styleSourceWidget.disabled = true;
                            styleSourceWidget.label = "Style Source [Auto: Scene]";
                            styleSourceWidget.tooltip = "Uses the Scene image as style conditioning.";
                        } else {
                            styleSourceWidget.disabled = false;
                            styleSourceWidget.label = "Style Source";
                            styleSourceWidget.tooltip = "Select the image to use as style conditioning source.";
                        }
                    }

                    if (styleInput) {
                        const styleSocketIsUsedAsOutfit = usesOutfit && outfitSource === "style image";
                        if (isStyleDisabled) {
                            styleInput.disabled = !styleSocketIsUsedAsOutfit;
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
                    const supportsSubjectFields = [...IDENTITY_TEST_PRESETS, "subject_transfer", "outfit_transfer"].includes(preset);
                    const enableSubjectFields = (useDefaultWidget ? useDefaultWidget.value : false) && supportsSubjectFields;
                    if (refSubjWidget) refSubjWidget.disabled = !enableSubjectFields;
                    if (subjDescWidget) subjDescWidget.disabled = !enableSubjectFields;

                    if (isSubjectOnly || !hasDefault) {
                        if (useDefaultWidget) {
                            useDefaultWidget.disabled = true;
                        }
                        posPromptWidget.disabled = false;
                        posPromptWidget.readOnly = false;
                        if (posPromptWidget.inputEl) {
                            posPromptWidget.inputEl.readOnly = false;
                            posPromptWidget.inputEl.disabled = false;
                            posPromptWidget.inputEl.title = "";
                        }

                        if (node._isPromptSystemManaged && isSubjectOnly) {
                            posPromptWidget.value = "";
                            node._isPromptSystemManaged = false;
                        }
                    } else {
                        if (useDefaultWidget) {
                            useDefaultWidget.disabled = false;
                        }

                        if (useDefaultWidget && useDefaultWidget.value && hasDefault) {
                            posPromptWidget.value = text;
                            posPromptWidget.disabled = true;
                            posPromptWidget.readOnly = true;
                            if (posPromptWidget.inputEl) {
                                posPromptWidget.inputEl.readOnly = true;
                                posPromptWidget.inputEl.disabled = true;
                                posPromptWidget.inputEl.title = "Resolved preset prompt. Disable Use Default Prompt to edit manually.";
                            }
                            node._isPromptSystemManaged = true;
                        } else {
                            posPromptWidget.disabled = false;
                            posPromptWidget.readOnly = false;
                            if (posPromptWidget.inputEl) {
                                posPromptWidget.inputEl.readOnly = false;
                                posPromptWidget.inputEl.disabled = false;
                                posPromptWidget.inputEl.title = "";
                            }
                            node._isPromptSystemManaged = false;
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
                            const raw = typeof origValues === "function" ? origValues() : (origValues || Object.keys(PRESET_DISPLAY_LABELS));
                            return raw.map(v => PRESET_DISPLAY_LABELS[v] || v);
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
                    schedulePromptStateUpdate();
                };

                schedulePromptStateUpdate();
            }
        }
    }
});
