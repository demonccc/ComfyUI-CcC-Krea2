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
        }
    }
});
