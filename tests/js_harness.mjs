import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import assert from "assert";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const jsPath = path.resolve(__dirname, "../web/ccc_krea2.js");
let jsCode = fs.readFileSync(jsPath, "utf-8");

// Stub out export / import
jsCode = jsCode.replace(/import\s+\{\s*app\s*\}\s+from\s+["'].*?["'];?/, "");

let registeredExtension = null;
const app = {
    registerExtension(ext) {
        registeredExtension = ext;
    }
};

let rafCallbacks = [];
global.requestAnimationFrame = (cb) => {
    rafCallbacks.push(cb);
    cb();
    return rafCallbacks.length;
};
global.cancelAnimationFrame = (id) => {};

// Evaluate script in function scope with stubbed app
const fn = new Function("app", jsCode);
fn(app);

assert(registeredExtension, "Extension was not registered");

function createMockNode(comfyClass = "CcCKrea2EasyEdit", fixtureType = "legacy") {
    const node = {
        comfyClass,
        inputs: [
            { name: "subject", link: null },
            { name: "scene", link: null },
            { name: "outfit", link: null },
            { name: "style", link: null }
        ],
        widgets: [],
        setDirtyCanvasCalls: 0,
        setDirtyCanvas(a, b) {
            this.setDirtyCanvasCalls++;
        }
    };

    let posPromptWidget;
    if (fixtureType === "legacy") {
        posPromptWidget = {
            name: "positive_prompt",
            value: "",
            inputEl: { tagName: "TEXTAREA", readOnly: false, disabled: false, title: "", style: {} }
        };
    } else if (fixtureType === "modern") {
        let _storeValue = "";
        posPromptWidget = {
            name: "positive_prompt",
            get value() { return _storeValue; },
            set value(v) { _storeValue = v; },
            valueStore: {
                set(v) { _storeValue = v; }
            },
            component: {
                setValue(v) { _storeValue = v; }
            }
        };
    } else if (fixtureType === "delayed") {
        posPromptWidget = {
            name: "positive_prompt",
            value: ""
        };
    }

    const useDefaultWidget = { name: "use_default_prompt", value: true };
    const presetWidget = { name: "preset", value: "balanced", options: { values: [] } };
    const refSubjWidget = { name: "reference_subject", value: "main subject" };
    const subjDescWidget = { name: "subject_description", value: "main subject" };
    const outfitSourceWidget = {
        name: "outfit_source",
        value: "outfit image",
        options: { values: ["none", "subject image", "scene image", "outfit image", "style image"] }
    };
    const styleSourceWidget = {
        name: "style_source",
        value: "none",
        options: { values: ["none", "scene image", "subject image", "style image"] }
    };
    const patchWidget = {
        name: comfyClass === "CcCKrea2EasyEditOstris" ? "apply_ostris_edit_patch" : "apply_krea2_edit_patch",
        value: true
    };

    node.widgets = [
        posPromptWidget,
        useDefaultWidget,
        presetWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        patchWidget
    ];

    registeredExtension.nodeCreated(node);

    const setPreset = (val) => {
        presetWidget.value = val;
        if (presetWidget.callback) {
            presetWidget.callback(val);
        }
    };

    return {
        node,
        posPromptWidget,
        useDefaultWidget,
        presetWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        patchWidget,
        setPreset
    };
}

async function runTests() {
    const nodeClasses = ["CcCKrea2EasyEdit", "CcCKrea2EasyEditOstris"];

    for (const nodeClass of nodeClasses) {
        // 1. Test Identity Transfer Baseline (Legacy Fixture)
        {
            const { node, posPromptWidget, styleSourceWidget, setPreset } = createMockNode(nodeClass, "legacy");
            node.inputs.find(i => i.name === "subject").link = 1;
            node.inputs.find(i => i.name === "scene").link = 2;

            setPreset("identity_transfer");
            await new Promise(r => setTimeout(r, 60));

            assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
            assert.strictEqual(styleSourceWidget.disabled, true);
            assert.strictEqual(posPromptWidget.disabled, false, "Widget object must NOT be disabled so ComfyUI doesn't hide it!");
            assert.strictEqual(posPromptWidget.readOnly, true);
            assert.strictEqual(posPromptWidget.inputEl.readOnly, true);
            assert.strictEqual(posPromptWidget.inputEl.disabled, false, "DOM inputEl must NOT be disabled so text remains scrollable!");
            assert(posPromptWidget.value.includes("Replace only the identity"), "Prompt should contain canonical identity transfer text");
        }

        // 2. Modern Component Widget Test (No inputEl, store-backed)
        {
            const { node, useDefaultWidget, posPromptWidget, setPreset } = createMockNode(nodeClass, "modern");
            node.inputs.find(i => i.name === "subject").link = 1;
            node.inputs.find(i => i.name === "scene").link = 2;

            setPreset("identity_transfer");
            await new Promise(r => setTimeout(r, 60));

            assert.strictEqual(posPromptWidget.disabled, false, "Modern widget must remain visible (disabled=false)");
            assert.strictEqual(posPromptWidget.readOnly, true, "Modern widget must be readOnly=true");
            assert(posPromptWidget.value.includes("Replace only the identity"), "Canonical prompt assigned to widget.value");
            assert.strictEqual(node._isPromptSystemManaged, true);

            // User edit attempt must be rejected/restored while managed
            if (posPromptWidget.callback) {
                posPromptWidget.callback("User attempted edit");
            }
            assert(posPromptWidget.value.includes("Replace only the identity"), "User edit attempt restored to canonical prompt!");

            // Scene Reinterpretation promoted from the useful Test 5 routing
            setPreset("Scene Reinterpretation");
            await new Promise(r => setTimeout(r, 60));
            assert(posPromptWidget.value.includes("Create a new image"), "Scene Reinterpretation prompt assigned");

            // Switch to Manual Mode
            useDefaultWidget.value = false;
            useDefaultWidget.callback();
            await new Promise(r => setTimeout(r, 60));

            assert.strictEqual(posPromptWidget.disabled, false);
            assert.strictEqual(posPromptWidget.readOnly, false);
            assert.strictEqual(node._isPromptSystemManaged, false);

            // User edit in Manual Mode must persist
            posPromptWidget.value = "User custom prompt in manual mode";
            if (posPromptWidget.callback) {
                posPromptWidget.callback("User custom prompt in manual mode");
            }
            assert.strictEqual(posPromptWidget.value, "User custom prompt in manual mode");

            // Preset change in Manual Mode must NOT overwrite user custom prompt
            setPreset("Scene Reinterpretation");
            await new Promise(r => setTimeout(r, 60));
            assert.strictEqual(posPromptWidget.value, "User custom prompt in manual mode");

            // Toggle back to Managed Mode
            useDefaultWidget.value = true;
            useDefaultWidget.callback();
            await new Promise(r => setTimeout(r, 60));

            assert.strictEqual(node._isPromptSystemManaged, true);
            assert(posPromptWidget.value.includes("Create a new image"), "Scene Reinterpretation prompt returned on toggling default prompt ON");
        }

        // 3. Delayed Render Fixture Test
        {
            const { node, posPromptWidget, setPreset } = createMockNode(nodeClass, "delayed");
            node.inputs.find(i => i.name === "subject").link = 1;
            node.inputs.find(i => i.name === "scene").link = 2;

            setPreset("identity_transfer");
            await new Promise(r => setTimeout(r, 60));

            assert(posPromptWidget.value.includes("Replace only the identity"), "Prompt stored in widget.value prior to DOM render");
            assert.strictEqual(posPromptWidget.readOnly, true);

            // Component element mounts after initial resolution
            posPromptWidget.element = { tagName: "TEXTAREA", readOnly: false, disabled: false, title: "", style: {} };
            
            // Trigger deferred frame pass
            node.onConnectionsChange();
            await new Promise(r => setTimeout(r, 60));

            assert.strictEqual(posPromptWidget.element.readOnly, true, "Delayed element received readOnly=true");
            assert.strictEqual(posPromptWidget.element.disabled, false, "Delayed element received disabled=false");
        }
    }

    // 4. Scene Reinterpretation allows manual Style and Outfit selection
    {
        const { node, outfitSourceWidget, styleSourceWidget, setPreset } = createMockNode("CcCKrea2EasyEdit", "legacy");
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("scene_reinterpretation");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.disabled, false);
        assert.strictEqual(outfitSourceWidget.disabled, false);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, false);
        assert.strictEqual(node.inputs.find(i => i.name === "outfit").disabled, false);
    }

    // 5. Flexible Subject Transfer exposes independent semantic Outfit and artistic Style selectors
    {
        const { node, outfitSourceWidget, styleSourceWidget, posPromptWidget, setPreset } = createMockNode("CcCKrea2EasyEdit", "legacy");
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("flexible_subject_transfer_1");
        await new Promise(r => setTimeout(r, 60));

        assert.deepStrictEqual(outfitSourceWidget.options.values, ["none", "subject image", "scene image"]);
        assert.strictEqual(outfitSourceWidget.value, "subject image");
        assert.strictEqual(outfitSourceWidget.disabled, false);
        assert.strictEqual(styleSourceWidget.value, "none");
        assert.strictEqual(styleSourceWidget.disabled, false);
        assert.strictEqual(node.inputs.find(i => i.name === "outfit").disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, false);
        assert(posPromptWidget.value.includes("Preserve the face, body shape, body proportions, clothing, and accessories"));
    }

    // 7. Workflow restoration sequence with modern widget ordering
    {
        const { node, useDefaultWidget, posPromptWidget } = createMockNode("CcCKrea2EasyEdit", "modern");
        const infoOld = {
            widgets_values: ["", "identity_transfer", "outfit image", "style image", true]
        };
        node.onConfigure(infoOld);
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true, "use_default_prompt preference migrated to true");
        assert.strictEqual(infoOld.widgets_values[0], "");
        assert.strictEqual(infoOld.widgets_values[1], true);

        // Links appear asynchronously
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;
        node.onConnectionsChange();
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true);
        assert.strictEqual(node._isPromptSystemManaged, true, "Managed state active after links connected");
        assert.strictEqual(posPromptWidget.disabled, false, "Widget object not disabled");
        assert.strictEqual(posPromptWidget.readOnly, true, "Widget readOnly when managed");
        assert(posPromptWidget.value.includes("Replace only the identity"), "Canonical prompt appears after links connected");
    }

    // 8. Preset display ordering test
    {
        const { presetWidget } = createMockNode("CcCKrea2EasyEdit", "legacy");
        assert(presetWidget.options && typeof presetWidget.options.values === "function");
        const displayedValues = presetWidget.options.values();

        const expectedStableFirst = [
            "Flexible",
            "Balanced",
            "Consistent",
            "Preserve Identity",
            "Max Identity",
            "Identity Transfer",
            "Subject Transfer 1",
            "Subject Transfer 2",
            "Flexible Subject Transfer 1",
            "Flexible Subject Transfer 2",
            "Preserve Scene",
            "Outfit Transfer",
            "Style Transfer",
            "Scene Reinterpretation"
        ];

        const actualStableFirst = displayedValues.slice(0, 14);
        assert.deepStrictEqual(actualStableFirst, expectedStableFirst, "First 14 presets must be stable presets in exact order");
        assert.strictEqual(displayedValues.length, 14, "No experimental Test 5 preset should remain");
    }

    console.log("All modern & legacy JavaScript frontend tests PASSED successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
