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

// Global timers for sync testing
global.requestAnimationFrame = (cb) => {
    cb();
    return 1;
};
global.cancelAnimationFrame = (id) => {};

// Evaluate script in function scope with stubbed app
const fn = new Function("app", jsCode);
fn(app);

assert(registeredExtension, "Extension was not registered");

function createMockNode(comfyClass = "CcCKrea2EasyEdit") {
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

    const presetWidget = { name: "preset", value: "balanced", options: { values: [] } };
    const useDefaultWidget = { name: "use_default_prompt", value: true };
    const refSubjWidget = { name: "reference_subject", value: "main subject" };
    const subjDescWidget = { name: "subject_description", value: "main subject" };
    const outfitSourceWidget = { name: "outfit_source", value: "outfit image" };
    const styleSourceWidget = { name: "style_source", value: "style image" };
    const posPromptWidget = { name: "positive_prompt", value: "", inputEl: { readOnly: false, disabled: false, title: "" } };

    node.widgets = [
        presetWidget,
        useDefaultWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        posPromptWidget
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
        presetWidget,
        useDefaultWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        posPromptWidget,
        setPreset
    };
}

async function runTests() {
    // 1. Test Identity Transfer Baseline
    {
        const { node, posPromptWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("identity_transfer");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(posPromptWidget.disabled, true);
        assert.strictEqual(posPromptWidget.readOnly, true);
        assert.strictEqual(posPromptWidget.inputEl.readOnly, true);
        assert.strictEqual(posPromptWidget.inputEl.disabled, true);
        assert(posPromptWidget.value.includes("Replace only the identity"), "Prompt should contain canonical identity transfer text");
    }

    // 2. Test 3 (transfer_identity_test_3)
    {
        const { node, presetWidget, posPromptWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("[Experimental] Identity Test 3 — Scene 4 / Subject 7");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(presetWidget.value, "transfer_identity_test_3");

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert(posPromptWidget.value.includes("Replace only the identity"), "Test 3 canonical prompt preserved");
    }

    // 3. Test 5 (transfer_identity_test_5)
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_5");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Disabled]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 4. Group A & B
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_a_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 5. Group C
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_c_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene Outfit]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 6. Group D
    {
        const { node, outfitSourceWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_d_s2_5_o2_5");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(outfitSourceWidget.label, "Outfit Source [Auto: Scene]");
        assert.strictEqual(outfitSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "outfit").disabled, true);
        assert.strictEqual(styleSourceWidget.label, "Style Source [Disabled]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 7. Normal presets
    {
        const { outfitSourceWidget, setPreset } = createMockNode();
        setPreset("preserve_scene");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(outfitSourceWidget.disabled, true, "preserve_scene disables outfitSourceWidget");

        setPreset("outfit_transfer");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(outfitSourceWidget.disabled, false, "outfit_transfer enables outfitSourceWidget");
    }

    // 8. Workflow restoration sequence
    {
        const { node, useDefaultWidget, posPromptWidget } = createMockNode();
        const info = {
            widgets_values: ["identity_transfer", "main subject", "main subject", "outfit image", "style image", ""]
        };
        node.onConfigure(info);
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true);
        assert.strictEqual(node._isPromptSystemManaged, true);

        // Links appear asynchronously
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;
        node.onConnectionsChange();
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true);
        assert.strictEqual(node._isPromptSystemManaged, true);
        assert.strictEqual(posPromptWidget.disabled, true);
        assert.strictEqual(posPromptWidget.readOnly, true);
        assert(posPromptWidget.value.includes("Replace only the identity"));
    }

    // 9. Manual prompt mode
    {
        const { node, useDefaultWidget, posPromptWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;
        node.onConnectionsChange();
        await new Promise(r => setTimeout(r, 60));

        useDefaultWidget.value = false;
        useDefaultWidget.callback();

        posPromptWidget.value = "User custom prompt text";

        setPreset("transfer_identity_test_a_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(posPromptWidget.value, "User custom prompt text");
        assert.strictEqual(posPromptWidget.disabled, false);
        assert.strictEqual(posPromptWidget.readOnly, false);
    }

    // 10. Migration idempotency
    {
        const { node } = createMockNode();
        const infoLegacy = {
            widgets_values: ["identity_transfer", "main subject", "main subject", "outfit image", "style image"]
        };
        node.onConfigure(infoLegacy);
        assert.strictEqual(infoLegacy.widgets_values[1], true);

        node.onConfigure(infoLegacy);
        assert.strictEqual(infoLegacy.widgets_values[1], true);
        assert.strictEqual(infoLegacy.widgets_values.length, 6); // Not duplicated!

        const infoCurrent = {
            widgets_values: ["identity_transfer", false, "main subject", "main subject", "outfit image", "style image"]
        };
        node.onConfigure(infoCurrent);
        assert.strictEqual(infoCurrent.widgets_values[1], false); // Preserved!
    }

    console.log("All 10 JavaScript frontend behavior tests PASSED successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
