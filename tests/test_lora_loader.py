"""Tests for the standalone Krea2 10-slot LoRA loader (no ComfyUI runtime required)."""

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TenSlotLoRATests(unittest.TestCase):
    def test_defaults_and_limits(self):
        # Load just the module with mocked relative dependencies.
        package = types.ModuleType("ccc_krea2")
        package.__path__ = [str(ROOT / "ccc_krea2")]
        constants = types.ModuleType("ccc_krea2.constants")
        constants.NODE_CATEGORY = "CcC/Krea2"
        lora = types.ModuleType("ccc_krea2.lora")
        lora.get_lora_names = lambda: ["None", "demo.safetensors"]

        class Loader:
            def __init__(self, index):
                self.index = index

            def load(self, model, name, strength):
                return model + [(self.index, name, strength)]

        lora.SlotLoraLoader = Loader
        with patch.dict(sys.modules, {"ccc_krea2": package, "ccc_krea2.constants": constants, "ccc_krea2.lora": lora}):
            spec = importlib.util.spec_from_file_location(
                "ccc_krea2.lora_loader", ROOT / "ccc_krea2" / "lora_loader.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            node = module.CcCKrea2LoRALoader()
            required = node.INPUT_TYPES()["required"]
            self.assertEqual(len(required), 31)
            for i in range(1, 11):
                self.assertEqual(required[f"lora_{i}_enabled"][1]["default"], False)
                strength = required[f"lora_{i}_strength"][1]
                self.assertEqual((strength["min"], strength["max"], strength["default"]), (-10.0, 10.0, 0.0))
            self.assertEqual(node.apply_loras([]), ([],))
            result, = node.apply_loras(
                [],
                lora_1_name="demo.safetensors", lora_1_enabled=True, lora_1_strength=1.5,
                lora_2_name="demo.safetensors", lora_2_enabled=True, lora_2_strength=0.0,
                lora_10_name="demo.safetensors", lora_10_enabled=True, lora_10_strength=-2.0,
            )
            self.assertEqual(result, [(1, "demo.safetensors", 1.5), (10, "demo.safetensors", -2.0)])


if __name__ == "__main__":
    unittest.main()
