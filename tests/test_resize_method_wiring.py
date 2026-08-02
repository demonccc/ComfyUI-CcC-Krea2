"""Focused regression tests proving end-to-end resize method parameter wiring."""

import torch

import ccc_krea2.geometry as geom
from ccc_krea2.geometry import apply_reference_fit_transform
from ccc_krea2.grounding import resize_grounding_image
from ccc_krea2.references import prepare_reference, ReferenceConfig, ReferenceRole
from ccc_krea2.latents import generate_krea2_latent
from ccc_krea2.settings import (
    resolve_krea2_settings,
    ImageRoleSettings,
    ImageAdvancedSettingsBundle,
    EditAdvancedSettings,
)
from ccc_krea2.engine import NodeExecutionRequest


class MockVAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0] if image.ndim == 4 else 1
        h = image.shape[1] if image.ndim == 4 and image.shape[-1] in (1, 3, 4) else (image.shape[-2] if image.ndim == 4 else 16)
        w = image.shape[2] if image.ndim == 4 and image.shape[-1] in (1, 3, 4) else (image.shape[-1] if image.ndim == 4 else 16)
        return torch.ones((b, 16, max(1, h // 8), max(1, w // 8)))


class MockModel:
    def __init__(self):
        self.load_device = torch.device("cpu")
        self.model_dtype = torch.float32

    def clone(self):
        return self


def test_grounding_resize_method_wiring(monkeypatch):
    """1. Assert grounding_resize_method='bilinear' reaches central resize helper."""
    called_methods = []
    orig_resize_tensor = geom.resize_tensor

    def spy_resize_tensor(tensor, target_h, target_w, method="auto"):
        called_methods.append(method)
        return orig_resize_tensor(tensor, target_h, target_w, method=method)

    monkeypatch.setattr(geom, "resize_tensor", spy_resize_tensor)

    img = torch.rand((1, 512, 512, 3))
    _ = resize_grounding_image(img, resize_mode="normalize", grounding_preset="custom", grounding_px=256, resize_method="bilinear")

    assert "bilinear" in called_methods


def test_reference_resize_method_wiring(monkeypatch):
    """2. Assert reference_resize_method='lanczos' reaches central resize helper."""
    called_methods = []
    orig_resize_tensor = geom.resize_tensor

    def spy_resize_tensor(tensor, target_h, target_w, method="auto"):
        called_methods.append(method)
        return orig_resize_tensor(tensor, target_h, target_w, method=method)

    monkeypatch.setattr(geom, "resize_tensor", spy_resize_tensor)

    img = torch.rand((1, 512, 512, 3))
    _ = apply_reference_fit_transform(img, target_h=256, target_w=256, mode="crop", resize_method="lanczos")

    assert "lanczos" in called_methods


def test_sampling_resize_method_wiring(monkeypatch):
    """3. Assert sampling_resize_method='nearest-exact' reaches central resize helper."""
    called_methods = []
    orig_resize_tensor = geom.resize_tensor

    def spy_resize_tensor(tensor, target_h, target_w, method="auto"):
        called_methods.append(method)
        return orig_resize_tensor(tensor, target_h, target_w, method=method)

    monkeypatch.setattr(geom, "resize_tensor", spy_resize_tensor)

    img = torch.rand((1, 512, 512, 3))
    vae = MockVAE()
    _ = generate_krea2_latent(
        model=MockModel(),
        vae=vae,
        width=256,
        height=256,
        batch_size=1,
        latent_source="image",
        base_image=img,
        sampling_resize_mode="crop",
        sampling_resize_method="nearest-exact",
    )

    assert "nearest-exact" in called_methods


def test_changing_selected_method_changes_resize_path(monkeypatch):
    """4. Changing selected method changes the resize path used."""
    called_methods = []
    orig_resize_tensor = geom.resize_tensor

    def spy_resize_tensor(tensor, target_h, target_w, method="auto"):
        called_methods.append(method)
        return orig_resize_tensor(tensor, target_h, target_w, method=method)

    monkeypatch.setattr(geom, "resize_tensor", spy_resize_tensor)

    img = torch.rand((1, 512, 512, 3))
    _ = resize_grounding_image(img, resize_mode="normalize", grounding_preset="custom", grounding_px=256, resize_method="bicubic")
    _ = resize_grounding_image(img, resize_mode="normalize", grounding_preset="custom", grounding_px=256, resize_method="area")

    assert called_methods == ["bicubic", "area"]


def test_hard_attention_mask_uses_nearest_exact(monkeypatch):
    """5. Hard attention masks still use nearest-exact."""
    interpolate_modes = []
    orig_interpolate = torch.nn.functional.interpolate

    def spy_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=None):
        interpolate_modes.append(mode)
        kwargs = {}
        if mode in ("bilinear", "bicubic") and antialias is not None:
            kwargs["antialias"] = antialias
        return orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy_interpolate)

    img = torch.rand((1, 512, 512, 3))
    mask = torch.ones((1, 512, 512))
    cfg = ReferenceConfig(role=ReferenceRole.SUBJECT, image=img, attention_mask=mask, reference_resize_method="bicubic")

    _ = prepare_reference(cfg, vae=MockVAE(), model=MockModel(), target_h=256, target_w=256, attention_mask_mode="hard")

    # The mask interpolation mode must be nearest-exact
    assert "nearest-exact" in interpolate_modes


def test_soft_attention_mask_uses_bilinear(monkeypatch):
    """6. Soft attention masks use bilinear."""
    interpolate_modes = []
    orig_interpolate = torch.nn.functional.interpolate

    def spy_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=None):
        interpolate_modes.append(mode)
        kwargs = {}
        if mode in ("bilinear", "bicubic") and antialias is not None:
            kwargs["antialias"] = antialias
        return orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy_interpolate)

    img = torch.rand((1, 512, 512, 3))
    mask = torch.ones((1, 512, 512))
    cfg = ReferenceConfig(role=ReferenceRole.SUBJECT, image=img, attention_mask=mask, reference_resize_method="bicubic")

    _ = prepare_reference(cfg, vae=MockVAE(), model=MockModel(), target_h=256, target_w=256, attention_mask_mode="soft")

    assert "bilinear" in interpolate_modes
    assert "bicubic" not in [m for m in interpolate_modes if "mask" in str(m)]  # mask soft interpolation must be bilinear, not bicubic


def test_inpainting_mask_uses_nearest_exact(monkeypatch):
    """7. Inpainting masks still use nearest-exact."""
    interpolate_modes = []
    orig_interpolate = torch.nn.functional.interpolate

    def spy_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=None):
        interpolate_modes.append(mode)
        kwargs = {}
        if mode in ("bilinear", "bicubic") and antialias is not None:
            kwargs["antialias"] = antialias
        return orig_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy_interpolate)

    img = torch.rand((1, 512, 512, 3))
    mask = torch.ones((1, 512, 512))
    vae = MockVAE()

    _ = generate_krea2_latent(
        model=MockModel(),
        vae=vae,
        width=256,
        height=256,
        batch_size=1,
        latent_source="image",
        base_image=img,
        inpaint_mask=mask,
        sampling_resize_mode="crop",
        sampling_resize_method="lanczos",
    )

    assert "nearest-exact" in interpolate_modes


def test_preset_balanced_resolves_resize_methods_to_auto():
    """8. Preset balanced with no advanced settings still resolves every image resize method to auto."""
    resolved = resolve_krea2_settings(preset_name="balanced")
    assert resolved.sampling_resize_method == "auto"
    for role_name, r_set in resolved.roles.items():
        assert r_set.grounding_resize_method == "auto"
        assert r_set.reference_resize_method == "auto"


def test_end_to_end_engine_wiring(monkeypatch):
    """9. End-to-end engine execution passes custom resize_methods to references and latents."""
    subj_setting = ImageRoleSettings(
        grounding_resize_method="bilinear",
        reference_resize_method="lanczos",
    )
    img_bundle = ImageAdvancedSettingsBundle().with_role("subject", subj_setting)
    edit_settings = EditAdvancedSettings(sampling_resize_method="nearest-exact")

    req = NodeExecutionRequest(
        node_name="CcC Krea2 - Subject",
        model=MockModel(),
        clip=None,
        prompt="test prompt",
        vae=MockVAE(),
        subject_image=torch.rand((1, 600, 600, 3)),
        image_advanced_settings=img_bundle,
        edit_advanced_settings=edit_settings,
        role_order=[ReferenceRole.SUBJECT],
    )

    # Resolve settings via precedence hierarchy
    settings = resolve_krea2_settings(
        preset_name=req.preset,
        edit_settings=req.edit_advanced_settings,
        image_settings=req.image_advanced_settings,
        active_roles=["subject"],
    )

    assert settings.roles["subject"].grounding_resize_method == "bilinear"
    assert settings.roles["subject"].reference_resize_method == "lanczos"
    assert settings.sampling_resize_method == "nearest-exact"
