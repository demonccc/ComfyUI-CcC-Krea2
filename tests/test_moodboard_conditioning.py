"""Tests for Moodboard conditioning, real Qwen vision row-span extraction, style fidelity, indirect style transfer, physical directives, and Krea2Edit geometry parity."""

import torch
import pytest
from ccc_krea2.conditioning import (
    extract_vision_spans_from_tokens,
    resolve_qwen_token_stream,
    encode_krea2_qwen_context
)
from ccc_krea2.style_processing import (
    apply_statistical_style_fidelity,
    StyleSpanOperation,
    slice_style_image
)
from ccc_krea2.reference_specs import (
    SubjectReferenceSpec,
    SceneReferenceSpec,
    StyleReferenceSpec,
    ReferenceChain,
    PreparedVisionImage,
    VisionPrepSpec
)
from ccc_krea2.reference_slots import resolve_reference_slots_and_aliases
from ccc_krea2.reference_directives import build_automatic_role_directive
from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry, process_image_and_mask_geometry


class DummyClip:
    def __init__(self, token_dict=None):
        self.token_dict = token_dict or {}

    def tokenize(self, prompt, images=None, llama_template=None):
        if self.token_dict:
            return self.token_dict
        tok_pairs = []
        if images:
            for img in images:
                tok_pairs.append([{"type": "image", "data": img}, None])
        else:
            tok_pairs.append([100, None])
        return {"qwen3vl": [tok_pairs]}

    def encode_from_tokens_scheduled(self, tokens):
        # 1 text prefix token + 256 image 1 rows + 64 image 2 rows = 321 rows
        seq_len = 321
        fused_dim = 1536  # divisible by 12 (1536 / 12 = 128)
        cond = torch.randn(1, seq_len, fused_dim)
        att_mask = torch.ones(1, seq_len, dtype=torch.float32)
        return [[cond, {"attention_mask": att_mask}]]


def make_dummy_prep_image(shape=(1, 512, 512, 3)):
    tensor = torch.zeros(shape, dtype=torch.float32)
    spec = VisionPrepSpec("native", 0.0, 1.0, 1.0, "auto", "auto", "Qwen3-VL", 32, {"min_pixels": 3136, "max_pixels": 12845056})
    return PreparedVisionImage(tensor, tensor, spec, {"src_hw": (shape[1], shape[2]), "prep_hw": (shape[1], shape[2]), "direction": "none", "resolved_method": "none", "config": None})


# --- 14.1 Real Span Mapping Tests ---

def test_real_span_mapping_with_text_prefix_and_bhwc_data():
    img1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)  # 512/16 = 32, 32*32/4 = 256 rows
    img2 = torch.zeros((1, 256, 256, 3), dtype=torch.float32)  # 256/16 = 16, 16*16/4 = 64 rows

    tok_pairs = [
        [100, None],  # Normal text token
        [{"type": "image", "data": img1}, None],
        [{"type": "image", "data": img2}, None]
    ]
    tokens = {"qwen3vl_4b": [tok_pairs]}

    pairs, key = resolve_qwen_token_stream(tokens)
    assert key == "qwen3vl_4b"
    assert pairs == tok_pairs

    phys_map = [
        {"image": img1, "role": "subject"},
        {"image": img2, "role": "scene"}
    ]

    spans, warnings, key, t_prefix = extract_vision_spans_from_tokens(tokens, phys_map)
    assert key == "qwen3vl_4b"
    assert len(spans) == 2
    # First span starts after 1 text token
    assert spans[0][0] == 1
    assert spans[0][1] == 1 + 256
    assert spans[1][0] == 1 + 256
    assert spans[1][1] == 1 + 256 + 64


def test_unsupported_token_keys_raise_clear_error():
    tokens = {"unsupported_key": "not_a_token_list"}
    phys_map = [{"role": "subject"}]
    with pytest.raises(ValueError) as excinfo:
        extract_vision_spans_from_tokens(tokens, phys_map)
    assert "Incompatible CLIP token structure" in str(excinfo.value)
    assert "unsupported_key" in str(excinfo.value)


def test_span_validation_fails_on_mismatched_image_count():
    img1 = torch.zeros((1, 256, 256, 3))
    tok_pairs = [[{"type": "image", "data": img1}, None]]
    tokens = {"qwen3vl": [tok_pairs]}
    phys_map = [{"role": "subject"}, {"role": "scene"}]  # Expected 2, got 1 span

    with pytest.raises(ValueError) as excinfo:
        extract_vision_spans_from_tokens(tokens, phys_map)
    assert "Vision span validation failed" in str(excinfo.value)


def test_encode_krea2_qwen_context_execution():
    clip = DummyClip()
    img1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    img2 = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    phys_images = [img1, img2]
    phys_map = [
        {"image": img1, "role": "subject", "physical_qwen_image_index": 1},
        {"image": img2, "role": "scene", "physical_qwen_image_index": 2}
    ]
    encoded = encode_krea2_qwen_context(
        clip=clip,
        prompt="a test prompt",
        physical_images=phys_images,
        physical_image_map=phys_map
    )
    assert encoded.pos_rows_before == 321
    assert len(encoded.vision_row_spans) == 2


# --- 14.2 Style Fidelity Tests ---

def test_style_fidelity_identity_and_statistical_target():
    cond = torch.randn(1, 10, 24)  # 24 is divisible by 12

    # Fidelity 1.0 (Exact Identity)
    op_identity = StyleSpanOperation(
        logical_reference_id="style", logical_vision_slot=1, physical_qwen_index=1,
        row_start=2, row_end=6, style_fidelity=1.0, indirect_style_transfer=False
    )
    res_id, _, _ = apply_statistical_style_fidelity(cond, [op_identity])
    assert torch.allclose(cond[:, 2:6], res_id[:, 2:6])

    # Fidelity 0.0 (Statistical Target)
    op_target = StyleSpanOperation(
        logical_reference_id="style", logical_vision_slot=1, physical_qwen_index=1,
        row_start=2, row_end=6, style_fidelity=0.0, indirect_style_transfer=False
    )
    res_tgt, _, _ = apply_statistical_style_fidelity(cond, [op_target])
    assert not torch.allclose(cond[:, 2:6], res_tgt[:, 2:6])
    # Unmodified rows remain identical
    assert torch.allclose(cond[:, 0:2], res_tgt[:, 0:2])
    assert torch.allclose(cond[:, 6:], res_tgt[:, 6:])


def test_slice_style_image_modes():
    img = torch.rand(1, 512, 512, 3)
    full_crops = slice_style_image(img, mode="full")
    assert len(full_crops) == 1
    crops_2x2 = slice_style_image(img, mode="2x2")
    assert len(crops_2x2) == 4
    crops_4x4 = slice_style_image(img, mode="4x4")
    assert len(crops_4x4) == 16


# --- 14.3 Indirect Style Transfer Tests ---

def test_indirect_style_transfer_single_operation_keep_mask():
    cond = torch.randn(1, 20, 24)

    op_dir = StyleSpanOperation(
        logical_reference_id="style1", logical_vision_slot=1, physical_qwen_index=1,
        row_start=2, row_end=5, style_fidelity=0.5, indirect_style_transfer=False
    )
    op_indir = StyleSpanOperation(
        logical_reference_id="style2", logical_vision_slot=2, physical_qwen_index=2,
        row_start=10, row_end=15, style_fidelity=0.0, indirect_style_transfer=True
    )

    res, indirect_applied, removed_indices = apply_statistical_style_fidelity(cond, [op_dir, op_indir])
    assert indirect_applied is True
    assert removed_indices == list(range(10, 15))
    assert res.shape[1] == 15  # 20 - 5 removed rows = 15


# --- 14.4 Physical Directive Mapping & Slot Ordering Tests ---

def test_style_before_non_style_is_rejected():
    prep = make_dummy_prep_image()
    s_style = StyleReferenceSpec(
        role="style",
        prepared_image=prep,
        requested_vision_slot=1,
        aliases_template="style_image",
        parsed_aliases=("style_image",),
        extra_vision_directive="",
        style_fidelity=0.5,
        style_processing="2x2",
        indirect_style_transfer=True,
        style_directive="modern"
    )
    s_subj = SubjectReferenceSpec("subject", prep, 2, "subject_image", ("subject_image",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, "auto")

    chain = ReferenceChain().append(s_style).append(s_subj)
    with pytest.raises(ValueError) as excinfo:
        resolve_reference_slots_and_aliases(chain)
    assert "Style references expand into multiple physical Qwen images" in str(excinfo.value)


def test_conflicting_literal_alias_raises_error():
    prep = make_dummy_prep_image()
    s_subj = SubjectReferenceSpec("subject", prep, 1, "Image 5", ("Image 5",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, "auto")

    chain = ReferenceChain().append(s_subj)
    with pytest.raises(ValueError) as excinfo:
        resolve_reference_slots_and_aliases(chain)
    assert "Conflicting literal positional alias 'Image 5': physical Qwen index is 1." in str(excinfo.value)


def test_physical_automatic_directives_for_expanded_style():
    prep = make_dummy_prep_image()
    s_scene = SceneReferenceSpec("scene", prep, 1, "scene_image", ("scene_image",), "", 1.0, 0.0, None, 1.0, 0.0, "auto")
    s_subj = SubjectReferenceSpec("subject", prep, 2, "subject_image", ("subject_image",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, "auto")
    s_style = StyleReferenceSpec(
        role="style",
        prepared_image=prep,
        requested_vision_slot=3,
        aliases_template="style_image",
        parsed_aliases=("style_image",),
        extra_vision_directive="",
        style_fidelity=0.5,
        style_processing="2x2",
        indirect_style_transfer=True,
        style_directive="vibrant"
    )

    chain = ReferenceChain().append(s_scene).append(s_subj).append(s_style)
    resolved_refs, _ = resolve_reference_slots_and_aliases(chain)

    dir_scene = build_automatic_role_directive(resolved_refs[0])
    dir_subj = build_automatic_role_directive(resolved_refs[1])
    dir_style = build_automatic_role_directive(resolved_refs[2])

    assert "Image 1" in dir_scene
    assert "Image 2" in dir_subj
    assert "Images 3 through 6" in dir_style


# --- 14.5 Krea2Edit Geometry Parity Tests ---

def test_krea2edit_geometry_parity_floor_vs_round():
    # External target: 490 x 245 (divisible by 8)
    geom = resolve_krea2edit_geometry(src_h=500, src_w=1000, tgt_h=245, tgt_w=490, fit_mode="fit")
    # Near-match branch uses exact target pixel dimensions for crop_and_resize
    assert geom.vae_input_pixel_size == (490, 245)
    # Source crop rectangle uses round() for exact aspect math
    left, top, crop_w, crop_h = geom.crop_rectangle
    assert crop_w > 0 and crop_h > 0


def test_process_image_and_mask_geometry_parity():
    img = torch.rand(1, 500, 1000, 3)
    mask = torch.rand(1, 500, 1000)
    geom = resolve_krea2edit_geometry(src_h=500, src_w=1000, tgt_h=245, tgt_w=490, fit_mode="fit")
    fit_img, fit_mask = process_image_and_mask_geometry(img, mask, geom)
    assert fit_img.shape[1] == geom.vae_input_pixel_size[1]
    assert fit_img.shape[2] == geom.vae_input_pixel_size[0]
    assert fit_mask.shape[1] == geom.vae_input_pixel_size[1]
    assert fit_mask.shape[2] == geom.vae_input_pixel_size[0]


def test_unvalidated_raw_list_rejected_when_test_helper_disabled():
    tok_pairs = [[100, None]]
    with pytest.raises(ValueError) as excinfo:
        resolve_qwen_token_stream(tok_pairs, allow_raw_list_test_helper=False)
    assert "Raw list token stream is permitted only as an internal test helper" in str(excinfo.value)


def test_style_fidelity_out_of_bounds_raises_error():
    cond = torch.randn(1, 10, 24)
    op_out_of_bounds = StyleSpanOperation(
        logical_reference_id="style", logical_vision_slot=1, physical_qwen_index=1,
        row_start=8, row_end=15, style_fidelity=0.5, indirect_style_transfer=False
    )
    with pytest.raises(ValueError) as excinfo:
        apply_statistical_style_fidelity(cond, [op_out_of_bounds])
    assert "Style span validation failed" in str(excinfo.value)


def test_production_qwen_processor_failure_raises_runtime_error(monkeypatch):
    """Assert production (test_mode=False) raises RuntimeError if process_qwen2vl_images is unavailable or fails."""
    import sys
    from unittest.mock import MagicMock
    from ccc_krea2.conditioning import calculate_qwen_rows_from_embedded_image

    img1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    elem = {"type": "image", "data": img1}

    # Simulate production environment where comfy module is present but process_qwen2vl_images fails
    mock_comfy = MagicMock()
    monkeypatch.setitem(sys.modules, "comfy", mock_comfy)

    with pytest.raises(RuntimeError) as excinfo:
        calculate_qwen_rows_from_embedded_image(elem, clip=None, test_mode=False)
    assert "Required Qwen processor" in str(excinfo.value) or "Qwen visual processor failed" in str(excinfo.value)


def test_production_qwen_processor_positive_path_with_image_grid_thw(monkeypatch):
    """Verify Qwen processor row calculation when image_grid_thw is produced by processor."""
    import sys
    from unittest.mock import MagicMock
    from ccc_krea2.conditioning import calculate_qwen_rows_from_embedded_image

    img1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    img2 = torch.ones((1, 256, 256, 3), dtype=torch.float32)
    elem1 = {"type": "image", "data": img1}
    elem2 = {"type": "image", "data": img2}

    fake_calls = []

    def fake_process_qwen2vl_images(image_data, min_pixels=None, max_pixels=None, patch_size=None):
        fake_calls.append({
            "image_data": image_data,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "patch_size": patch_size,
        })
        # Deliberately return fake grids inconsistent with native dimensions:
        # For 512x512 img1: native grid is 32x32=1024 -> 256 rows. Fake grid [1, 16, 16]=256 -> 64 rows.
        # For 256x256 img2: fake grid [1, 8, 8]=64 -> 16 rows.
        if len(fake_calls) == 1:
            grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)
        else:
            grid = torch.tensor([[1, 8, 8]], dtype=torch.int64)
        return None, grid

    mock_comfy = MagicMock()
    mock_qwen_vl = MagicMock()
    mock_qwen_vl.process_qwen2vl_images = fake_process_qwen2vl_images

    monkeypatch.setitem(sys.modules, "comfy", mock_comfy)
    monkeypatch.setitem(sys.modules, "comfy.text_encoders", MagicMock())
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.qwen_vl", mock_qwen_vl)

    dummy_clip = DummyClip()

    rows1 = calculate_qwen_rows_from_embedded_image(elem1, clip=dummy_clip, test_mode=False)
    rows2 = calculate_qwen_rows_from_embedded_image(elem2, clip=dummy_clip, test_mode=False)

    # 1. Assert fake was called once per embedded physical image
    assert len(fake_calls) == 2, "fake process_qwen2vl_images must be called once per embedded image"

    # 2. Captured call arguments & exact elem['data'] & BHWC tensor format & data integrity
    assert torch.equal(fake_calls[0]["image_data"], img1)
    assert fake_calls[0]["image_data"].shape == (1, 512, 512, 3)
    assert torch.equal(fake_calls[1]["image_data"], img2)
    assert fake_calls[1]["image_data"].shape == (1, 256, 256, 3)

    # 3. Span length follows fake grid & merge_size compression (2x2=4)
    # (16*16 // 4) = 64 rows (deliberately inconsistent with 512x512 native 256 rows)
    assert rows1 == 64, f"Expected 64 rows from fake grid [1, 16, 16], got {rows1}"
    # (8*8 // 4) = 16 rows
    assert rows2 == 16, f"Expected 16 rows from fake grid [1, 8, 8], got {rows2}"

    # 4. Validate span ordering with preceding text rows
    text_prefix_rows = 5
    span1_start = text_prefix_rows
    span1_end = span1_start + rows1
    span2_start = span1_end
    span2_end = span2_start + rows2

    assert span1_start > 0, "First visual span must not begin at zero when text precedes it"
    assert span1_end == 5 + 64 == 69
    assert span2_start == 69
    assert span2_end == 69 + 16 == 85
    assert span2_start > span1_start, "Multiple images must preserve physical order"


def test_integrated_qwen_token_to_span_mapping(monkeypatch):
    """Verify integrated token-to-span mapping calling extract_vision_spans_from_tokens."""
    import sys
    from unittest.mock import MagicMock
    from ccc_krea2.conditioning import extract_vision_spans_from_tokens

    token_img_1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    token_img_2 = torch.ones((1, 256, 256, 3), dtype=torch.float32)

    elem1 = {"type": "image", "data": token_img_1}
    elem2 = {"type": "image", "data": token_img_2}

    physical_sentinel_1 = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
    physical_sentinel_2 = torch.full((1, 64, 64, 3), 0.75, dtype=torch.float32)

    fake_calls = []

    def fake_process_qwen2vl_images(image_data, min_pixels=None, max_pixels=None, patch_size=None):
        fake_calls.append({
            "image_data": image_data,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "patch_size": patch_size,
        })
        if len(fake_calls) == 1:
            grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)
        else:
            grid = torch.tensor([[1, 8, 8]], dtype=torch.int64)
        return None, grid

    mock_comfy = MagicMock()
    mock_qwen_vl = MagicMock()
    mock_qwen_vl.process_qwen2vl_images = fake_process_qwen2vl_images

    monkeypatch.setitem(sys.modules, "comfy", mock_comfy)
    monkeypatch.setitem(sys.modules, "comfy.text_encoders", MagicMock())
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.qwen_vl", mock_qwen_vl)

    dummy_clip = DummyClip()

    IM_START = 151644
    USER = 872
    NEWLINE = 198

    token_pairs = [
        [IM_START, None],
        [USER, None],
        [NEWLINE, None],
        [101, None],
        [102, None],
        [103, None],
        [104, None],
        [105, None],
        [elem1, None],
        [201, None],
        [202, None],
        [203, None],
        [204, None],
        [205, None],
        [206, None],
        [207, None],
        [208, None],
        [209, None],
        [210, None],
        [elem2, None],
    ]

    tokens_dict = {"qwen3vl": [token_pairs]}
    physical_image_map = [
        {"role": "subject", "image": physical_sentinel_1},
        {"role": "scene", "image": physical_sentinel_2},
    ]

    spans, warnings, stream_key, prefix_removed = extract_vision_spans_from_tokens(
        tokens=tokens_dict,
        physical_image_map=physical_image_map,
        clip=dummy_clip,
        allow_raw_list_test_helper=False,
        test_mode=False,
    )

    # 1. Assert processor called once per embedded image in token order by object identity
    assert len(fake_calls) == 2
    assert fake_calls[0]["image_data"] is token_img_1
    assert fake_calls[1]["image_data"] is token_img_2
    assert fake_calls[0]["image_data"] is not physical_sentinel_1
    assert fake_calls[0]["image_data"] is not physical_sentinel_2
    assert fake_calls[1]["image_data"] is not physical_sentinel_1
    assert fake_calls[1]["image_data"] is not physical_sentinel_2
    assert fake_calls[0]["image_data"].shape == (1, 512, 512, 3)
    assert fake_calls[1]["image_data"].shape == (1, 256, 256, 3)

    # 2. Assert stream_key and template prefix stripping
    assert stream_key == "qwen3vl"
    assert prefix_removed == 3

    # 3. Assert span count
    assert len(spans) == 2

    s1, e1 = spans[0]
    s2, e2 = spans[1]

    # 4. Assert half-open spans, prefix stripping reflection, text separation, and physical order
    assert s1 == 5
    assert e1 == 5 + 64 == 69
    assert e1 > s1

    assert s2 == 69 + 10 == 79
    assert e2 == 79 + 16 == 95
    assert e2 > s2

    # No overlap and physical order
    assert s2 >= e1


def test_attach_reference_latents_appends_to_existing():
    from ccc_krea2.conditioning import attach_reference_latents_to_conditioning

    existing_ref = torch.rand(1, 16, 64, 64)
    new_ref = torch.rand(1, 16, 64, 64)

    cond = [("tensor", {"reference_latents": [existing_ref]})]
    updated = attach_reference_latents_to_conditioning(cond, [new_ref])

    attached_refs = updated[0][1]["reference_latents"]
    assert len(attached_refs) == 2
    assert attached_refs[0] is existing_ref
    assert attached_refs[1] is new_ref

