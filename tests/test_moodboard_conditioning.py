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
    def __init__(self, token_dict):
        self.token_dict = token_dict

    def tokenize(self, prompt, images=None, llama_template=None):
        return self.token_dict

    def encode_from_tokens_scheduled(self, tokens):
        # 1 text prefix token + 64 image 1 rows + 64 image 2 rows = 129 rows
        seq_len = 129
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
        style_directive=True
    )
    s_subj = SubjectReferenceSpec("subject", prep, 2, "subject_image", ("subject_image",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, 0.0, "auto")

    chain = ReferenceChain().append(s_style).append(s_subj)
    with pytest.raises(ValueError) as excinfo:
        resolve_reference_slots_and_aliases(chain)
    assert "Style references expand into multiple physical Qwen images" in str(excinfo.value)


def test_conflicting_literal_alias_raises_error():
    prep = make_dummy_prep_image()
    s_subj = SubjectReferenceSpec("subject", prep, 1, "Image 5", ("Image 5",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, 0.0, "auto")

    chain = ReferenceChain().append(s_subj)
    with pytest.raises(ValueError) as excinfo:
        resolve_reference_slots_and_aliases(chain)
    assert "Conflicting literal positional alias 'Image 5': physical Qwen index is 1." in str(excinfo.value)


def test_physical_automatic_directives_for_expanded_style():
    prep = make_dummy_prep_image()
    s_scene = SceneReferenceSpec("scene", prep, 1, "scene_image", ("scene_image",), "", 1.0, 0.0, None, 1.0, 0.0, "auto")
    s_subj = SubjectReferenceSpec("subject", prep, 2, "subject_image", ("subject_image",), "", 1.0, 0.0, 0.0, None, 1.0, 0.0, 0.0, "auto")
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
        style_directive=True
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
    # External target: 490 x 245 (divisible by 8: 488x240, not 16)
    geom = resolve_krea2edit_geometry(src_h=500, src_w=1000, tgt_h=245, tgt_w=490, fit_mode="fit")
    # Upstream calculation uses floor // 16 for fitted VAE input dimensions
    assert geom.vae_input_pixel_size[0] % 16 == 0
    assert geom.vae_input_pixel_size[1] % 16 == 0
    # Source crop rectangle uses round() for exact aspect math
    left, top, crop_w, crop_h = geom.crop_rectangle
    assert crop_w > 0 and crop_h > 0
