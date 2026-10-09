"""Unit tests for prompt augmentation value objects and helper functions."""

from ccc_krea2.prompt_augmentation import (
    PromptAugmentation,
    EMPTY_PROMPT_AUGMENTATION,
    apply_prompt_augmentation,
)


def test_empty_augmentation_returns_original_prompts():
    pos = "a subject standing in a room"
    neg = "low quality, blurry"
    eff_pos, eff_neg = apply_prompt_augmentation(pos, neg, EMPTY_PROMPT_AUGMENTATION)
    assert eff_pos == pos
    assert eff_neg == neg


def test_positive_prepend_and_append_order_and_separator():
    aug = PromptAugmentation(
        positive_prepend=("masterpiece", "best quality"),
        positive_append=("cyberpunk style", "8k resolution"),
    )
    pos = "a woman sitting on a bench"
    neg = ""

    eff_pos, _ = apply_prompt_augmentation(pos, neg, aug)
    expected = "masterpiece\n\nbest_quality\n\na woman sitting on a bench\n\ncyberpunk style\n\n8k resolution".replace(
        "best_quality", "best quality"
    )
    assert eff_pos == expected


def test_negative_prepend_and_append_order_and_separator():
    aug = PromptAugmentation(
        negative_prepend=("deformed", "bad anatomy"),
        negative_append=("text, watermark", "worst quality"),
    )
    pos = ""
    neg = "disfigured"

    _, eff_neg = apply_prompt_augmentation(pos, neg, aug)
    expected = "deformed\n\nbad anatomy\n\ndisfigured\n\ntext, watermark\n\nworst quality"
    assert eff_neg == expected


def test_blank_augmentation_entries_are_ignored():
    aug = PromptAugmentation(
        positive_prepend=("", "  ", "valid prepend"),
        positive_append=("valid append", ""),
        negative_prepend=("", "valid neg prepend"),
    )
    pos = "base prompt"
    neg = "base negative"

    eff_pos, eff_neg = apply_prompt_augmentation(pos, neg, aug)
    assert eff_pos == "valid prepend\n\nbase prompt\n\nvalid append"
    assert eff_neg == "valid neg prepend\n\nbase negative"


def test_original_prompt_strings_remain_unchanged():
    orig_pos = "original positive string"
    orig_neg = "original negative string"
    aug = PromptAugmentation(
        positive_prepend=("prefix",),
        positive_append=("suffix",),
    )

    eff_pos, eff_neg = apply_prompt_augmentation(orig_pos, orig_neg, aug)
    assert orig_pos == "original positive string"
    assert orig_neg == "original negative string"
    assert eff_pos != orig_pos


def test_tuple_ordering_preserved_across_stacks():
    stack1_aug = PromptAugmentation(
        positive_prepend=("stack1_pre",),
        positive_append=("stack1_app",),
    )

    stack2_aug = PromptAugmentation(
        positive_prepend=stack1_aug.positive_prepend + ("stack2_pre",),
        positive_append=stack1_aug.positive_append + ("stack2_app",),
    )

    eff_pos, _ = apply_prompt_augmentation("base", "", stack2_aug)
    assert eff_pos == "stack1_pre\n\nstack2_pre\n\nbase\n\nstack1_app\n\nstack2_app"


def test_qwen_base_prompt_treated_like_any_base_prompt():
    qwen_prompt = "Analyze the scene image and replace person with subject."
    aug = PromptAugmentation(
        positive_prepend=("detailed portrait",),
        positive_append=("photorealistic",),
    )
    eff_pos, _ = apply_prompt_augmentation(qwen_prompt, "", aug)
    assert eff_pos == "detailed portrait\n\nAnalyze the scene image and replace person with subject.\n\nphotorealistic"
