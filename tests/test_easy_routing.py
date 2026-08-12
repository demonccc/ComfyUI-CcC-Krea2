"""Table-driven exhaustive test suite for Easy Edit 3-Phase Routing Engine."""

import pytest
from ccc_krea2.easy_routing import (
    resolve_easy_sources,
    route_easy_preset,
    BALANCED_SUBJECT_BOOST,
    MAX_IDENTITY_SUBJECT_BOOST,
    PRESERVE_IDENTITY_SUBJECT_BOOST,
    PRESERVE_SCENE_BOOST,
    OUTFIT_EMPHASIS_BOOST,
    OUTFIT_TRANSFER_BOOST,
    NORMAL_BOOST,
    EASY_SCENE_AND_OUTFIT_INSTRUCTION,
)


class DummyImg:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"<DummyImg:{self.name}>"


@pytest.fixture
def dummy_sources():
    S = DummyImg("Subject")
    Sc = DummyImg("Scene")
    Ou = DummyImg("Outfit")
    St = DummyImg("Style")
    return S, Sc, Ou, St


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def assert_refs(refs, expected):
    """Assert edit_references matches list of (img, boost, alias) — ignoring instruction."""
    assert len(refs) == len(expected), f"Expected {len(expected)} refs, got {len(refs)}: {refs}"
    for (img, boost, alias, _instr), (exp_img, exp_boost, exp_alias) in zip(refs, expected):
        assert img is exp_img, f"Wrong image for alias {alias}"
        assert boost == pytest.approx(exp_boost, rel=1e-6), f"Wrong boost for alias {alias}: {boost} != {exp_boost}"
        assert alias == exp_alias, f"Wrong alias: {alias} != {exp_alias}"


def assert_no_more_than_2_refs(route):
    assert len(route.edit_references) <= 2, f"Too many appearance refs: {route.edit_references}"


# ---------------------------------------------------------------------------
# BALANCED + STYLE_TRANSFER: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["balanced", "style_transfer"])
class TestBalancedMatrix:
    def test_none(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Ou
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, BALANCED_SUBJECT_BOOST, "subject")])

    def test_subject_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

    def test_scene_outfit(self, dummy_sources, preset):
        """Scene + Outfit without Subject: target=scene, geometry=scene, refs=scene+outfit."""
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_geometry_source is Sc
        assert len(route.edit_references) == 2
        # Both scene and outfit appear as distinct refs
        aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "scene" in aliases
        assert "outfit" in aliases
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])


# ---------------------------------------------------------------------------
# Combined Scene+Outfit ref (scene image used as outfit)
# ---------------------------------------------------------------------------

class TestCombinedSceneOutfitRef:
    def test_outfit_from_scene_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="balanced")
        # effective_outfit is Sc (same object as Sc)
        # Scene + Outfit (same): should produce combined ref
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert len(route.edit_references) == 1
        _, _, alias, instruction = route.edit_references[0]
        assert alias == "scene+outfit"
        assert "scene composition" in instruction
        assert "clothing" in instruction

    def test_outfit_from_scene_with_subject(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="balanced")
        # S + Sc + same-outfit → S, combined_scene_outfit
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        # Subject + combined scene+outfit
        assert len(route.edit_references) == 2
        aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "subject" in aliases
        assert "scene+outfit" in aliases

    def test_combined_ref_instruction_present(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert len(route.edit_references) >= 1
        combined = [(alias, instr) for _, _, alias, instr in route.edit_references if alias == "scene+outfit"]
        assert len(combined) == 1
        assert combined[0][1] == EASY_SCENE_AND_OUTFIT_INSTRUCTION

    def test_outfit_from_style_distinct(self, dummy_sources):
        """Outfit from style: style image is distinct — creates two separate appearance refs."""
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, style=St, outfit_source="style image")
        route = route_easy_preset(sources, preset="balanced")
        # effective_outfit = St (distinct from S), no scene
        # → S + St (as outfit)
        assert len(route.edit_references) == 2
        aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "subject" in aliases
        assert "outfit" in aliases


# ---------------------------------------------------------------------------
# PRESERVE_IDENTITY: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------

class TestPreserveIdentityMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="preserve_identity")
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert_refs(route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="preserve_identity")
        # warning issued, balanced fallback
        assert any("missing" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert any("missing" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")])

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert_refs(route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

    def test_scene_outfit_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        # no subject warning, balanced fallback: Sc+Ou
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert_refs(route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"), (Ou, OUTFIT_EMPHASIS_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, OUTFIT_EMPHASIS_BOOST, "scene+outfit"), (S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")])


# ---------------------------------------------------------------------------
# MAX_IDENTITY: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------

class TestMaxIdentityMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="max_identity")
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert_refs(route.edit_references, [(S, MAX_IDENTITY_SUBJECT_BOOST, "subject")])

    def test_scene_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="max_identity")
        assert any("missing" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="max_identity")
        assert any("missing" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, MAX_IDENTITY_SUBJECT_BOOST, "subject")])

    def test_scene_outfit_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="max_identity")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit"), (S, MAX_IDENTITY_SUBJECT_BOOST, "subject")])

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(Sc, OUTFIT_EMPHASIS_BOOST, "scene+outfit"), (S, MAX_IDENTITY_SUBJECT_BOOST, "subject")])


# ---------------------------------------------------------------------------
# PRESERVE_SCENE: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------

class TestPreserveSceneMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)
        assert len(route.edit_references) == 0

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene"), (S, NORMAL_BOOST, "subject")])

    def test_subject_scene_outfit_semantic(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        # outfit becomes semantic-only when S+Sc fill appearance slots
        assert len(route.edit_references) == 2
        assert len(route.semantic_only_references) == 1
        sem_aliases = [alias for _, alias in route.semantic_only_references]
        assert "outfit" in sem_aliases

    def test_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_no_more_than_2_refs(route)


# ---------------------------------------------------------------------------
# OUTFIT_TRANSFER: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------

class TestOutfitTransferMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert any("missing" in w for w in route.warnings)

    def test_outfit_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Ou
        assert_refs(route.edit_references, [(Ou, OUTFIT_TRANSFER_BOOST, "outfit")])

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert_refs(route.edit_references, [(S, NORMAL_BOOST, "subject"), (Ou, OUTFIT_TRANSFER_BOOST, "outfit")])

    def test_scene_outfit(self, dummy_sources):
        """Scene + Outfit without Subject: target=scene, refs=scene+outfit."""
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert len(route.edit_references) == 2
        aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "scene" in aliases or "scene+outfit" in aliases
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert_refs(route.edit_references, [(S, NORMAL_BOOST, "subject"), (Ou, OUTFIT_TRANSFER_BOOST, "outfit")])

    def test_scene_outfit_combined(self, dummy_sources):
        """outfit_source=scene image: Sc+Sc (combined)."""
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 1
        _, _, alias, instr = route.edit_references[0]
        assert alias == "scene+outfit"
        assert instr == EASY_SCENE_AND_OUTFIT_INSTRUCTION


# ---------------------------------------------------------------------------
# Style configs
# ---------------------------------------------------------------------------

def test_style_active_for_all_presets(dummy_sources):
    S, _, _, St = dummy_sources
    sources = resolve_easy_sources(subject=S, style=St)

    for p in ("balanced", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer"):
        r = route_easy_preset(sources, preset=p)
        assert r.style_active is True
        assert r.style_source is St
        assert r.style_config.style_fidelity == 1.0
        assert r.style_config.style_processing == "2x2"
        assert r.style_config.indirect_style_transfer is False
        assert r.style_config.vision_instruction == ""

    r_st = route_easy_preset(sources, preset="style_transfer")
    assert r_st.style_active is True
    assert r_st.style_source is St
    assert r_st.style_config.style_fidelity == 1.0
    assert r_st.style_config.style_processing == "4x4"
    assert r_st.style_config.indirect_style_transfer is False
    assert "artistic style" in r_st.style_config.vision_instruction


# ---------------------------------------------------------------------------
# Selector tests
# ---------------------------------------------------------------------------

class TestSelectors:
    def test_outfit_from_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        assert sources.effective_outfit is Sc
        assert sources.outfit_source_kind == "scene"
        assert len(sources.warnings) == 0

    def test_outfit_from_style(self, dummy_sources):
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, style=St, outfit_source="style image")
        assert sources.effective_outfit is St
        assert sources.outfit_source_kind == "style"
        assert len(sources.warnings) == 0

    def test_style_from_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, style_source="scene image")
        assert sources.effective_style is Sc
        assert sources.style_source_kind == "scene"
        assert len(sources.warnings) == 0

    def test_style_from_subject(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, style_source="subject image")
        assert sources.effective_style is S
        assert sources.style_source_kind == "subject"
        assert len(sources.warnings) == 0

    def test_disconnected_outfit_source_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit_source="scene image")
        assert sources.effective_outfit is None
        assert len(sources.warnings) == 1
        assert "disconnected" in sources.warnings[0]

    def test_disconnected_style_source_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, style_source="scene image")
        assert sources.effective_style is None
        assert len(sources.warnings) == 1
        assert "disconnected" in sources.warnings[0]

    def test_outfit_from_style_disconnected(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit_source="style image")
        assert sources.effective_outfit is None
        assert len(sources.warnings) == 1
        assert "disconnected" in sources.warnings[0]

    def test_style_from_subject_disconnected(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(style_source="subject image")
        assert sources.effective_style is None
        assert len(sources.warnings) == 1


# ---------------------------------------------------------------------------
# Cross-route: same image for multiple logical functions
# ---------------------------------------------------------------------------

class TestCrossRouting:
    def test_outfit_from_style_and_style_active(self, dummy_sources):
        """Same style image used as outfit AND style moodboard simultaneously."""
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, style=St, outfit_source="style image")
        route = route_easy_preset(sources, preset="style_transfer")
        # effective_outfit == St (distinct from S), effective_style == St
        # Style is a separate moodboard path, so outfit+style can both be St
        assert route.style_active is True
        assert route.style_source is St
        app_aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "subject" in app_aliases
        # outfit alias should be in edit_references (appearance ref)
        assert "outfit" in app_aliases

    def test_style_from_scene_with_distinct_style(self, dummy_sources):
        """style_source=scene image: effective_style=Sc, but Sc also used as scene appearance ref."""
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, style_source="scene image")
        route = route_easy_preset(sources, preset="balanced")
        # Scene as style — the style path is separate Moodboard
        assert route.style_active is True
        assert route.style_source is Sc
        # Sc should still be an appearance ref in balanced
        app_images = [img for img, _, _, _ in route.edit_references]
        assert Sc in app_images

    def test_semantic_only_references_routing(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        # 3 sources in balanced: S and Ou fill appearance slots, Sc→target_content
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="balanced")
        assert route.target_content_source is Sc
        assert len(route.edit_references) == 2
        assert (S, BALANCED_SUBJECT_BOOST, "subject") in [(img, boost, alias) for img, boost, alias, _ in route.edit_references]
        assert (Ou, NORMAL_BOOST, "outfit") in [(img, boost, alias) for img, boost, alias, _ in route.edit_references]
        assert isinstance(route.semantic_only_references, tuple)


# ---------------------------------------------------------------------------
# Never more than 2 appearance refs — all presets, several combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["balanced", "style_transfer", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer"])
def test_never_more_than_2_refs(dummy_sources, preset):
    S, Sc, Ou, St = dummy_sources
    for args in [
        {},
        {"subject": S},
        {"scene": Sc},
        {"outfit": Ou},
        {"subject": S, "scene": Sc},
        {"subject": S, "outfit": Ou},
        {"scene": Sc, "outfit": Ou},
        {"subject": S, "scene": Sc, "outfit": Ou},
    ]:
        sources = resolve_easy_sources(**args)
        route = route_easy_preset(sources, preset=preset)
        assert len(route.edit_references) <= 2, (
            f"Preset '{preset}' with args {args} produced {len(route.edit_references)} refs"
        )
