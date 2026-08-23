"""Table-driven exhaustive test suite for Easy Edit 3-Phase Routing Engine."""

import pytest
from ccc_krea2.easy_routing import (
    resolve_easy_sources,
    route_easy_preset,
    FLEXIBLE_SUBJECT_BOOST,
    BALANCED_SUBJECT_BOOST,
    CONSISTENT_SUBJECT_BOOST,
    PRESERVE_IDENTITY_SUBJECT_BOOST,
    MAX_IDENTITY_SUBJECT_BOOST,
    IDENTITY_TRANSFER_SCENE_BOOST,
    IDENTITY_TRANSFER_SUBJECT_BOOST,
    SUBJECT_TRANSFER_SCENE_BOOST,
    SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST,
    SUBJECT_TRANSFER_SUBJECT_BOOST,
    SUBJECT_TRANSFER_OUTFIT_BOOST,
    PRESERVE_SCENE_BOOST,
    OUTFIT_EMPHASIS_BOOST,
    OUTFIT_TRANSFER_SUBJECT_BOOST,
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


def assert_bounded_appearance_refs(route):
    limit = 3 if route.preset == "subject_transfer" else 2
    assert len(route.edit_references) <= limit, (
        f"Too many appearance refs for preset {route.preset}: {route.edit_references}"
    )


def assert_no_more_than_2_refs(route):
    assert_bounded_appearance_refs(route)


# ---------------------------------------------------------------------------
# FLEXIBLE: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------


class TestFlexibleMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="flexible")
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="flexible")
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, FLEXIBLE_SUBJECT_BOOST, "subject")])
        assert route.edit_references[0][1] == pytest.approx(1.0)
        assert_no_more_than_2_refs(route)

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="flexible")
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="flexible")
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, FLEXIBLE_SUBJECT_BOOST, "subject")])
        for _, b, _, _ in route.edit_references:
            assert b == pytest.approx(1.0)

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        assert_refs(route.edit_references, [(S, FLEXIBLE_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])
        for _, b, _, _ in route.edit_references:
            assert b == pytest.approx(1.0)

    def test_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(S, FLEXIBLE_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])
        for _, b, _, _ in route.edit_references:
            assert b == pytest.approx(1.0)
        assert_no_more_than_2_refs(route)


# ---------------------------------------------------------------------------
# BALANCED + STYLE_TRANSFER: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["balanced"])
class TestBalancedMatrix:
    def test_none(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset)
        route = route_easy_preset(sources, preset=preset)
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])
        assert_no_more_than_2_refs(route)

    def test_outfit_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Ou
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S, scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, BALANCED_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_subject_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_scene_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_scene_as_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene+outfit"
        assert route.target_geometry_source is Sc
        assert len(route.edit_references) == 1
        img, boost, alias, instr = route.edit_references[0]
        assert img is Sc
        assert alias == "scene+outfit"
        assert instr == EASY_SCENE_AND_OUTFIT_INSTRUCTION

    def test_subject_scene_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])
        assert len(route.edit_references) == 2
        assert_no_more_than_2_refs(route)

    def test_subject_scene_as_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene+outfit"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Sc, NORMAL_BOOST, "scene+outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_outfit_from_style_distinct(self, dummy_sources, preset):
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(preset=preset, subject=S, style=St, outfit_source="style image")
        route = route_easy_preset(sources, preset=preset)
        assert len(route.edit_references) == 2
        aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "subject" in aliases
        assert "outfit" in aliases


# ---------------------------------------------------------------------------
# CONSISTENT: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------


class TestConsistentMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="consistent")
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="consistent")
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, CONSISTENT_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="consistent")
        assert any("preset 'consistent'" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="consistent")
        assert any("preset 'consistent'" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="consistent")
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, CONSISTENT_SUBJECT_BOOST, "subject")])

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="consistent")
        assert_refs(route.edit_references, [(S, CONSISTENT_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

    def test_scene_outfit_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="consistent")
        assert any("preset 'consistent'" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="consistent")
        assert route.target_content_mode == "image"
        assert_refs(
            route.edit_references, [(S, CONSISTENT_SUBJECT_BOOST, "subject"), (Ou, OUTFIT_EMPHASIS_BOOST, "outfit")]
        )
        assert_no_more_than_2_refs(route)

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="consistent")
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [(Sc, OUTFIT_EMPHASIS_BOOST, "scene+outfit"), (S, CONSISTENT_SUBJECT_BOOST, "subject")],
        )


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
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert any("preset 'preserve_identity'" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert any("preset 'preserve_identity'" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")]
        )

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")]
        )

    def test_scene_outfit_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert any("preset 'preserve_identity'" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references, [(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")]
        )
        assert_no_more_than_2_refs(route)

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references, [(Sc, NORMAL_BOOST, "scene+outfit"), (S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")]
        )


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
        assert any("preset 'max_identity'" in w for w in route.warnings)
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])

    def test_outfit_only_fallback(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="max_identity")
        assert any("preset 'max_identity'" in w for w in route.warnings)
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
        assert any("preset 'max_identity'" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert len(route.edit_references) == 2
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_content_role == "subject"
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references, [(Sc, NORMAL_BOOST, "scene+outfit"), (S, MAX_IDENTITY_SUBJECT_BOOST, "subject")]
        )
        sc_ref = next(r for r in route.edit_references if r[2] == "scene+outfit")
        assert sc_ref[3] == EASY_SCENE_AND_OUTFIT_INSTRUCTION
        for r in route.edit_references:
            if r[2] != "subject":
                assert r[1] <= NORMAL_BOOST


# ---------------------------------------------------------------------------
# Subject-only runtime contracts: Flexible, Balanced, Consistent, Preserve Identity, Max Identity
# ---------------------------------------------------------------------------


class TestSubjectOnlyPresetContracts:
    def test_flexible_subject_only_contract(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="flexible")
        assert route.preset == "flexible"
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert len(route.edit_references) == 1
        img, boost, alias, _ = route.edit_references[0]
        assert img is S
        assert alias == "subject"
        assert boost == pytest.approx(1.0)

    def test_balanced_subject_only_contract(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="balanced")
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert len(route.edit_references) == 1
        img, boost, alias, _ = route.edit_references[0]
        assert img is S
        assert alias == "subject"
        assert boost == pytest.approx(2.5)

    def test_consistent_subject_only_contract(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="consistent")
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert len(route.edit_references) == 1
        img, boost, alias, _ = route.edit_references[0]
        assert img is S
        assert alias == "subject"
        assert boost == pytest.approx(4.0)

    def test_preserve_identity_subject_only_contract(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="preserve_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert len(route.edit_references) == 1
        img, boost, alias, _ = route.edit_references[0]
        assert img is S
        assert alias == "subject"
        assert boost == pytest.approx(6.0)

    def test_max_identity_subject_only_contract(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="max_identity")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert len(route.edit_references) == 1
        img, boost, alias, _ = route.edit_references[0]
        assert img is S
        assert alias == "subject"
        assert boost == pytest.approx(10.0)

    def test_subject_only_presets_distinctness(self, dummy_sources):
        S, _, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        flexible_route = route_easy_preset(sources, preset="flexible")
        balanced_route = route_easy_preset(sources, preset="balanced")
        consistent_route = route_easy_preset(sources, preset="consistent")
        preserve_route = route_easy_preset(sources, preset="preserve_identity")
        max_route = route_easy_preset(sources, preset="max_identity")

        # Flexible != Balanced due to boost (1.0 != 2.5)
        assert flexible_route.edit_references[0][1] == pytest.approx(1.0)
        assert balanced_route.edit_references[0][1] == pytest.approx(2.5)
        assert flexible_route.edit_references[0][1] != balanced_route.edit_references[0][1]

        # Balanced != Consistent due to boost (2.5 != 4.0)
        assert consistent_route.edit_references[0][1] == pytest.approx(4.0)
        assert balanced_route.edit_references[0][1] != consistent_route.edit_references[0][1]

        # Consistent != Preserve Identity due to target_content_mode ("empty" != "image") and boost (4.0 != 6.0)
        assert consistent_route.target_content_mode == "empty"
        assert preserve_route.target_content_mode == "image"
        assert consistent_route.target_content_mode != preserve_route.target_content_mode
        assert consistent_route.edit_references[0][1] == pytest.approx(4.0)
        assert preserve_route.edit_references[0][1] == pytest.approx(6.0)

        # Preserve Identity != Max Identity due to boost (6.0 != 10.0)
        assert preserve_route.edit_references[0][1] == pytest.approx(6.0)
        assert max_route.edit_references[0][1] == pytest.approx(10.0)
        assert preserve_route.edit_references[0][1] != max_route.edit_references[0][1]


# ---------------------------------------------------------------------------
# Flexible All-Reference Attention Boost Neutrality (1.0)
# ---------------------------------------------------------------------------


class TestFlexibleAllReferenceAttention:
    def test_flexible_subject_scene(self, dummy_sources):
        S, Sc, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="flexible")
        for img, boost, alias, _ in route.edit_references:
            assert boost == pytest.approx(1.0), f"Reference '{alias}' boost is {boost}, expected 1.0"

    def test_flexible_subject_outfit(self, dummy_sources):
        S, _, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        for img, boost, alias, _ in route.edit_references:
            assert boost == pytest.approx(1.0), f"Reference '{alias}' boost is {boost}, expected 1.0"

    def test_flexible_subject_scene_distinct_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="flexible")
        for img, boost, alias, _ in route.edit_references:
            assert boost == pytest.approx(1.0), f"Reference '{alias}' boost is {boost}, expected 1.0"

    def test_flexible_scene_reused_as_outfit(self, dummy_sources):
        S, Sc, _, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="flexible")
        for img, boost, alias, _ in route.edit_references:
            assert boost == pytest.approx(1.0), f"Reference '{alias}' boost is {boost}, expected 1.0"


# ---------------------------------------------------------------------------
# Preserve Identity vs Max Identity Topology Parity
# ---------------------------------------------------------------------------


class TestPreserveIdentityVsMaxIdentityTopology:
    @pytest.mark.parametrize(
        "src_kwargs",
        [
            {},
            {"subject": "S"},
            {"scene": "Sc"},
            {"outfit": "Ou"},
            {"subject": "S", "scene": "Sc"},
            {"subject": "S", "outfit": "Ou"},
            {"scene": "Sc", "outfit": "Ou"},
            {"subject": "S", "scene": "Sc", "outfit": "Ou"},
            {"subject": "S", "scene": "Sc", "outfit_source": "scene image"},
        ],
    )
    def test_topology_parity(self, dummy_sources, src_kwargs):
        S, Sc, Ou, St = dummy_sources
        resolved_kwargs = {}
        for k, v in src_kwargs.items():
            if v == "S":
                resolved_kwargs[k] = S
            elif v == "Sc":
                resolved_kwargs[k] = Sc
            elif v == "Ou":
                resolved_kwargs[k] = Ou
            else:
                resolved_kwargs[k] = v

        sources = resolve_easy_sources(**resolved_kwargs)
        preserve_route = route_easy_preset(sources, preset="preserve_identity")
        max_route = route_easy_preset(sources, preset="max_identity")

        # Topology parity checks
        assert preserve_route.target_content_mode == max_route.target_content_mode
        assert preserve_route.target_content_source is max_route.target_content_source
        assert preserve_route.target_geometry_mode == max_route.target_geometry_mode
        assert preserve_route.target_geometry_source is max_route.target_geometry_source

        # Alias role order checks
        p_aliases = [alias for _, _, alias, _ in preserve_route.edit_references]
        m_aliases = [alias for _, _, alias, _ in max_route.edit_references]
        assert p_aliases == m_aliases

        p_sem_aliases = [alias for _, alias in preserve_route.semantic_only_references]
        m_sem_aliases = [alias for _, alias in max_route.semantic_only_references]
        assert p_sem_aliases == m_sem_aliases

        # Boost difference checks
        for (p_img, p_boost, alias, _), (m_img, m_boost, _, _) in zip(
            preserve_route.edit_references, max_route.edit_references
        ):
            if alias == "subject":
                assert p_boost == pytest.approx(6.0)
                assert m_boost == pytest.approx(10.0)
            else:
                assert p_boost == pytest.approx(m_boost)


# ---------------------------------------------------------------------------
# Consistent Regression (verifying old preserve_identity weaker routing)
# ---------------------------------------------------------------------------


class TestConsistentRegression:
    def test_consistent_retains_weak_preserve_identity_topology(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources

        # Subject only -> empty target content
        sources_s = resolve_easy_sources(subject=S)
        r_s = route_easy_preset(sources_s, preset="consistent")
        assert r_s.target_content_mode == "empty"
        assert r_s.edit_references[0][1] == pytest.approx(4.0)

        # Subject + Scene -> empty target content
        sources_ssc = resolve_easy_sources(subject=S, scene=Sc)
        r_ssc = route_easy_preset(sources_ssc, preset="consistent")
        assert r_ssc.target_content_mode == "empty"
        assert_refs(r_ssc.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, CONSISTENT_SUBJECT_BOOST, "subject")])

        # Subject + Outfit -> empty target content
        sources_sou = resolve_easy_sources(subject=S, outfit=Ou)
        r_sou = route_easy_preset(sources_sou, preset="consistent")
        assert r_sou.target_content_mode == "empty"
        assert_refs(r_sou.edit_references, [(S, CONSISTENT_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

        # Subject + Scene + Outfit -> target content image (Scene)
        sources_all = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        r_all = route_easy_preset(sources_all, preset="consistent")
        assert r_all.target_content_mode == "image"
        assert r_all.target_content_source is Sc
        assert_refs(
            r_all.edit_references, [(S, CONSISTENT_SUBJECT_BOOST, "subject"), (Ou, OUTFIT_EMPHASIS_BOOST, "outfit")]
        )


# ---------------------------------------------------------------------------
# Missing Subject Fallbacks
# ---------------------------------------------------------------------------


class TestMissingSubjectFallbacks:
    @pytest.mark.parametrize("preset", ["consistent", "preserve_identity", "max_identity"])
    def test_missing_subject_warnings(self, dummy_sources, preset):
        _, Sc, _, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert len(route.warnings) == 1
        assert f"preset '{preset}'" in route.warnings[0]
        assert "Subject source is missing" in route.warnings[0]
        assert route.target_content_mode == "empty"
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])


# ---------------------------------------------------------------------------
# PRESERVE_SCENE: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------


class TestPreserveSceneMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"
        assert len(route.edit_references) == 0

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene")])

    def test_outfit_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene"), (S, NORMAL_BOOST, "subject")])

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"
        assert_refs(route.edit_references, [(S, NORMAL_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])

    def test_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene"), (S, NORMAL_BOOST, "subject")])
        assert len(route.semantic_only_references) == 1
        assert route.semantic_only_references[0] == (Ou, "outfit")
        assert_no_more_than_2_refs(route)

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="preserve_scene")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, PRESERVE_SCENE_BOOST, "scene+outfit"), (S, NORMAL_BOOST, "subject")])
        assert len(route.semantic_only_references) == 0


# ---------------------------------------------------------------------------
# OUTFIT_TRANSFER: exhaustive 8-combo matrix
# ---------------------------------------------------------------------------


class TestOutfitTransferMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert any("missing" in w for w in route.warnings)

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
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
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_content_role == "subject"
        assert route.target_geometry_mode == "favor_image"
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references,
            [(S, OUTFIT_TRANSFER_SUBJECT_BOOST, "subject"), (Ou, OUTFIT_TRANSFER_BOOST, "outfit")],
        )
        assert route.edit_references[0][1] == 8.0
        assert route.edit_references[1][1] == 6.0

    def test_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, OUTFIT_TRANSFER_BOOST, "outfit")])

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(S, NORMAL_BOOST, "subject"), (Ou, OUTFIT_TRANSFER_BOOST, "outfit")])

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="outfit_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert_refs(route.edit_references, [(S, NORMAL_BOOST, "subject"), (Sc, OUTFIT_TRANSFER_BOOST, "scene+outfit")])


# ---------------------------------------------------------------------------
# SUBJECT_TRANSFER: exhaustive routing matrix
# ---------------------------------------------------------------------------


class TestSubjectTransferMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset="subject_transfer")
        assert any("missing" in w for w in route.warnings)
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject")])
        assert route.edit_references[0][1] == pytest.approx(8.0)

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert_refs(route.edit_references, [(Sc, SUBJECT_TRANSFER_SCENE_BOOST, "scene")])
        assert route.edit_references[0][1] == pytest.approx(2.5)
        assert len(route.semantic_only_references) == 0

    def test_outfit_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is Ou
        assert_refs(route.edit_references, [(Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit")])
        assert route.edit_references[0][1] == pytest.approx(4.0)

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="subject_transfer", subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, SUBJECT_TRANSFER_SCENE_BOOST, "scene"),
                (S, SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST, "subject"),
            ],
        )
        assert route.edit_references[0][1] == pytest.approx(2.5)
        assert route.edit_references[1][1] == pytest.approx(7.0)
        assert len(route.semantic_only_references) == 0
        assert route.style_active is True
        assert route.style_source is Sc
        assert route.style_config.style_processing == "2x2"
        assert route.style_config.indirect_style_transfer is False

    def test_subject_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "empty"
        assert route.target_geometry_source is S
        assert_refs(
            route.edit_references,
            [(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject"), (Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit")],
        )
        assert route.edit_references[0][1] == pytest.approx(8.0)
        assert route.edit_references[1][1] == pytest.approx(4.0)

    def test_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert any("missing" in w for w in route.warnings)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, SUBJECT_TRANSFER_SCENE_BOOST, "scene"),
                (Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit"),
            ],
        )
        assert route.edit_references[0][1] == pytest.approx(2.5)
        assert route.edit_references[1][1] == pytest.approx(4.0)
        assert len(route.semantic_only_references) == 0

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, SUBJECT_TRANSFER_SCENE_BOOST, "scene"),
                (S, SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST, "subject"),
                (Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit"),
            ],
        )
        assert route.edit_references[0][1] == pytest.approx(2.5)
        assert route.edit_references[1][1] == pytest.approx(7.0)
        assert route.edit_references[2][1] == pytest.approx(4.0)
        assert len(route.semantic_only_references) == 0

    def test_subject_scene_as_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene+outfit"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, SUBJECT_TRANSFER_OUTFIT_BOOST, "scene+outfit"),
                (S, SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST, "subject"),
            ],
        )
        assert route.edit_references[0][1] == pytest.approx(4.0)
        assert route.edit_references[1][1] == pytest.approx(7.0)
        assert len(route.semantic_only_references) == 0


# ---------------------------------------------------------------------------
# IDENTITY_TRANSFER: exhaustive routing matrix
# ---------------------------------------------------------------------------


class TestIdentityTransferMatrix:
    def test_none(self, dummy_sources):
        sources = resolve_easy_sources(preset="identity_transfer")
        route = route_easy_preset(sources, preset="identity_transfer")
        assert any("missing" in w for w in route.warnings)
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="identity_transfer", subject=S)
        route = route_easy_preset(sources, preset="identity_transfer")
        assert any("Scene source is missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject")])

    def test_scene_only(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="identity_transfer", scene=Sc)
        route = route_easy_preset(sources, preset="identity_transfer")
        assert any("Subject source is missing" in w for w in route.warnings)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene")])

    def test_subject_scene(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="identity_transfer", subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="identity_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_content_role == "subject"
        assert route.target_content_fit == "contain_no_upscale"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"),
                (S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"),
            ],
        )
        assert route.edit_references[0][1] == pytest.approx(2.5)
        assert route.edit_references[1][1] == pytest.approx(7.0)
        assert len(route.semantic_only_references) == 0
        assert route.style_active is True
        assert route.style_source is Sc
        assert route.style_config.style_processing == "2x2"
        assert route.style_config.indirect_style_transfer is False

    def test_subject_scene_outfit_disabled(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="identity_transfer", subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="identity_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is S
        assert route.target_content_role == "subject"
        assert route.target_geometry_source is Sc
        # Outfit policy is disabled, so Ou is ignored completely
        assert_refs(
            route.edit_references,
            [
                (Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"),
                (S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"),
            ],
        )
        assert len(route.edit_references) == 2


class TestIdentityTestPresetsMatrix:
    def test_family_e_preset(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(preset="transfer_identity_preserve_scene_subject_5", subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="transfer_identity_preserve_scene_subject_5")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_content_fit == "crop"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [
                (Sc, 2.5, "scene"),
                (S, 5.0, "subject"),
            ],
        )
        assert len(route.edit_references) == 2
        assert route.style_active is True
        assert route.style_source is Sc


    def test_scene_as_outfit_without_subject(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene+outfit"
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [(Sc, SUBJECT_TRANSFER_OUTFIT_BOOST, "scene+outfit")],
        )
        assert route.edit_references[0][1] == pytest.approx(4.0)
        assert len(route.semantic_only_references) == 0
        assert any("missing" in w for w in route.warnings)

    def test_no_fallback_to_other_presets(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        # Without Scene, target_content_mode must be empty (NOT image S like preserve_identity, NOT image Ou like outfit_transfer)
        sources_s = resolve_easy_sources(subject=S)
        route_s = route_easy_preset(sources_s, preset="subject_transfer")
        assert route_s.target_content_mode == "empty"

        sources_sou = resolve_easy_sources(subject=S, outfit=Ou)
        route_sou = route_easy_preset(sources_sou, preset="subject_transfer")
        assert route_sou.target_content_mode == "empty"

    def test_subject_transfer_target_vision_dedup(self, dummy_sources):
        from ccc_krea2.target_latent import should_include_target_in_vision, TargetVisionContext
        from ccc_krea2.reference_specs import ReferenceSpec, ReferenceChain

        S, Sc, Ou, _ = dummy_sources
        ctx = TargetVisionContext(include_in_vision="auto", target_image=S)
        chain = ReferenceChain(
            specs=(
                ReferenceSpec(prepared_image=Sc, role="scene", attention_boost=2.5),
                ReferenceSpec(prepared_image=S, role="subject", attention_boost=7.0),
            )
        )
        assert should_include_target_in_vision(ctx, chain) is False

    def test_subject_transfer_subject_geometry_adaptation(self):
        import torch
        from ccc_krea2.target_latent import calculate_target_latent_resolution
        from ccc_krea2.vision_prep import prepare_vision_image

        scene_img = torch.rand(1, 304, 464, 3)  # 464x304 scene (~0.14 MP)
        sc_prep = prepare_vision_image(image=scene_img, clip=None, mode="native")
        th, tw, geom_src, active_mp, src_dims, warnings = calculate_target_latent_resolution(
            geometry_mode="favor_image",
            target_megapixels=2.0,
            geometry_image=sc_prep,
            force_target_megapixels=True,
        )
        assert th % 16 == 0
        assert tw % 16 == 0
        assert abs((tw / float(th)) - (464.0 / 304.0)) < 0.05
        assert (th * tw) / 1_000_000.0 == pytest.approx(2.0, abs=0.05)


class TestCommonMultiReferenceGeometry:
    """Test suite for generalized Easy Edit common multi-reference geometry logic."""

    def test_scene_subject_common_geometry(self):
        import torch
        from unittest.mock import MagicMock
        from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

        node = CcCKrea2EasyEdit()
        subj_img = torch.rand(1, 1000, 800, 3)  # 4:5 portrait subject
        scene_img = torch.rand(1, 300, 600, 3)  # 2:1 landscape scene

        def mock_tokenize(prompt, images=None, **kwargs):
            tok_pairs = []
            if images:
                for img in images:
                    tok_pairs.append([{"type": "image", "data": img}, None])
            else:
                tok_pairs.append([100, None])
            return {"qwen3vl": [tok_pairs]}

        mock_model = MagicMock()
        mock_clip = MagicMock()
        mock_clip.is_test_dummy = True
        mock_clip.tokenize.side_effect = mock_tokenize
        mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]
        mock_vae = MagicMock()
        mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 128))}

        _, _, _, _, report = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="balanced",
            subject=subj_img,
            scene=scene_img,
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: yes" in report
        assert "Common Geometry Anchor: scene" in report

    def test_subject_outfit_common_geometry(self):
        import torch
        from unittest.mock import MagicMock
        from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

        def mock_tokenize(prompt, images=None, **kwargs):
            tok_pairs = []
            if images:
                for img in images:
                    tok_pairs.append([{"type": "image", "data": img}, None])
            else:
                tok_pairs.append([100, None])
            return {"qwen3vl": [tok_pairs]}

        node = CcCKrea2EasyEdit()
        subj_img = torch.rand(1, 600, 300, 3)  # 1:2 portrait subject
        outfit_img = torch.rand(1, 400, 400, 3)  # 1:1 outfit
        mock_model = MagicMock()
        mock_clip = MagicMock()
        mock_clip.is_test_dummy = True
        mock_clip.tokenize.side_effect = mock_tokenize
        mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]
        mock_vae = MagicMock()
        mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 128, 64))}

        _, _, _, _, report = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="outfit_transfer",
            subject=subj_img,
            outfit=outfit_img,
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: yes" in report
        assert "Common Geometry Anchor: subject" in report

    def test_scene_subject_outfit_common_geometry(self):
        import torch
        from unittest.mock import MagicMock
        from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

        def mock_tokenize(prompt, images=None, **kwargs):
            tok_pairs = []
            if images:
                for img in images:
                    tok_pairs.append([{"type": "image", "data": img}, None])
            else:
                tok_pairs.append([100, None])
            return {"qwen3vl": [tok_pairs]}

        node = CcCKrea2EasyEdit()
        subj_img = torch.rand(1, 500, 500, 3)
        scene_img = torch.rand(1, 300, 600, 3)
        outfit_img = torch.rand(1, 400, 300, 3)
        mock_model = MagicMock()
        mock_clip = MagicMock()
        mock_clip.is_test_dummy = True
        mock_clip.tokenize.side_effect = mock_tokenize
        mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]
        mock_vae = MagicMock()
        mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 128))}

        _, _, _, _, report = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="consistent",
            subject=subj_img,
            scene=scene_img,
            outfit=outfit_img,
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: yes" in report
        assert "Common Geometry Anchor: scene" in report

    def test_scene_subject_outfit_dedup_common_geometry(self):
        import torch
        from unittest.mock import MagicMock
        from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

        def mock_tokenize(prompt, images=None, **kwargs):
            tok_pairs = []
            if images:
                for img in images:
                    tok_pairs.append([{"type": "image", "data": img}, None])
            else:
                tok_pairs.append([100, None])
            return {"qwen3vl": [tok_pairs]}

        node = CcCKrea2EasyEdit()
        subj_img = torch.rand(1, 500, 500, 3)
        scene_img = torch.rand(1, 300, 600, 3)
        mock_model = MagicMock()
        mock_clip = MagicMock()
        mock_clip.is_test_dummy = True
        mock_clip.tokenize.side_effect = mock_tokenize
        mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]
        mock_vae = MagicMock()
        mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 128))}

        _, _, _, _, report = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="subject_transfer",
            subject=subj_img,
            scene=scene_img,
            outfit_source="scene image",
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: yes" in report
        assert "Common Geometry Anchor: scene" in report

    def test_single_source_no_common_geometry(self):
        import torch
        from unittest.mock import MagicMock
        from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

        def mock_tokenize(prompt, images=None, **kwargs):
            tok_pairs = []
            if images:
                for img in images:
                    tok_pairs.append([{"type": "image", "data": img}, None])
            else:
                tok_pairs.append([100, None])
            return {"qwen3vl": [tok_pairs]}

        node = CcCKrea2EasyEdit()
        subj_img = torch.rand(1, 500, 500, 3)
        scene_img = torch.rand(1, 300, 600, 3)
        mock_model = MagicMock()
        mock_clip = MagicMock()
        mock_clip.is_test_dummy = True
        mock_clip.tokenize.side_effect = mock_tokenize
        mock_clip.encode_from_tokens_scheduled.return_value = [[torch.randn(1, 2000, 1536), {}]]
        mock_vae = MagicMock()
        mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

        # Subject only
        _, _, _, _, report_s = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="preserve_identity",
            subject=subj_img,
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: no" in report_s
        assert "Common Geometry Anchor: none" in report_s

        # Scene only
        _, _, _, _, report_sc = node.process(
            model=mock_model,
            clip=mock_clip,
            positive_prompt="test prompt",
            preset="preserve_scene",
            scene=scene_img,
            vae=mock_vae,
            apply_krea2_edit_patch=False,
        )
        assert "Common Geometry: no" in report_sc
        assert "Common Geometry Anchor: none" in report_sc

    def test_crop_alignment_zero_rope_offset(self):
        from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry

        # Target working geometry: 1152 x 1744 (H=1152, W=1744)
        # Source image: 1000 x 1000 (H=1000, W=1000)
        res = resolve_krea2edit_geometry(
            src_h=1000,
            src_w=1000,
            tgt_h=1152,
            tgt_w=1744,
            fit_mode="crop",
        )
        assert res.mode_resolved == "crop"
        assert res.vae_input_pixel_size == (1744, 1152)
        assert res.vae_latent_grid_size == (218, 144)
        assert res.centered_fractional_offset == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Style configs
# ---------------------------------------------------------------------------


def test_style_active_for_all_presets(dummy_sources):
    S, _, _, St = dummy_sources
    sources = resolve_easy_sources(subject=S, style=St)

    for p in (
        "flexible",
        "balanced",
        "consistent",
        "preserve_identity",
        "max_identity",
        "subject_transfer",
        "preserve_scene",
        "outfit_transfer",
    ):
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
        assert route.style_active is True
        assert route.style_source is St
        app_aliases = [alias for _, _, alias, _ in route.edit_references]
        assert "subject" in app_aliases
        assert "outfit" in app_aliases

    def test_style_from_scene_with_distinct_style(self, dummy_sources):
        """style_source=scene image: effective_style=Sc, but Sc also used as scene appearance ref."""
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, style_source="scene image")
        route = route_easy_preset(sources, preset="balanced")
        assert route.style_active is True
        assert route.style_source is Sc
        app_images = [img for img, _, _, _ in route.edit_references]
        assert Sc in app_images

    def test_semantic_only_references_routing(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="balanced")
        assert route.target_content_source is Sc
        assert len(route.edit_references) == 2
        assert (S, BALANCED_SUBJECT_BOOST, "subject") in [
            (img, boost, alias) for img, boost, alias, _ in route.edit_references
        ]
        assert (Ou, NORMAL_BOOST, "outfit") in [(img, boost, alias) for img, boost, alias, _ in route.edit_references]
        assert isinstance(route.semantic_only_references, tuple)


# ---------------------------------------------------------------------------
# Contract-aware appearance ref counts and capability gating tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [
        "flexible",
        "balanced",
        "consistent",
        "preserve_identity",
        "max_identity",
        "preserve_scene",
        "outfit_transfer",
        "style_transfer",
    ],
)
def test_bounded_appearance_refs(dummy_sources, preset):
    """Test that 2-ref bounded presets produce at most 2 appearance refs."""
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
        sources = resolve_easy_sources(preset=preset, **args)
        route = route_easy_preset(sources, preset=preset)
        assert len(route.edit_references) <= 2, (
            f"Preset '{preset}' with args {args} produced {len(route.edit_references)} refs"
        )


def test_subject_transfer_multi_reference_routing(dummy_sources):
    """Verify that subject_transfer with distinct Scene, Subject, and Outfit produces 3 appearance refs (scene, subject, outfit)."""
    S, Sc, Ou, _ = dummy_sources
    sources = resolve_easy_sources(preset="subject_transfer", subject=S, scene=Sc, outfit=Ou)
    route = route_easy_preset(sources, preset="subject_transfer")
    assert len(route.edit_references) == 3
    aliases = [alias for _, _, alias, _ in route.edit_references]
    assert aliases == ["scene", "subject", "outfit"]
    assert len(route.semantic_only_references) == 0


def test_preset_capability_gating_outfit(dummy_sources):
    """Verify preserve_scene, style_transfer, and scene_reinterpretation ignore Outfit even when connected."""
    S, Sc, Ou, _ = dummy_sources

    # preserve_scene with connected outfit
    sources_ps = resolve_easy_sources(preset="preserve_scene", subject=S, scene=Sc, outfit=Ou)
    assert sources_ps.effective_outfit is None
    route_ps = route_easy_preset(sources_ps, preset="preserve_scene")
    ps_aliases = [alias for _, _, alias, _ in route_ps.edit_references]
    assert "outfit" not in ps_aliases

    # style_transfer with connected outfit
    sources_st = resolve_easy_sources(preset="style_transfer", subject=S, outfit=Ou)
    assert sources_st.effective_outfit is None
    route_st = route_easy_preset(sources_st, preset="style_transfer")
    st_aliases = [alias for _, _, alias, _ in route_st.edit_references]
    assert "outfit" not in st_aliases

    # scene_reinterpretation with connected outfit
    sources_sr = resolve_easy_sources(preset="scene_reinterpretation", subject=S, scene=Sc, outfit=Ou)
    assert sources_sr.effective_outfit is None


def test_scene_reinterpretation_routing(dummy_sources):
    """Verify scene_reinterpretation preset behavior and routing contract."""
    S, Sc, Ou, St = dummy_sources

    # Scene + Subject -> target content empty, geometry Scene, style source scene
    sources = resolve_easy_sources(preset="scene_reinterpretation", subject=S, scene=Sc, outfit=Ou, style=St)
    assert sources.effective_outfit is None
    assert sources.effective_style is Sc

    route = route_easy_preset(sources, preset="scene_reinterpretation")
    assert route.target_content_mode == "empty"
    assert route.target_geometry_mode == "favor_image"
    assert route.target_geometry_source is Sc
    assert route.style_source is Sc

    aliases = [alias for _, _, alias, _ in route.edit_references]
    assert aliases == ["scene", "subject"]
    assert route.edit_references[0][1] == pytest.approx(1.0)  # Scene boost
    assert route.edit_references[1][1] == pytest.approx(4.0)  # Subject boost


def test_none_sources_resolution(dummy_sources):
    """Verify that setting outfit_source or style_source to 'none' disables the respective source."""
    S, Sc, Ou, St = dummy_sources

    sources = resolve_easy_sources(
        preset="balanced",
        subject=S,
        scene=Sc,
        outfit=Ou,
        style=St,
        outfit_source="none",
        style_source="none",
    )
    assert sources.effective_outfit is None
    assert sources.effective_style is None

    route = route_easy_preset(sources, preset="balanced")
    aliases = [alias for _, _, alias, _ in route.edit_references]
    assert "outfit" not in aliases
    assert route.style_source is None


def test_scene_auto_ignores_stale_style_source_selector(dummy_sources):
    """Verify that presets with scene_auto style policy use scene as style even if style_source selector is set to 'none' or 'style image'."""
    S, Sc, _, St = dummy_sources
    sources = resolve_easy_sources(
        preset="subject_transfer",
        subject=S,
        scene=Sc,
        style=St,
        style_source="none",  # Stale stored selector
    )
    assert sources.effective_style is Sc


def test_subject_transfer_node_visual_reference_fit_per_role(monkeypatch):
    """Verify that Subject Transfer uses visual_reference_fit == "contain" for Scene, Subject, Outfit, and Scene+Outfit appearance references."""
    import torch
    from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit

    S = torch.ones((1, 64, 64, 3), dtype=torch.float32)
    Sc = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5
    Ou = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.2
    node = CcCKrea2EasyEdit()

    class DummyVAE:
        def encode(self, x):
            return torch.zeros((1, 16, 8, 8), dtype=torch.float32)

    captured_chains = {}

    def mock_orchestrator(model, clip, vae, references, **kwargs):
        captured_chains["chain"] = references
        return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

    monkeypatch.setattr(
        "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
        mock_orchestrator,
    )

    # Subject Transfer + Subject + Scene
    node.process(
        model="model",
        clip="clip",
        vae=DummyVAE(),
        positive_prompt="",
        use_default_prompt=True,
        preset="subject_transfer",
        subject=S,
        scene=Sc,
    )
    st_chain = captured_chains["chain"]
    st_app_refs = [
        r for r in st_chain.references if r.reference_path == "edit" and getattr(r, "appearance_reference", False)
    ]
    assert len(st_app_refs) == 2
    ref_map = {r.alias: r.visual_reference_fit for r in st_app_refs}
    assert ref_map["scene"] == "contain"
    assert ref_map["subject"] == "contain"

    # Subject Transfer + Subject + Scene + distinct Outfit
    node.process(
        model="model",
        clip="clip",
        vae=DummyVAE(),
        positive_prompt="",
        use_default_prompt=True,
        preset="subject_transfer",
        subject=S,
        scene=Sc,
        outfit=Ou,
    )
    st_outfit_chain = captured_chains["chain"]
    st_outfit_refs = [
        r
        for r in st_outfit_chain.references
        if r.reference_path == "edit" and getattr(r, "appearance_reference", False)
    ]
    assert len(st_outfit_refs) == 3
    ref_map3 = {r.alias: r.visual_reference_fit for r in st_outfit_refs}
    assert ref_map3["scene"] == "contain"
    assert ref_map3["subject"] == "contain"
    assert ref_map3["outfit"] == "contain"

    # Subject Transfer + Subject + Scene-as-Outfit
    node.process(
        model="model",
        clip="clip",
        vae=DummyVAE(),
        positive_prompt="",
        use_default_prompt=True,
        preset="subject_transfer",
        subject=S,
        scene=Sc,
        outfit_source="scene image",
    )
    st_sc_outfit_chain = captured_chains["chain"]
    st_sc_outfit_refs = [
        r
        for r in st_sc_outfit_chain.references
        if r.reference_path == "edit" and getattr(r, "appearance_reference", False)
    ]
    assert len(st_sc_outfit_refs) == 2
    ref_map_sc = {r.alias: r.visual_reference_fit for r in st_sc_outfit_refs}
    assert ref_map_sc["scene+outfit"] == "contain"
    assert ref_map_sc["subject"] == "contain"

    # Balanced preset + Subject + Scene
    node.process(
        model="model",
        clip="clip",
        vae=DummyVAE(),
        positive_prompt="Custom balanced prompt",
        use_default_prompt=False,
        preset="balanced",
        subject=S,
        scene=Sc,
    )
    bal_chain = captured_chains["chain"]
    bal_edit_refs = [r for r in bal_chain.references if r.reference_path == "edit"]
    assert len(bal_edit_refs) == 2
    assert all(r.visual_reference_fit == "contain" for r in bal_edit_refs)


def test_easy_visual_reference_fit_invariant_all_presets_and_sources():
    """Verify that resolve_easy_visual_reference_fit returns contain for all 16 Easy Edit presets across Easy Edit appearance/edit reference roles (subject, scene, outfit, scene+outfit) under both single and multi-source conditions."""
    from ccc_krea2.easy_routing import EASY_PRESET_CAPABILITIES, resolve_easy_visual_reference_fit

    all_presets = list(EASY_PRESET_CAPABILITIES.keys())
    assert len(all_presets) == 43

    roles = ["subject", "scene", "outfit", "scene+outfit"]

    for preset in all_presets:
        for role in roles:
            # Multi-source / common geometry active
            fit_multi = resolve_easy_visual_reference_fit(preset, role, common_geometry_active=True)
            assert fit_multi == "contain", (
                f"Preset '{preset}' for role '{role}' under multi-source returned '{fit_multi}', expected 'contain'"
            )
            # Single-source / common geometry inactive
            fit_single = resolve_easy_visual_reference_fit(preset, role, common_geometry_active=False)
            assert fit_single == "contain", (
                f"Preset '{preset}' for role '{role}' under single-source returned '{fit_single}', expected 'contain'"
            )


@pytest.mark.parametrize(
    "preset,exp_sc_boost,exp_s_boost",
    [
        ("transfer_identity_preserve_scene_subject_1", 2.5, 1.0),
        ("transfer_identity_preserve_scene_subject_2", 2.5, 2.0),
        ("transfer_identity_preserve_scene_subject_2_5", 2.5, 2.5),
        ("transfer_identity_preserve_scene_subject_4", 2.5, 4.0),
        ("transfer_identity_preserve_scene_subject_5", 2.5, 5.0),
        ("transfer_identity_preserve_scene_subject_6", 2.5, 6.0),
        ("transfer_identity_preserve_scene_subject_8", 2.5, 8.0),
    ],
)
def test_identity_transfer_experimental_presets_family_e(preset, exp_sc_boost, exp_s_boost):
    """Verify Family E Preserve Scene presets route scene boost 2.5, subject boost, image content mode, and automatic generic scene style accurately."""
    from ccc_krea2.easy_routing import (
        resolve_easy_sources,
        route_easy_preset,
        get_easy_preset_capabilities,
        STYLE_POLICY_SCENE_AUTO,
        OUTFIT_POLICY_DISABLED,
    )

    S = object()
    Sc = object()

    caps = get_easy_preset_capabilities(preset)
    assert caps.style_policy == STYLE_POLICY_SCENE_AUTO
    assert caps.outfit_policy == OUTFIT_POLICY_DISABLED

    src = resolve_easy_sources(subject=S, scene=Sc, preset=preset)
    route = route_easy_preset(src, preset)

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_content_role == "scene"
    assert route.target_content_fit == "crop"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0][:3] == (Sc, exp_sc_boost, "scene")
    assert route.edit_references[1][:3] == (S, exp_s_boost, "subject")
    assert route.style_active is True
    assert route.style_source is Sc
    assert not any(alias == "outfit" for _, _, alias, _ in route.edit_references)


@pytest.mark.parametrize(
    "preset,exp_sc_boost,exp_s_boost",
    [
        ("transfer_identity_preserve_scene_subject_1_outfit_style", 2.5, 1.0),
        ("transfer_identity_preserve_scene_subject_2_outfit_style", 2.5, 2.0),
        ("transfer_identity_preserve_scene_subject_2_5_outfit_style", 2.5, 2.5),
        ("transfer_identity_preserve_scene_subject_4_outfit_style", 2.5, 4.0),
        ("transfer_identity_preserve_scene_subject_5_outfit_style", 2.5, 5.0),
        ("transfer_identity_preserve_scene_subject_6_outfit_style", 2.5, 6.0),
        ("transfer_identity_preserve_scene_subject_8_outfit_style", 2.5, 8.0),
    ],
)
def test_identity_transfer_experimental_presets_family_f(preset, exp_sc_boost, exp_s_boost):
    """Verify Family F Preserve Scene presets use automatic scene outfit style instruction and policy."""
    from ccc_krea2.easy_routing import (
        resolve_easy_sources,
        route_easy_preset,
        get_easy_preset_capabilities,
        STYLE_POLICY_SCENE_OUTFIT_AUTO,
        OUTFIT_POLICY_DISABLED,
        EASY_SCENE_OUTFIT_STYLE_INSTRUCTION,
    )

    S = object()
    Sc = object()

    caps = get_easy_preset_capabilities(preset)
    assert caps.style_policy == STYLE_POLICY_SCENE_OUTFIT_AUTO
    assert caps.outfit_policy == OUTFIT_POLICY_DISABLED

    src = resolve_easy_sources(subject=S, scene=Sc, preset=preset)
    route = route_easy_preset(src, preset)

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_content_role == "scene"
    assert route.target_content_fit == "crop"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0][:3] == (Sc, exp_sc_boost, "scene")
    assert route.edit_references[1][:3] == (S, exp_s_boost, "subject")
    assert route.style_active is True
    assert route.style_source is Sc
    assert route.style_config.style_fidelity == 1.0
    assert route.style_config.style_processing == "2x2"
    assert route.style_config.indirect_style_transfer is False
    assert route.style_config.vision_instruction == EASY_SCENE_OUTFIT_STYLE_INSTRUCTION
    assert not any(alias == "outfit" for _, _, alias, _ in route.edit_references)


@pytest.mark.parametrize(
    "preset,exp_sc_boost,exp_s_boost,exp_style_policy",
    [
        ("subject_transfer_1", 2.5, 5.0, "scene_auto"),
        ("subject_transfer_2", 2.5, 6.0, "scene_auto"),
        ("flexible_subject_transfer_1", 2.5, 5.0, "scene_outfit_auto"),
        ("flexible_subject_transfer_2", 2.5, 6.0, "scene_outfit_auto"),
    ],
)
def test_protected_subject_transfer_presets(preset, exp_sc_boost, exp_s_boost, exp_style_policy):
    """Verify protected Subject Transfer presets remain strictly unchanged in routing and capabilities."""
    from ccc_krea2.easy_routing import (
        resolve_easy_sources,
        route_easy_preset,
        get_easy_preset_capabilities,
        STYLE_POLICY_SCENE_AUTO,
        STYLE_POLICY_SCENE_OUTFIT_AUTO,
        OUTFIT_POLICY_DISABLED,
        EASY_SCENE_OUTFIT_STYLE_INSTRUCTION,
    )

    S = object()
    Sc = object()

    caps = get_easy_preset_capabilities(preset)
    if exp_style_policy == "scene_auto":
        assert caps.style_policy == STYLE_POLICY_SCENE_AUTO
    else:
        assert caps.style_policy == STYLE_POLICY_SCENE_OUTFIT_AUTO
    assert caps.outfit_policy == OUTFIT_POLICY_DISABLED

    src = resolve_easy_sources(subject=S, scene=Sc, preset=preset)
    route = route_easy_preset(src, preset)

    assert route.target_content_mode == "empty"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0][:3] == (Sc, exp_sc_boost, "scene")
    assert route.edit_references[1][:3] == (S, exp_s_boost, "subject")
    assert route.style_active is True
    assert route.style_source is Sc
    if exp_style_policy == "scene_outfit_auto":
        assert route.style_config.vision_instruction == EASY_SCENE_OUTFIT_STYLE_INSTRUCTION


def test_family_e_f_preserve_scene_fit_parity():
    """Verify Family E and Family F experimental presets match preserve_scene target_content_fit ('crop')."""
    from ccc_krea2.easy_routing import resolve_easy_sources, route_easy_preset

    S = object()
    Sc = object()

    src_base = resolve_easy_sources(subject=S, scene=Sc, preset="preserve_scene")
    route_base = route_easy_preset(src_base, preset="preserve_scene")

    src_e = resolve_easy_sources(subject=S, scene=Sc, preset="transfer_identity_preserve_scene_subject_5")
    route_e = route_easy_preset(src_e, preset="transfer_identity_preserve_scene_subject_5")

    src_f = resolve_easy_sources(subject=S, scene=Sc, preset="transfer_identity_preserve_scene_subject_5_outfit_style")
    route_f = route_easy_preset(src_f, preset="transfer_identity_preserve_scene_subject_5_outfit_style")

    assert route_base.target_content_fit == "crop"
    assert route_e.target_content_fit == "crop"
    assert route_f.target_content_fit == "crop"
    assert route_e.target_content_fit == route_base.target_content_fit
    assert route_f.target_content_fit == route_base.target_content_fit


@pytest.mark.parametrize(
    "preset,exp_sc_boost,exp_s_boost",
    [
        ("transfer_identity_scene_1_subject_1", 1.0, 1.0),
        ("transfer_identity_scene_2_subject_1", 2.0, 1.0),
        ("transfer_identity_scene_2_5_subject_1", 2.5, 1.0),
        ("transfer_identity_scene_4_subject_1", 4.0, 1.0),
        ("transfer_identity_scene_5_subject_1", 5.0, 1.0),
        ("transfer_identity_scene_6_subject_1", 6.0, 1.0),
        ("transfer_identity_scene_8_subject_1", 8.0, 1.0),
        ("transfer_identity_scene_1_subject_2", 1.0, 2.0),
        ("transfer_identity_scene_2_subject_2", 2.0, 2.0),
        ("transfer_identity_scene_2_5_subject_2", 2.5, 2.0),
        ("transfer_identity_scene_4_subject_2", 4.0, 2.0),
        ("transfer_identity_scene_5_subject_2", 5.0, 2.0),
        ("transfer_identity_scene_6_subject_2", 6.0, 2.0),
        ("transfer_identity_scene_8_subject_2", 8.0, 2.0),
    ],
)
def test_identity_transfer_scene_boost_experimental_presets(preset, exp_sc_boost, exp_s_boost):
    """Verify Scene Boost variants retain Family E routing, exact boost values, and generic Scene Style."""
    from ccc_krea2.easy_routing import (
        resolve_easy_sources,
        route_easy_preset,
        get_easy_preset_capabilities,
        STYLE_POLICY_SCENE_AUTO,
        OUTFIT_POLICY_DISABLED,
        EASY_SCENE_OUTFIT_STYLE_INSTRUCTION,
    )

    S = object()
    Sc = object()

    caps = get_easy_preset_capabilities(preset)
    assert caps.style_policy == STYLE_POLICY_SCENE_AUTO
    assert caps.outfit_policy == OUTFIT_POLICY_DISABLED

    src = resolve_easy_sources(subject=S, scene=Sc, preset=preset)
    route = route_easy_preset(src, preset)

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_content_role == "scene"
    assert route.target_content_fit == "crop"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0][:3] == (Sc, exp_sc_boost, "scene")
    assert route.edit_references[1][:3] == (S, exp_s_boost, "subject")
    assert route.style_active is True
    assert route.style_source is Sc
    assert route.style_config.vision_instruction != EASY_SCENE_OUTFIT_STYLE_INSTRUCTION
    assert not any(alias == "outfit" for _, _, alias, _ in route.edit_references)
