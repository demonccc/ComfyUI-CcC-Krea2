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


def assert_no_more_than_2_refs(route):
    assert len(route.edit_references) <= 2, f"Too many appearance refs: {route.edit_references}"


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


@pytest.mark.parametrize("preset", ["balanced", "style_transfer"])
class TestBalancedMatrix:
    def test_none(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources()
        route = route_easy_preset(sources, preset=preset)
        assert len(route.edit_references) == 0

    def test_subject_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_scene_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene")])
        assert_no_more_than_2_refs(route)

    def test_outfit_only(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Ou
        assert_refs(route.edit_references, [(Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_scene(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (S, BALANCED_SUBJECT_BOOST, "subject")])
        assert_no_more_than_2_refs(route)

    def test_subject_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "empty"
        assert route.target_content_source is None
        assert route.target_geometry_source is S
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_scene_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(Sc, NORMAL_BOOST, "scene"), (Ou, NORMAL_BOOST, "outfit")])
        assert_no_more_than_2_refs(route)

    def test_scene_as_outfit(self, dummy_sources, preset):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(scene=Sc, outfit_source="scene image")
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
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
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
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")
        route = route_easy_preset(sources, preset=preset)
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_content_role == "scene+outfit"
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, BALANCED_SUBJECT_BOOST, "subject"), (Sc, NORMAL_BOOST, "scene+outfit")])
        assert_no_more_than_2_refs(route)

    def test_subject_outfit_from_style_distinct(self, dummy_sources, preset):
        S, Sc, Ou, St = dummy_sources
        sources = resolve_easy_sources(subject=S, style=St, outfit_source="style image")
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
        assert len(route.edit_references) == 0

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
        sources = resolve_easy_sources(subject=S, scene=Sc)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_geometry_source is Sc
        assert_refs(route.edit_references, [(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject")])
        assert route.edit_references[0][1] == pytest.approx(8.0)

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
        assert_refs(route.edit_references, [(Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit")])
        assert route.edit_references[0][1] == pytest.approx(4.0)

    def test_subject_scene_outfit(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        sources = resolve_easy_sources(subject=S, scene=Sc, outfit=Ou)
        route = route_easy_preset(sources, preset="subject_transfer")
        assert route.target_content_mode == "image"
        assert route.target_content_source is Sc
        assert route.target_geometry_source is Sc
        assert_refs(
            route.edit_references,
            [(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject"), (Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit")],
        )
        assert route.edit_references[0][1] == pytest.approx(8.0)
        assert route.edit_references[1][1] == pytest.approx(4.0)

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
            [(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject"), (Sc, SUBJECT_TRANSFER_OUTFIT_BOOST, "scene+outfit")],
        )
        assert route.edit_references[0][1] == pytest.approx(8.0)
        assert route.edit_references[1][1] == pytest.approx(4.0)

    def test_no_fallback_to_other_presets(self, dummy_sources):
        S, Sc, Ou, _ = dummy_sources
        # Without Scene, target_content_mode must be empty (NOT image S like preserve_identity, NOT image Ou like outfit_transfer)
        sources_s = resolve_easy_sources(subject=S)
        route_s = route_easy_preset(sources_s, preset="subject_transfer")
        assert route_s.target_content_mode == "empty"

        sources_sou = resolve_easy_sources(subject=S, outfit=Ou)
        route_sou = route_easy_preset(sources_sou, preset="subject_transfer")
        assert route_sou.target_content_mode == "empty"


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
# Never more than 2 appearance refs — all 9 presets, several combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [
        "flexible",
        "balanced",
        "consistent",
        "preserve_identity",
        "max_identity",
        "subject_transfer",
        "preserve_scene",
        "outfit_transfer",
        "style_transfer",
    ],
)
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
