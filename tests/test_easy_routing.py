"""Table-driven test suite for Easy Edit 3-Phase Routing Engine."""

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
    O = DummyImg("Outfit")
    St = DummyImg("Style")
    return S, Sc, O, St


def test_balanced_routing_subject_only(dummy_sources):
    S, _, _, _ = dummy_sources
    sources = resolve_easy_sources(subject=S)
    route = route_easy_preset(sources, preset="balanced")

    assert route.target_content_mode == "empty"
    assert route.target_content_source is None
    assert route.target_geometry_mode == "favor_image"
    assert route.target_geometry_source is S
    assert len(route.edit_references) == 1
    assert route.edit_references[0] == (S, BALANCED_SUBJECT_BOOST, "subject")


def test_balanced_routing_subject_and_scene(dummy_sources):
    S, Sc, _, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc)
    route = route_easy_preset(sources, preset="balanced")

    assert route.target_content_mode == "empty"
    assert route.target_content_source is None
    assert route.target_geometry_mode == "favor_image"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (Sc, NORMAL_BOOST, "scene")
    assert route.edit_references[1] == (S, BALANCED_SUBJECT_BOOST, "subject")


def test_balanced_routing_subject_and_distinct_outfit(dummy_sources):
    S, _, O, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, outfit=O)
    route = route_easy_preset(sources, preset="balanced")

    assert route.target_content_mode == "empty"
    assert route.target_geometry_mode == "favor_image"
    assert route.target_geometry_source is S
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (S, BALANCED_SUBJECT_BOOST, "subject")
    assert route.edit_references[1] == (O, NORMAL_BOOST, "outfit")


def test_balanced_routing_3_sources(dummy_sources):
    S, Sc, O, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc, outfit=O)
    route = route_easy_preset(sources, preset="balanced")

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_geometry_mode == "favor_image"
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (S, BALANCED_SUBJECT_BOOST, "subject")
    assert route.edit_references[1] == (O, NORMAL_BOOST, "outfit")


def test_preserve_identity_3_sources_outfit_emphasis(dummy_sources):
    S, Sc, O, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc, outfit=O)
    route = route_easy_preset(sources, preset="preserve_identity")

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject")
    assert route.edit_references[1] == (O, OUTFIT_EMPHASIS_BOOST, "outfit")


def test_max_identity_routing(dummy_sources):
    S, Sc, O, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc, outfit=O)
    route = route_easy_preset(sources, preset="max_identity")

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (S, MAX_IDENTITY_SUBJECT_BOOST, "subject")
    assert route.edit_references[1] == (O, NORMAL_BOOST, "outfit")


def test_preserve_scene_routing(dummy_sources):
    S, Sc, _, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc)
    route = route_easy_preset(sources, preset="preserve_scene")

    assert route.target_content_mode == "image"
    assert route.target_content_source is Sc
    assert route.target_geometry_source is Sc
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (Sc, PRESERVE_SCENE_BOOST, "scene")
    assert route.edit_references[1] == (S, NORMAL_BOOST, "subject")


def test_outfit_transfer_routing(dummy_sources):
    S, _, O, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, outfit=O)
    route = route_easy_preset(sources, preset="outfit_transfer")

    assert route.target_content_mode == "image"
    assert route.target_content_source is O
    assert route.target_geometry_source is O
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (S, NORMAL_BOOST, "subject")
    assert route.edit_references[1] == (O, OUTFIT_TRANSFER_BOOST, "outfit")


def test_outfit_source_selector_scene_image(dummy_sources):
    S, Sc, _, _ = dummy_sources
    sources = resolve_easy_sources(subject=S, scene=Sc, outfit_source="scene image")

    assert sources.effective_outfit is Sc
    assert len(sources.warnings) == 0

    route = route_easy_preset(sources, preset="balanced")
    # effective_outfit == scene -> do not duplicate scene
    assert len(route.edit_references) == 2
    assert route.edit_references[0] == (Sc, NORMAL_BOOST, "scene")
    assert route.edit_references[1] == (S, BALANCED_SUBJECT_BOOST, "subject")


def test_style_source_selector_disconnected_warning():
    sources = resolve_easy_sources(style=None, style_source="style image")
    assert sources.effective_style is None
    assert len(sources.warnings) == 0

    sources_disc = resolve_easy_sources(scene=None, style_source="scene image")
    assert sources_disc.effective_style is None
    assert len(sources_disc.warnings) == 1
    assert "disconnected" in sources_disc.warnings[0]


def test_style_active_for_all_presets(dummy_sources):
    S, _, _, St = dummy_sources
    sources = resolve_easy_sources(subject=S, style=St)

    for p in ("balanced", "preserve_identity", "max_identity", "preserve_scene", "outfit_transfer"):
        r = route_easy_preset(sources, preset=p)
        assert r.style_active is True
        assert r.style_source is St
        assert r.style_strength == 1.0

    r_st = route_easy_preset(sources, preset="style_transfer")
    assert r_st.style_active is True
    assert r_st.style_source is St
    assert r_st.style_strength == 2.0
