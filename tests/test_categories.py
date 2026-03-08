"""Tests for article category profiles and research_mode defaults."""

from mycoswarm.categories import CATEGORIES, get_category, list_categories


class TestCategoryResearchMode:
    """Verify research_mode field on all category profiles."""

    def test_all_categories_have_research_mode(self):
        for name, profile in CATEGORIES.items():
            assert "research_mode" in profile, f"{name} missing research_mode"
            assert profile["research_mode"] in ("standard", "rlm"), (
                f"{name} has invalid research_mode: {profile['research_mode']}"
            )

    def test_rlm_defaults(self):
        rlm_categories = ["research-driven", "buying-guide", "vs", "how-to"]
        for name in rlm_categories:
            assert CATEGORIES[name]["research_mode"] == "rlm", (
                f"{name} should default to rlm"
            )

    def test_standard_defaults(self):
        standard_categories = [
            "news", "model-release", "experience-driven", "explainer", "opinion",
        ]
        for name in standard_categories:
            assert CATEGORIES[name]["research_mode"] == "standard", (
                f"{name} should default to standard"
            )

    def test_get_category_includes_research_mode(self):
        cat = get_category("buying-guide")
        assert cat["research_mode"] == "rlm"

    def test_get_category_fallback_includes_research_mode(self):
        cat = get_category("nonexistent-category")
        assert "research_mode" in cat
        assert cat["research_mode"] == "rlm"  # falls back to research-driven


class TestCLIResearchModeResolution:
    """Verify CLI override logic for research_mode."""

    def test_category_default_used_when_no_flag(self):
        """Simulate: no --research-mode flag → use category default."""
        category = get_category("research-driven")
        cli_research_mode = None  # user didn't pass --research-mode
        resolved = cli_research_mode if cli_research_mode else category.get("research_mode", "standard")
        assert resolved == "rlm"

    def test_cli_flag_overrides_category(self):
        """Simulate: --research-mode standard on an rlm category."""
        category = get_category("research-driven")
        cli_research_mode = "standard"  # user explicitly passed --research-mode standard
        resolved = cli_research_mode if cli_research_mode else category.get("research_mode", "standard")
        assert resolved == "standard"

    def test_cli_rlm_overrides_standard_category(self):
        """Simulate: --research-mode rlm on a standard category."""
        category = get_category("news")
        cli_research_mode = "rlm"
        resolved = cli_research_mode if cli_research_mode else category.get("research_mode", "standard")
        assert resolved == "rlm"

    def test_standard_category_default_when_no_flag(self):
        """Simulate: no --research-mode on a standard category."""
        category = get_category("news")
        cli_research_mode = None
        resolved = cli_research_mode if cli_research_mode else category.get("research_mode", "standard")
        assert resolved == "standard"
