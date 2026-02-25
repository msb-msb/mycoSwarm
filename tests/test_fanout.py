"""Tests for Phase 37b: fan-out web search query variant generation."""

from mycoswarm.solo import generate_search_variants


def test_generate_search_variants_basic():
    """Original query is always first variant."""
    variants = generate_search_variants("latest AI news")
    assert variants[0] == "latest AI news"
    assert len(variants) >= 2


def test_generate_search_variants_keywords():
    """Filler words stripped in keyword variant."""
    variants = generate_search_variants("what is the latest news about AI safety")
    assert any("latest" in v and "what" not in v.lower() for v in variants[1:])


def test_generate_search_variants_year():
    """Recency variant appends current year."""
    from datetime import datetime
    year = str(datetime.now().year)
    variants = generate_search_variants("best local AI models")
    assert any(year in v for v in variants)


def test_generate_search_variants_year_already_present():
    """No duplicate year if already in query."""
    from datetime import datetime
    year = str(datetime.now().year)
    query = f"AI news {year}"
    variants = generate_search_variants(query)
    for v in variants:
        assert v.count(year) <= 1


def test_generate_search_variants_max():
    """Never returns more than max_variants."""
    variants = generate_search_variants("test query", max_variants=2)
    assert len(variants) <= 2


def test_generate_search_variants_short_query():
    """Short queries still produce at least 2 variants."""
    variants = generate_search_variants("bitcoin price")
    assert len(variants) >= 2


def test_generate_search_variants_single_word():
    """Single content word after filler removal still works."""
    variants = generate_search_variants("what is bitcoin")
    assert variants[0] == "what is bitcoin"
    assert len(variants) >= 2


def test_generate_search_variants_all_filler():
    """Query of only filler words still returns original + year variant."""
    variants = generate_search_variants("what is the")
    assert variants[0] == "what is the"
    # No keyword variant (only 1 word after filler removal)
    # But year variant should still appear
    assert len(variants) >= 2


def test_generate_search_variants_no_mutation():
    """Original query is never modified."""
    query = "How does photosynthesis work?"
    variants = generate_search_variants(query)
    assert variants[0] == query
