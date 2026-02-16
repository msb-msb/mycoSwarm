"""Tests for Phase 21a: Fact Lifecycle Tags."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from mycoswarm.memory import (
    FACT_TYPE_PREFERENCE,
    FACT_TYPE_FACT,
    FACT_TYPE_PROJECT,
    FACT_TYPE_EPHEMERAL,
    DEFAULT_FACT_TYPE,
    VALID_FACT_TYPES,
    _migrate_fact,
    load_facts,
    save_facts,
    add_fact,
    remove_fact,
    reference_fact,
    get_stale_facts,
    format_facts_for_prompt,
)


# --- Migration ---


def test_migrate_adds_missing_fields():
    old = {"id": 1, "text": "test", "added": "2026-01-01T00:00:00"}
    migrated = _migrate_fact(old)
    assert migrated["type"] == DEFAULT_FACT_TYPE
    assert migrated["last_referenced"] == "2026-01-01T00:00:00"
    assert migrated["reference_count"] == 0


def test_migrate_preserves_existing_fields():
    full = {
        "id": 1, "text": "test", "added": "2026-01-01T00:00:00",
        "type": "preference", "last_referenced": "2026-02-01T00:00:00",
        "reference_count": 5,
    }
    migrated = _migrate_fact(full)
    assert migrated["type"] == "preference"
    assert migrated["last_referenced"] == "2026-02-01T00:00:00"
    assert migrated["reference_count"] == 5


# --- add_fact with type ---


def test_add_fact_default_type(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("Mark teaches Tai Chi")
        assert fact["type"] == FACT_TYPE_FACT
        assert fact["reference_count"] == 0
        assert "last_referenced" in fact


def test_add_fact_preference_type(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("Prefers carnivore diet", fact_type=FACT_TYPE_PREFERENCE)
        assert fact["type"] == FACT_TYPE_PREFERENCE


def test_add_fact_project_type(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("Working on mycoSwarm v0.1.9", fact_type=FACT_TYPE_PROJECT)
        assert fact["type"] == FACT_TYPE_PROJECT


def test_add_fact_ephemeral_type(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("Need to buy smoker fuel", fact_type=FACT_TYPE_EPHEMERAL)
        assert fact["type"] == FACT_TYPE_EPHEMERAL


def test_add_fact_invalid_type_defaults(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("Some text", fact_type="bogus")
        assert fact["type"] == DEFAULT_FACT_TYPE


# --- load_facts migration ---


def test_load_facts_migrates_old_format(tmp_path):
    old_data = {
        "version": 1,
        "facts": [
            {"id": 1, "text": "old fact", "added": "2026-01-01T00:00:00"},
        ],
    }
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(old_data))
    with patch("mycoswarm.memory.FACTS_PATH", facts_path):
        facts = load_facts()
        assert facts[0]["type"] == DEFAULT_FACT_TYPE
        assert facts[0]["reference_count"] == 0
        assert facts[0]["last_referenced"] == "2026-01-01T00:00:00"


# --- reference_fact ---


def test_reference_fact_updates_timestamp(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        fact = add_fact("test fact")
        original_ref = fact["last_referenced"]
        assert reference_fact(fact["id"]) is True
        facts = load_facts()
        assert facts[0]["reference_count"] == 1
        assert facts[0]["last_referenced"] >= original_ref


def test_reference_fact_not_found(tmp_path):
    with patch("mycoswarm.memory.FACTS_PATH", tmp_path / "facts.json"), \
         patch("mycoswarm.memory.MEMORY_DIR", tmp_path):
        assert reference_fact(999) is False


# --- get_stale_facts ---


def test_stale_facts_detects_old(tmp_path):
    old_date = (datetime.now() - timedelta(days=45)).isoformat()
    data = {
        "version": 2,
        "facts": [
            {"id": 1, "text": "stale", "type": "fact", "added": old_date,
             "last_referenced": old_date, "reference_count": 0},
            {"id": 2, "text": "fresh", "type": "fact",
             "added": datetime.now().isoformat(),
             "last_referenced": datetime.now().isoformat(),
             "reference_count": 3},
        ],
    }
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(data))
    with patch("mycoswarm.memory.FACTS_PATH", facts_path):
        stale = get_stale_facts(days=30)
        assert len(stale) == 1
        assert stale[0]["text"] == "stale"


def test_ephemeral_facts_shorter_window(tmp_path):
    ten_days_ago = (datetime.now() - timedelta(days=10)).isoformat()
    data = {
        "version": 2,
        "facts": [
            {"id": 1, "text": "temp note", "type": "ephemeral",
             "added": ten_days_ago, "last_referenced": ten_days_ago,
             "reference_count": 0},
            {"id": 2, "text": "regular fact", "type": "fact",
             "added": ten_days_ago, "last_referenced": ten_days_ago,
             "reference_count": 0},
        ],
    }
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(data))
    with patch("mycoswarm.memory.FACTS_PATH", facts_path):
        stale = get_stale_facts(days=30)
        # Ephemeral is stale at 10 days (threshold=7), regular is not (threshold=30)
        assert len(stale) == 1
        assert stale[0]["type"] == "ephemeral"


# --- format_facts_for_prompt ---


def test_format_groups_by_type():
    facts = [
        {"id": 1, "text": "Lives in Berkeley", "type": "fact"},
        {"id": 2, "text": "Prefers Python", "type": "preference"},
        {"id": 3, "text": "Building mycoSwarm", "type": "project"},
        {"id": 4, "text": "Buy smoker fuel", "type": "ephemeral"},
    ]
    result = format_facts_for_prompt(facts)
    assert "Known facts about the user:" in result
    assert "User preferences:" in result
    assert "Active projects:" in result
    assert "Temporary notes:" in result
    # Facts come before preferences in output
    assert result.index("Known facts") < result.index("User preferences")
    assert result.index("User preferences") < result.index("Active projects")


def test_format_empty_types_skipped():
    facts = [
        {"id": 1, "text": "Just a fact", "type": "fact"},
    ]
    result = format_facts_for_prompt(facts)
    assert "Known facts" in result
    assert "preferences" not in result
    assert "projects" not in result


def test_format_empty_facts():
    assert format_facts_for_prompt([]) == ""
