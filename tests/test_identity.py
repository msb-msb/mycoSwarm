"""Tests for Phase 31a: Identity Layer."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from mycoswarm.identity import (
    load_identity,
    save_identity,
    seed_identity,
    build_identity_prompt,
    IDENTITY_PATH,
)


@pytest.fixture(autouse=True)
def tmp_identity(tmp_path, monkeypatch):
    """Use temp dir for identity file."""
    id_path = tmp_path / "identity.json"
    monkeypatch.setattr("mycoswarm.identity.IDENTITY_PATH", id_path)
    return id_path


class TestLoadIdentity:
    def test_load_missing_file(self):
        assert load_identity() == {}

    def test_load_corrupt_file(self, tmp_identity):
        tmp_identity.write_text("not json{{{")
        assert load_identity() == {}


class TestSeedIdentity:
    def test_creates_correct_schema(self):
        identity = seed_identity("Monica")
        assert identity["name"] == "Monica"
        assert "Named by user" in identity["origin"]
        assert identity["substrate"] == "mycoSwarm distributed network"
        assert identity["developing"] is True
        assert "created" in identity

    def test_custom_origin(self):
        identity = seed_identity("Monica", origin="Named by Mark, February 2026")
        assert identity["origin"] == "Named by Mark, February 2026"


class TestSaveAndLoad:
    def test_roundtrip(self):
        identity = seed_identity("Monica")
        loaded = load_identity()
        assert loaded["name"] == "Monica"
        assert loaded["origin"] == identity["origin"]
        assert loaded["substrate"] == identity["substrate"]


class TestBuildIdentityPrompt:
    def test_with_name(self):
        identity = {
            "name": "Monica",
            "origin": "Named by Mark, February 2026",
            "substrate": "mycoSwarm distributed network",
            "developing": True,
        }
        prompt = build_identity_prompt(identity)
        assert "Monica" in prompt
        assert "named by mark, february 2026" in prompt.lower()
        assert "still developing" in prompt

    def test_empty_identity(self):
        prompt = build_identity_prompt({})
        assert "don't have a name yet" in prompt.lower()

    def test_developing_flag(self):
        prompt = build_identity_prompt({"name": "Monica", "developing": True})
        assert "still developing" in prompt

    def test_not_developing(self):
        prompt = build_identity_prompt({"name": "Monica", "developing": False})
        assert "still developing" not in prompt

    def test_self_referential_anchor(self):
        prompt = build_identity_prompt({"name": "Monica"})
        assert "You are Monica" in prompt
        assert "not a generic assistant" in prompt


class TestIdentityFactType:
    def test_identity_type_no_stale(self, tmp_path, monkeypatch):
        """Identity facts should never appear in stale list."""
        from mycoswarm.memory import (
            get_stale_facts,
            FACT_TYPE_IDENTITY,
            MEMORY_DIR,
        )
        monkeypatch.setattr("mycoswarm.memory.MEMORY_DIR", tmp_path)
        facts_path = tmp_path / "facts.json"
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        facts = [
            {
                "id": 1,
                "text": "My name is Monica",
                "type": "identity",
                "added": old_date,
                "last_referenced": old_date,
                "ref_count": 0,
            },
        ]
        facts_path.write_text(json.dumps(facts))
        stale = get_stale_facts(days=30)
        assert len(stale) == 0

    def test_identity_type_valid(self):
        """Identity should be a valid fact type."""
        from mycoswarm.memory import VALID_FACT_TYPES
        assert "identity" in VALID_FACT_TYPES
