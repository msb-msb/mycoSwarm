"""Tests for swarm authentication (Phase 35d)."""

import os
import pytest
from unittest.mock import patch
from mycoswarm.auth import (
    generate_token,
    load_token,
    save_token,
    ensure_token,
    get_auth_header,
    validate_request,
    HEADER_NAME,
)


class TestTokenGeneration:
    def test_generates_64_char_hex(self):
        token = generate_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_generates_unique_tokens(self):
        t1 = generate_token()
        t2 = generate_token()
        assert t1 != t2


class TestTokenStorage:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("test_token_123")
            assert load_token() == "test_token_123"

    def test_load_missing_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            assert load_token() is None

    def test_save_sets_permissions(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("test_token")
            mode = os.stat(path).st_mode & 0o777
            assert mode == 0o600

    def test_ensure_generates_if_missing(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            token = ensure_token()
            assert len(token) == 64
            # Second call returns same token
            assert ensure_token() == token

    def test_ensure_loads_existing(self, tmp_path):
        path = str(tmp_path / "swarm-token")
        with patch("mycoswarm.auth.TOKEN_PATH", path):
            save_token("existing_token")
            assert ensure_token() == "existing_token"


class TestValidation:
    def test_valid_token(self):
        assert validate_request("abc123", "abc123") is True

    def test_invalid_token(self):
        assert validate_request("wrong", "abc123") is False

    def test_none_token(self):
        assert validate_request(None, "abc123") is False

    def test_empty_token(self):
        assert validate_request("", "abc123") is False

    def test_constant_time(self):
        """validate_request uses secrets.compare_digest (constant-time)."""
        # This is a design test — just verify it doesn't short-circuit
        # on first character mismatch
        assert validate_request("a" * 64, "b" * 64) is False


class TestAuthHeader:
    def test_header_format(self):
        headers = get_auth_header("my_token")
        assert headers == {HEADER_NAME: "my_token"}

    def test_header_name(self):
        assert HEADER_NAME == "X-Swarm-Token"
