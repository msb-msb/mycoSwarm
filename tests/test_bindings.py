"""Tests for the role→model binding resolver (src/mycoswarm/bindings.py).

Covers the option-B precedence: --model override > role binding > named fallback,
and the decoupling of model choice from session continuity.
"""

from mycoswarm.bindings import (
    MODEL_BINDINGS,
    ROLE_FALLBACKS,
    bound_model,
    model_installed,
    resolve_model,
)


class TestBindingValue:
    def test_monica_chat_is_gemma_not_qwen(self):
        """The declared model is gemma3:27b — NOT the qwen3.5:27b lottery result.

        gemma3:27b has the actual track record. Guards against re-enshrining the
        accidental substring-lottery model that the July 8 notes used as an example.
        """
        assert MODEL_BINDINGS["monica_chat"] == "gemma3:27b"
        assert bound_model("monica_chat") == "gemma3:27b"

    def test_embedding_binding(self):
        assert MODEL_BINDINGS["embedding"] == "nomic-embed-text"


class TestPrecedence:
    def test_binding_is_the_default(self):
        model, how = resolve_model("monica_chat", ["gemma3:27b", "qwen3.5:27b"])
        assert (model, how) == ("gemma3:27b", "binding")

    def test_binding_beats_stale_saved_model(self):
        """A session's saved model must NOT win. resolve_model has no saved-model
        input at all — continuity of history is decoupled from model choice, so the
        binding wins by construction even when the stale model is installed."""
        installed = ["qwen3.5:35b-a3b", "gemma3:27b", "qwen3.5:9b"]  # 35b = old pin
        model, how = resolve_model("monica_chat", installed, override=None)
        assert model == "gemma3:27b"
        assert how == "binding"

    def test_override_beats_binding(self):
        model, how = resolve_model(
            "monica_chat", ["gemma3:27b"], override="qwen3.5:9b"
        )
        assert (model, how) == ("qwen3.5:9b", "override")

    def test_override_wins_even_if_not_installed(self):
        """--model is the escape hatch: it wins outright, always."""
        model, how = resolve_model("monica_chat", [], override="something:1b")
        assert (model, how) == ("something:1b", "override")


class TestFallback:
    def test_fallback_fires_when_bound_absent(self):
        model, how = resolve_model("monica_chat", ["qwen3.5:9b"])
        assert how == "fallback"
        assert model == ROLE_FALLBACKS["monica_chat"] == "qwen3.5:9b"

    def test_fallback_is_named_not_a_substring_lottery(self):
        """The old scan would have picked qwen2.5:32b for the '32b' substring.
        The named fallback must win instead — the substring trap is dead."""
        installed = ["qwen2.5:32b-instruct-q4_K_M", "qwen3.5:9b"]
        model, how = resolve_model("monica_chat", installed)
        assert how == "fallback"
        assert model == "qwen3.5:9b"  # named fallback, NOT the "32b" match

    def test_unbound_role(self):
        model, how = resolve_model("no_such_role", ["gemma3:27b"])
        assert (model, how) == ("", "unbound")


class TestLatestTag:
    def test_untagged_binding_matches_latest_install(self):
        """Ollama stores an untagged pull as `<name>:latest`. A binding written
        as `nomic-embed-text` must be satisfied by `nomic-embed-text:latest`."""
        assert model_installed("nomic-embed-text", ["nomic-embed-text:latest"])
        model, how = resolve_model("embedding", ["nomic-embed-text:latest"])
        assert (model, how) == ("nomic-embed-text", "binding")

    def test_exact_tag_still_matches(self):
        assert model_installed("gemma3:27b", ["gemma3:27b", "gemma3:1b"])

    def test_tagged_binding_not_satisfied_by_different_tag(self):
        """A binding WITH an explicit tag must not match a different tag."""
        assert not model_installed("gemma3:27b", ["gemma3:1b", "gemma3:4b"])
