"""Tests for the role→model binding resolver (src/mycoswarm/bindings.py).

Covers the option-B precedence: --model override > role binding > named fallback,
and the decoupling of model choice from session continuity.
"""

import pytest

from mycoswarm.bindings import (
    MODEL_BINDINGS,
    ROLE_FALLBACKS,
    bound_model,
    model_installed,
    resolve_model,
    unavailable_message,
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
        """Both are installed, so this isolates precedence: override wins."""
        model, how = resolve_model(
            "monica_chat", ["gemma3:27b", "qwen3.5:9b"], override="qwen3.5:9b"
        )
        assert (model, how) == ("qwen3.5:9b", "override")

    def test_override_honored_when_install_list_is_empty(self):
        """--model is the escape hatch. An empty installed list means enumeration
        failed (Ollama down / daemon unreachable), which is NOT evidence of
        absence — so the override is still honored and the HTTP layer catches it."""
        model, how = resolve_model("monica_chat", [], override="something:1b")
        assert (model, how) == ("something:1b", "override")

    def test_override_that_is_not_installed_is_unavailable(self):
        """When we CAN enumerate and the named model demonstrably isn't there,
        that's a typo, not an escape hatch. Surface it now, not as a 404 later."""
        model, how = resolve_model(
            "monica_chat", ["gemma3:1b", "gemma3:4b"], override="qwen3.5:70b"
        )
        assert (model, how) == (None, "unavailable")


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

    def test_fallback_must_itself_be_installed(self):
        """The regression under test. The bound 27b is absent, so the fallback
        fires — but the fallback is absent too. Returning the fallback name here
        is what produced the unhandled Ollama 404 on light nodes."""
        model, how = resolve_model("monica_chat", ["gemma3:1b"])
        assert how != "fallback"
        assert model != ROLE_FALLBACKS["monica_chat"]

    def test_unbound_role(self):
        model, how = resolve_model("no_such_role", ["gemma3:27b"])
        assert (model, how) == ("", "unbound")


class TestUnavailable:
    """The light-node case: neither the binding nor the fallback is present."""

    def test_boa_has_only_gemma3_1b(self):
        """boa, live: binding gemma3:27b absent, fallback qwen3.5:9b absent."""
        model, how = resolve_model("monica_chat", ["gemma3:1b"])
        assert (model, how) == (None, "unavailable")

    def test_luvia_models(self):
        """luvia, live: gemma3:4b / gemma3:1b / rwkv7:2.9b — no binding, no fallback."""
        installed = ["gemma3:4b", "gemma3:1b", "rwkv7:2.9b"]
        model, how = resolve_model("monica_chat", installed)
        assert (model, how) == (None, "unavailable")

    def test_no_models_at_all(self):
        model, how = resolve_model("monica_chat", [])
        assert (model, how) == (None, "unavailable")

    def test_model_is_none_iff_unavailable(self):
        """The contract callers rely on: None appears only with 'unavailable',
        so an uninstalled name can never reach Ollama by accident."""
        cases = [
            (["gemma3:27b"], None),
            (["qwen3.5:9b"], None),
            (["gemma3:1b"], None),
            ([], None),
            (["gemma3:27b"], "gemma3:27b"),
        ]
        for installed, override in cases:
            model, how = resolve_model("monica_chat", installed, override=override)
            assert (model is None) == (how == "unavailable"), (installed, override)

    def test_unavailable_message_names_node_role_and_both_models(self):
        msg = unavailable_message("monica_chat", node="boa")
        assert "boa" in msg
        assert "monica_chat" in msg
        assert "gemma3:27b" in msg      # what we wanted
        assert "qwen3.5:9b" in msg      # what we fell back to
        assert "ollama pull gemma3:27b" in msg
        assert "Traceback" not in msg


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


class TestCallerHandlesUnavailable:
    """Fix 2, layer 1: 'unavailable' must fail fast and legibly at resolution
    time — a clear message and a clean exit, never a stack trace."""

    def test_pick_model_exits_cleanly_on_light_node(self, capsys):
        """The boa path: `mycoswarm ask` on a node with only gemma3:1b."""
        from mycoswarm.solo import pick_model

        with pytest.raises(SystemExit) as exc:
            pick_model(["gemma3:1b"])
        assert exc.value.code == 1

        out = capsys.readouterr().out
        assert "No usable model installed" in out
        assert "monica_chat" in out
        assert "gemma3:27b" in out
        assert "qwen3.5:9b" in out
        assert "Traceback" not in out

    def test_pick_model_reports_a_bad_override_by_name(self, capsys):
        from mycoswarm.solo import pick_model

        with pytest.raises(SystemExit):
            pick_model(["gemma3:1b"], prefer="qwen3.5:70b")
        out = capsys.readouterr().out
        assert "qwen3.5:70b" in out
        assert "not installed" in out

    def test_pick_model_unaffected_when_the_model_is_present(self):
        """Zero-config nodes that have their model must be untouched."""
        from mycoswarm.solo import pick_model

        assert pick_model(["gemma3:27b", "gemma3:1b"]) == "gemma3:27b"
        assert pick_model(["qwen3.5:9b"]) == "qwen3.5:9b"      # fallback, installed
        assert pick_model(["gemma3:1b"], prefer="gemma3:1b") == "gemma3:1b"


class TestChatStreamHTTPStatusError:
    """Fix 2, layer 2: the safety net. Even if some path ever hands chat_stream
    an uninstalled model, Ollama's 404 must not propagate as a traceback."""

    def _raising_client(self, status: int):
        """A stand-in httpx.Client whose stream() raises HTTPStatusError."""
        import httpx

        class _Resp:
            status_code = status

            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    f"{status}", request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(status),
                )

        class _Stream:
            def __enter__(self_inner): return _Resp()
            def __exit__(self_inner, *a): return False

        class _Client:
            def __init__(self_inner, *a, **kw): pass
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): return False
            def stream(self_inner, *a, **kw): return _Stream()

        return _Client

    def test_404_is_caught_and_reported(self, capsys, monkeypatch):
        import mycoswarm.solo as solo

        monkeypatch.setattr(solo.httpx, "Client", self._raising_client(404))
        text, metrics = solo.chat_stream(
            [{"role": "user", "content": "hi"}], "qwen3.5:9b"
        )

        assert text == ""
        assert metrics == {}
        out = capsys.readouterr().out
        assert "qwen3.5:9b" in out
        assert "ollama pull qwen3.5:9b" in out
        assert "Traceback" not in out

    def test_other_bad_status_is_also_caught(self, capsys, monkeypatch):
        import mycoswarm.solo as solo

        monkeypatch.setattr(solo.httpx, "Client", self._raising_client(500))
        text, metrics = solo.chat_stream(
            [{"role": "user", "content": "hi"}], "gemma3:27b"
        )
        assert (text, metrics) == ("", {})
        assert "HTTP 500" in capsys.readouterr().out
