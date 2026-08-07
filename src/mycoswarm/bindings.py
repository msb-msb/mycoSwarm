"""Role → model bindings — the single source of truth for which model runs.

"What model is Monica running?" must have exactly one answer, and it must be a
*declared fact*, not an emergent property of Ollama's tag ordering or of whatever
model a prior chat session happened to save.

Resolution precedence (see ``resolve_model``), option B:

    1. explicit --model override  → wins outright. The escape hatch (still checked
                                     for installation — see ``resolve_model``).
    2. the role's binding          → the default. This BEATS any saved session model.
    3. a named, documented fallback → ONLY if the bound model is not installed on
                                      this node, and only if the fallback itself
                                      IS installed. Never a substring scan; logged
                                      loudly when it fires.
    4. otherwise                   → ``(None, "unavailable")``. Every resolution
                                      path returns either an installed model or
                                      nothing — never an uninstalled name.

There is deliberately NO substring matching here. The old ``14b``/``32b``/``27b``
scan is exactly why ``qwen3.5:35b-a3b`` was unreachable ("35b" matches none of the
tokens) and why ``qwen3.5:27b`` won the lottery over the intended ``gemma3:27b``.

To change the model, edit this file — one place, greppable, printable via
``mycoswarm model``.
"""

import socket

# --- The single source of truth -------------------------------------------------

MODEL_BINDINGS: dict[str, str] = {
    # Monica's interactive chat model. gemma3:27b is the model with the actual
    # track record: the Feb 24-27 session lineage, all of the vocabulary-emergence
    # work, and every session since July 8. It is DECLARED here, not selected by
    # matching substrings against Ollama's tag order.
    "monica_chat": "gemma3:27b",
    # Embedding role — the only other unambiguous, safe binding.
    "embedding": "nomic-embed-text",
}

# Named, documented fallbacks. Used ONLY when a role's bound model is absent on
# this node. This is a declared constant, NOT a substring scan. When a fallback
# fires, callers log it loudly, naming both the intended and the substituted model.
ROLE_FALLBACKS: dict[str, str] = {
    # A capable ~9b that fits nodes too small for a 27b (e.g. a 12GB card).
    "monica_chat": "qwen3.5:9b",
    "embedding": "nomic-embed-text",
}


def model_installed(name: str, installed) -> bool:
    """True if ``name`` is present, honoring Ollama's implicit ``:latest`` tag.

    Ollama stores an untagged pull as ``<name>:latest``, so a binding declared as
    ``nomic-embed-text`` is satisfied by an installed ``nomic-embed-text:latest``.
    This is exact-name matching, NOT the old substring lottery.
    """
    installed = set(installed or [])
    if name in installed:
        return True
    if ":" not in name and f"{name}:latest" in installed:
        return True
    return False


def bound_model(role: str) -> str | None:
    """Return the declared model for a role, or None if the role is unbound."""
    return MODEL_BINDINGS.get(role)


def role_fallback(role: str) -> str | None:
    """Return the named fallback model for a role, or None if none is declared."""
    return ROLE_FALLBACKS.get(role)


def unavailable_message(role: str, node: str | None = None) -> str:
    """The one canonical 'nothing usable here' message, shared by every caller.

    Names the node, the role, and both models we looked for, so a light node's
    failure is self-diagnosing: you can see what to pull without reading the code.
    """
    if node is None:
        node = socket.gethostname()
    bound = MODEL_BINDINGS.get(role)
    fb = ROLE_FALLBACKS.get(role)
    wanted = f"wanted '{bound}'" if bound else "no binding declared"
    fb_part = f", fallback '{fb}'" if fb else ", no fallback declared"
    hint = f"\n   Pull it with: ollama pull {bound}" if bound else ""
    return (
        f"❌ No usable model installed on {node} for role '{role}' "
        f"({wanted}{fb_part}).{hint}"
    )


def resolve_model(
    role: str,
    installed_models,
    override: str | None = None,
) -> tuple[str | None, str]:
    """Resolve which model to run for ``role``.

    Precedence (option B), explicit and returnable:

      1. ``override`` (the --model flag) → wins outright over binding and
         fallback. Returns ``(override, "override")``. It is still checked for
         installation: naming a model that demonstrably isn't there is a typo,
         not an escape hatch, and surfacing it here beats an Ollama 404 later.
         The check is skipped when ``installed_models`` is empty — an empty list
         means enumeration failed (Ollama down, daemon unreachable), which is not
         evidence of absence, so the override is honored and the HTTP layer stays
         the safety net.
      2. the role binding → the default. Returns ``(bound, "binding")`` when the
         bound model is installed on this node. NOTE: a session's saved model is
         *not an input here* — continuity of history is decoupled from model choice,
         so the binding always beats a stale saved model by construction.
      3. the named fallback → only when the bound model is absent, AND only when
         the fallback is itself installed. Returns ``(fallback, "fallback")``.
         Caller is expected to log loudly.
      4. nothing usable → returns ``(None, "unavailable")``. This is the light-node
         case (bound 27b absent, 9b fallback also absent). Returning ``None``
         rather than an uninstalled name is deliberate: it cannot be passed to
         Ollama by accident, so a caller that ignores ``how`` fails at the call
         site instead of 404-ing mid-stream.

    Args:
        role: the role key, e.g. ``"monica_chat"``.
        installed_models: iterable of model names present on this node/swarm.
        override: an explicit model name (the --model flag), or None.

    Returns:
        ``(model, how)`` where ``how`` is one of
        ``"override" | "binding" | "fallback" | "unbound" | "unavailable"``.
        ``model`` is None if and only if ``how == "unavailable"``.
    """
    installed = set(installed_models or [])

    if override:
        # Can't disprove absence against an empty list — honor the override.
        if not installed or model_installed(override, installed):
            return override, "override"
        return None, "unavailable"

    bound = MODEL_BINDINGS.get(role)
    if bound is None:
        return "", "unbound"

    if model_installed(bound, installed):
        return bound, "binding"

    # Bound model not installed here — try the named fallback, but only if it is
    # actually present. Handing back an uninstalled fallback is what produced the
    # unhandled 404 on light nodes that have neither the 27b nor the 9b.
    fallback = ROLE_FALLBACKS.get(role)
    if fallback is not None and model_installed(fallback, installed):
        return fallback, "fallback"

    return None, "unavailable"
