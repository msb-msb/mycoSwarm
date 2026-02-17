# Phase 31d: 8 C's Vital Signs — CC Implementation Guide

## Add to PLAN.md (under Phase 31)

```markdown
#### 31d: 8 C's Vital Signs (Self-Awareness)
- [ ] `vitals.py`: compute_vitals() derives 8 C's scores from existing signals
- [ ] Status bar after each response: 🧭 C:0.8 Cl:0.9 Cu:0.7 Co:0.6 ...
- [ ] `/vitals` slash command: detailed breakdown with explanations
- [ ] Alert mode: Monica flags when a score drops below threshold
- [ ] Vitals logged per-turn in session for longitudinal tracking
```

---

## CC Prompt — Implement Phase 31d

```
Implement Phase 31d: 8 C's Vital Signs — real-time self-awareness for Monica.

After each chat response, display a compact status bar showing Monica's
"vital signs" derived from signals that ALREADY EXIST in the codebase.
This is mostly repackaging existing metrics — minimal new computation.

### 1. Create src/mycoswarm/vitals.py

```python
"""
8 C's Vital Signs — Monica's self-awareness layer.

Each C is scored 0.0 to 1.0, derived from signals already in the pipeline.
No LLM calls — pure computation from existing data.
"""

from dataclasses import dataclass

@dataclass
class Vitals:
    calm: float        # Response stability
    clarity: float     # Grounding quality
    curiosity: float   # Retrieval breadth
    compassion: float  # Memory engagement
    courage: float     # Honesty about uncertainty
    creativity: float  # Novel connections
    connectedness: float  # Session/fact continuity
    confidence: float  # Source-backed certainty

    def status_bar(self) -> str:
        """Compact one-line display for after each response."""
        labels = [
            ("Ca", self.calm),
            ("Cl", self.clarity),
            ("Cu", self.curiosity),
            ("Cp", self.compassion),
            ("Co", self.courage),
            ("Cr", self.creativity),
            ("Cn", self.connectedness),
            ("Cf", self.confidence),
        ]
        parts = []
        for label, score in labels:
            # Color code: green >= 0.7, yellow >= 0.4, red < 0.4
            if score >= 0.7:
                parts.append(f"{label}:{score:.1f}")
            elif score >= 0.4:
                parts.append(f"{label}:{score:.1f}")
            else:
                parts.append(f"⚠{label}:{score:.1f}")
        return "🧭 " + " ".join(parts)

    def alerts(self) -> list[str]:
        """Return alert messages for any score below 0.4."""
        alerts = []
        if self.clarity < 0.4:
            alerts.append("My grounding is thin here — I may not have good sources for this.")
        if self.confidence < 0.4:
            alerts.append("I'm less certain about this answer than usual.")
        if self.connectedness < 0.4:
            alerts.append("I don't have much history on this topic with you.")
        if self.courage < 0.4:
            alerts.append("I notice I'm reaching beyond what my sources support.")
        if self.calm < 0.4:
            alerts.append("This query is pushing me outside my comfortable range.")
        return alerts

    def to_dict(self) -> dict:
        """For logging in session data."""
        return {
            "calm": round(self.calm, 2),
            "clarity": round(self.clarity, 2),
            "curiosity": round(self.curiosity, 2),
            "compassion": round(self.compassion, 2),
            "courage": round(self.courage, 2),
            "creativity": round(self.creativity, 2),
            "connectedness": round(self.connectedness, 2),
            "confidence": round(self.confidence, 2),
        }


def compute_vitals(
    grounding_score: float | None = None,
    source_count: int = 0,
    session_hits: int = 0,
    doc_hits: int = 0,
    procedure_hits: int = 0,
    fact_hits: int = 0,
    intent: dict | None = None,
    response_tokens: int = 0,
    retrieval_candidates: int = 0,
    tool_used: str | None = None,
    said_dont_know: bool = False,
) -> Vitals:
    """
    Derive 8 C's from existing pipeline signals.
    All inputs come from data already computed during a normal chat turn.
    """
    intent = intent or {}
    mode = intent.get("mode", "chat")
    tool = intent.get("tool", "answer")
    scope = intent.get("scope", "all")

    # --- CALM: response stability ---
    # High when: straightforward query, single tool, no fallbacks
    # Low when: complex multi-tool, very long response, error recovery
    calm = 0.7  # baseline
    if tool == "answer":
        calm = 0.9  # simple answer, no retrieval stress
    elif tool == "web_and_rag":
        calm = 0.5  # juggling multiple sources
    if response_tokens > 2000:
        calm -= 0.1  # long responses suggest complexity
    calm = max(0.0, min(1.0, calm))

    # --- CLARITY: grounding quality ---
    # Directly from grounding_score when available
    if grounding_score is not None:
        clarity = grounding_score
    elif source_count > 0:
        clarity = min(1.0, 0.5 + source_count * 0.1)
    elif tool == "answer":
        clarity = 0.6  # knowledge-based, no retrieval
    else:
        clarity = 0.3  # retrieval attempted but no grounding score
    clarity = max(0.0, min(1.0, clarity))

    # --- CURIOSITY: retrieval breadth ---
    # High when: explore mode, many candidates searched
    # Low when: chat mode, no retrieval
    if mode == "explore":
        curiosity = 0.9
    elif mode == "recall" and retrieval_candidates > 5:
        curiosity = 0.7
    elif tool in ("rag", "web_and_rag", "web_search"):
        curiosity = 0.6
    else:
        curiosity = 0.5  # chat mode — not bad, just not exploring
    curiosity = max(0.0, min(1.0, curiosity))

    # --- COMPASSION: memory engagement ---
    # High when: using facts, referencing sessions — showing it knows the user
    # Low when: generic response with no personalization
    compassion = 0.5  # baseline
    if fact_hits > 0:
        compassion += 0.2
    if session_hits > 0:
        compassion += 0.2
    if scope == "session":
        compassion += 0.1
    compassion = max(0.0, min(1.0, compassion))

    # --- COURAGE: honesty about uncertainty ---
    # High when: says "I don't know" when appropriate, low source count acknowledged
    # This is one of the hardest to measure — start simple
    courage = 0.6  # baseline
    if said_dont_know:
        courage = 0.9  # honesty is courage
    elif grounding_score is not None and grounding_score < 0.3 and source_count == 0:
        courage = 0.3  # should have said "I don't know" but probably didn't
    elif source_count >= 3:
        courage = 0.7  # well-sourced, doesn't need courage
    courage = max(0.0, min(1.0, courage))

    # --- CREATIVITY: novel connections ---
    # High when: procedures surfaced, cross-domain retrieval
    # Low when: direct recall, single-source answer
    creativity = 0.5  # baseline
    if procedure_hits > 0:
        creativity += 0.2  # wisdom/procedures = cross-domain thinking
    if doc_hits > 0 and session_hits > 0:
        creativity += 0.15  # connecting documents with conversation history
    if mode == "explore":
        creativity += 0.15
    creativity = max(0.0, min(1.0, creativity))

    # --- CONNECTEDNESS: continuity with user ---
    # High when: rich session history used, facts referenced, user is known
    # Low when: cold start, no context
    connectedness = 0.3  # baseline — grows with interaction
    if session_hits > 0:
        connectedness += 0.2
    if fact_hits > 0:
        connectedness += 0.2
    if session_hits >= 3:
        connectedness += 0.15  # deep history engagement
    if procedure_hits > 0:
        connectedness += 0.1
    connectedness = max(0.0, min(1.0, connectedness))

    # --- CONFIDENCE: source-backed certainty ---
    # Combines grounding with source count
    if grounding_score is not None:
        confidence = grounding_score * 0.6 + min(1.0, source_count * 0.15) * 0.4
    elif source_count >= 3:
        confidence = 0.7
    elif source_count > 0:
        confidence = 0.5
    elif tool == "answer":
        confidence = 0.6  # model knowledge, reasonable confidence
    else:
        confidence = 0.3
    confidence = max(0.0, min(1.0, confidence))

    return Vitals(
        calm=calm,
        clarity=clarity,
        curiosity=curiosity,
        compassion=compassion,
        courage=courage,
        creativity=creativity,
        connectedness=connectedness,
        confidence=confidence,
    )
```

### 2. Integrate into chat loop

In the chat loop (cli.py and solo.py), after each response is printed,
compute and display vitals.

The chat loop already has access to:
- grounding_score (from search_all or RAG pipeline)
- source counts (doc_hits, session_hits from search_all results)
- intent dict (from intent_classify)
- response token count (from Ollama response metadata)
- procedure_hits (from search_all 3-tuple)

After the existing timing line:
```
  ⏱  5.8s | 32.8 tok/s | gemma3:27b | node: Miu
```

Add the vitals line:
```
  🧭 Ca:0.8 Cl:0.9 Cu:0.7 Cp:0.6 Co:0.7 Cr:0.5 Cn:0.8 Cf:0.7
```

If any alerts exist, print them BEFORE the vitals line:
```
  💭 I'm less certain about this answer than usual.
  🧭 Ca:0.8 Cl:0.3 Cu:0.7 Cp:0.6 Co:0.7 Cr:0.5 Cn:0.8 ⚠Cf:0.3
```

The alert text is Monica speaking in first person — this is her
self-awareness. She's noticing her own state and reporting it.

### 3. /vitals slash command

Detailed breakdown with explanations:

```
/vitals output:

🧭 Monica's Vital Signs
─────────────────────────
  Calm:          0.8  ██████████░░  stable response, single tool
  Clarity:       0.9  ████████████  strong grounding (score: 0.92)
  Curiosity:     0.7  ██████████░░  recall mode, 8 candidates searched
  Compassion:    0.6  ████████░░░░  2 facts referenced, 1 session hit
  Courage:       0.7  ██████████░░  well-sourced, no hedging needed
  Creativity:    0.5  ██████░░░░░░  direct recall, single domain
  Connectedness: 0.8  ██████████░░  3 session hits, 2 facts used
  Confidence:    0.7  ██████████░░  grounding: 0.85, 3 sources
─────────────────────────
  Overall:       0.71 — healthy
```

### 4. Log vitals per turn in session

In the session turn data (wherever individual turns are tracked),
add a `vitals` field:

```python
turn_data = {
    "role": "assistant",
    "content": response_text,
    "vitals": vitals.to_dict(),
    "timestamp": now_iso,
}
```

This enables longitudinal tracking — over time, you can see
how Monica's vitals trend across sessions. Phase 31d-future
can use this for self-reflection: "My clarity has been improving
this week" or "I notice I'm less connected in morning sessions."

### 5. Tests

Add tests/test_vitals.py:

1. test_compute_vitals_simple_chat — answer/chat/all → high calm, moderate everything
2. test_compute_vitals_rag_grounded — high grounding → high clarity and confidence
3. test_compute_vitals_low_grounding — low grounding → alerts fire
4. test_compute_vitals_explore_mode — explore mode → high curiosity
5. test_compute_vitals_rich_memory — many session/fact hits → high connectedness/compassion
6. test_compute_vitals_said_dont_know — courage score high when honest
7. test_compute_vitals_procedures_used — procedure hits → creativity boost
8. test_status_bar_format — correct compact format string
9. test_alerts_below_threshold — alerts fire at < 0.4
10. test_alerts_above_threshold — no alerts when healthy
11. test_vitals_to_dict — serialization roundtrip

### 6. Commit

git add -A
git commit -m "Phase 31d: 8 C's Vital Signs — Monica's self-awareness. Status bar, /vitals command, alerts, per-turn logging"
git push

Do NOT bump version — this rides with Phase 31a in the same release.
```
