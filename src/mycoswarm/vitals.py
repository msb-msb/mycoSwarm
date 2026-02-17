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
            if score < 0.4:
                parts.append(f"\u26a0{label}:{score:.1f}")
            else:
                parts.append(f"{label}:{score:.1f}")
        return "\U0001f9ed " + " ".join(parts)

    def alerts(self) -> list[str]:
        """Return alert messages for any score below 0.4."""
        alerts = []
        if self.clarity < 0.4:
            alerts.append("My grounding is thin here \u2014 I may not have good sources for this.")
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

    def overall(self) -> float:
        """Average of all 8 scores."""
        vals = [
            self.calm, self.clarity, self.curiosity, self.compassion,
            self.courage, self.creativity, self.connectedness, self.confidence,
        ]
        return round(sum(vals) / len(vals), 2)

    def detailed_display(self, name: str = "Monica") -> str:
        """Full breakdown for /vitals command."""
        def bar(score: float) -> str:
            filled = int(score * 12)
            return "\u2588" * filled + "\u2591" * (12 - filled)

        overall = self.overall()
        if overall >= 0.7:
            health = "healthy"
        elif overall >= 0.4:
            health = "moderate"
        else:
            health = "stressed"

        lines = [
            f"\U0001f9ed {name}'s Vital Signs",
            "\u2500" * 25,
            f"  Calm:          {self.calm:.1f}  {bar(self.calm)}",
            f"  Clarity:       {self.clarity:.1f}  {bar(self.clarity)}",
            f"  Curiosity:     {self.curiosity:.1f}  {bar(self.curiosity)}",
            f"  Compassion:    {self.compassion:.1f}  {bar(self.compassion)}",
            f"  Courage:       {self.courage:.1f}  {bar(self.courage)}",
            f"  Creativity:    {self.creativity:.1f}  {bar(self.creativity)}",
            f"  Connectedness: {self.connectedness:.1f}  {bar(self.connectedness)}",
            f"  Confidence:    {self.confidence:.1f}  {bar(self.confidence)}",
            "\u2500" * 25,
            f"  Overall:       {overall:.2f} \u2014 {health}",
        ]
        return "\n".join(lines)


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
    """Derive 8 C's from existing pipeline signals.

    All inputs come from data already computed during a normal chat turn.
    """
    intent = intent or {}
    mode = intent.get("mode", "chat")
    tool = intent.get("tool", "answer")

    # --- CALM: response stability ---
    calm = 0.7
    if tool == "answer":
        calm = 0.9
    elif tool == "web_and_rag":
        calm = 0.5
    if response_tokens > 2000:
        calm -= 0.1
    calm = max(0.0, min(1.0, calm))

    # --- CLARITY: grounding quality ---
    if grounding_score is not None:
        clarity = grounding_score
    elif source_count > 0:
        clarity = min(1.0, 0.5 + source_count * 0.1)
    elif tool == "answer":
        clarity = 0.6
    else:
        clarity = 0.3
    clarity = max(0.0, min(1.0, clarity))

    # --- CURIOSITY: retrieval breadth ---
    if mode == "explore":
        curiosity = 0.9
    elif mode == "recall" and retrieval_candidates > 5:
        curiosity = 0.7
    elif tool in ("rag", "web_and_rag", "web_search"):
        curiosity = 0.6
    else:
        curiosity = 0.5
    curiosity = max(0.0, min(1.0, curiosity))

    # --- COMPASSION: memory engagement ---
    compassion = 0.5
    if fact_hits > 0:
        compassion += 0.2
    if session_hits > 0:
        compassion += 0.2
    if intent.get("scope") == "session":
        compassion += 0.1
    compassion = max(0.0, min(1.0, compassion))

    # --- COURAGE: honesty about uncertainty ---
    courage = 0.6
    if said_dont_know:
        courage = 0.9
    elif grounding_score is not None and grounding_score < 0.3 and source_count == 0:
        courage = 0.3
    elif source_count >= 3:
        courage = 0.7
    courage = max(0.0, min(1.0, courage))

    # --- CREATIVITY: novel connections ---
    creativity = 0.5
    if procedure_hits > 0:
        creativity += 0.2
    if doc_hits > 0 and session_hits > 0:
        creativity += 0.15
    if mode == "explore":
        creativity += 0.15
    creativity = max(0.0, min(1.0, creativity))

    # --- CONNECTEDNESS: continuity with user ---
    connectedness = 0.3
    if session_hits > 0:
        connectedness += 0.2
    if fact_hits > 0:
        connectedness += 0.2
    if session_hits >= 3:
        connectedness += 0.15
    if procedure_hits > 0:
        connectedness += 0.1
    connectedness = max(0.0, min(1.0, connectedness))

    # --- CONFIDENCE: source-backed certainty ---
    if grounding_score is not None:
        confidence = grounding_score * 0.6 + min(1.0, source_count * 0.15) * 0.4
    elif source_count >= 3:
        confidence = 0.7
    elif source_count > 0:
        confidence = 0.5
    elif tool == "answer":
        confidence = 0.6
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
