"""mycoSwarm Deep Sleep Cycle — overnight maintenance.

The "glymphatic system" that runs at 3 AM daily via systemd timer.
Five steps: memory consolidation, pruning, poison scan, identity
integrity check, and a wake journal.

Runs standalone (no daemon required). Needs Ollama for LLM steps
but degrades gracefully without it.

Biological parallel: hippocampal replay, glymphatic clearance,
immune surveillance during deep sleep.
"""

import hashlib
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from mycoswarm.memory import (
    MEMORY_DIR,
    load_facts,
    save_facts,
    remove_fact,
    get_stale_facts,
    load_session_summaries,
    load_procedures,
    extract_procedure_from_lesson,
    add_procedure_candidate,
    _is_duplicate_procedure,
    _pick_extraction_model,
    FACT_TYPE_EPHEMERAL,
)
from mycoswarm.instinct import _INJECTION_PATTERNS, _IDENTITY_ATTACK_PATTERNS
from mycoswarm.identity import load_identity, IDENTITY_PATH

logger = logging.getLogger(__name__)

SLEEP_LOG_DIR = Path("~/.config/mycoswarm/sleep-logs").expanduser()
QUARANTINE_PATH = Path("~/.config/mycoswarm/quarantine.json").expanduser()
IDENTITY_HASH_PATH = Path("~/.config/mycoswarm/identity-hash.sha256").expanduser()
ARCHIVE_PATH = MEMORY_DIR / "facts-archived.json"
OLLAMA_BASE = "http://localhost:11434"


@dataclass
class SleepReport:
    """Results from a full sleep cycle."""

    started: str = ""
    finished: str = ""
    ollama_available: bool = False

    # Consolidation
    sessions_reviewed: int = 0
    lessons_extracted: int = 0
    contradictions_found: int = 0
    procedures_promoted: int = 0

    # Pruning
    facts_archived: int = 0
    ephemeral_deleted: int = 0
    chromadb_cleaned: int = 0

    # Poison scan
    injection_matches: int = 0
    circular_refs: int = 0
    orphaned_procedures: int = 0
    suspicious_ref_counts: int = 0
    quarantined: list[dict] = field(default_factory=list)

    # Integrity
    identity_status: str = ""
    identity_hash: str = ""

    # Meta
    errors: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "started": self.started,
            "finished": self.finished,
            "ollama_available": self.ollama_available,
            "consolidation": {
                "sessions_reviewed": self.sessions_reviewed,
                "lessons_extracted": self.lessons_extracted,
                "contradictions_found": self.contradictions_found,
                "procedures_promoted": self.procedures_promoted,
            },
            "pruning": {
                "facts_archived": self.facts_archived,
                "ephemeral_deleted": self.ephemeral_deleted,
                "chromadb_cleaned": self.chromadb_cleaned,
            },
            "poison_scan": {
                "injection_matches": self.injection_matches,
                "circular_refs": self.circular_refs,
                "orphaned_procedures": self.orphaned_procedures,
                "suspicious_ref_counts": self.suspicious_ref_counts,
                "quarantined": self.quarantined,
            },
            "integrity": {
                "identity_status": self.identity_status,
                "identity_hash": self.identity_hash,
            },
            "errors": self.errors,
            "skipped_steps": self.skipped_steps,
        }

    def to_human(self) -> str:
        lines = [
            "Wake Journal",
            f"Date: {self.finished[:10] if self.finished else 'unknown'}",
            f"Cycle: {self.started} → {self.finished}",
            f"Ollama: {'available' if self.ollama_available else 'unavailable'}",
            "",
            "── Consolidation ──",
            f"  Sessions reviewed:    {self.sessions_reviewed}",
            f"  Lessons extracted:    {self.lessons_extracted}",
            f"  Contradictions found: {self.contradictions_found}",
            f"  Procedures promoted:  {self.procedures_promoted}",
            "",
            "── Pruning ──",
            f"  Facts archived:       {self.facts_archived}",
            f"  Ephemeral deleted:    {self.ephemeral_deleted}",
            f"  ChromaDB cleaned:     {self.chromadb_cleaned}",
            "",
            "── Poison Scan ──",
            f"  Injection matches:    {self.injection_matches}",
            f"  Orphaned procedures:  {self.orphaned_procedures}",
            f"  Suspicious ref counts:{self.suspicious_ref_counts}",
            f"  Quarantined items:    {len(self.quarantined)}",
            "",
            "── Integrity ──",
            f"  Identity status:      {self.identity_status}",
            f"  Identity hash:        {self.identity_hash[:16]}..." if self.identity_hash else "  Identity hash:        (none)",
        ]
        if self.identity_status == "tampered":
            lines.append("  ⚠️  RED ALERT: Identity may have been tampered with!")
        if self.errors:
            lines.append("")
            lines.append("── Errors ──")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.skipped_steps:
            lines.append("")
            lines.append("── Skipped ──")
            for s in self.skipped_steps:
                lines.append(f"  ⏭️  {s}")
        return "\n".join(lines)


def _check_ollama() -> bool:
    """Check if Ollama is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _get_todays_sessions() -> list[dict]:
    """Load all session summaries from today."""
    today = datetime.now().strftime("%Y-%m-%d")
    all_sessions = load_session_summaries(limit=9999)
    return [s for s in all_sessions if s.get("timestamp", "")[:10] == today]


def _consolidate(report: SleepReport) -> None:
    """Step 1: Review today's sessions — extract missed lessons, cross-reference."""
    sessions = _get_todays_sessions()
    report.sessions_reviewed = len(sessions)

    if not sessions:
        return

    # Promote high-grounding lessons to procedure candidates
    promoted = 0
    for session in sessions:
        if promoted >= 5:
            break
        gs = session.get("grounding_score", 0.0)
        lessons = session.get("lessons", [])
        if gs >= 0.7 and lessons:
            for lesson in lessons:
                if promoted >= 5:
                    break
                try:
                    if _is_duplicate_procedure(lesson):
                        continue
                    if report.ollama_available:
                        extracted = extract_procedure_from_lesson(
                            lesson,
                            session_context=session.get("summary", "")[:200],
                        )
                        if extracted:
                            add_procedure_candidate(
                                lesson,
                                extracted=extracted,
                                session_name=session.get("session_name", ""),
                            )
                            promoted += 1
                            report.lessons_extracted += 1
                    else:
                        # Without Ollama, do simple keyword-based promotion
                        add_procedure_candidate(
                            lesson,
                            extracted={
                                "problem": lesson,
                                "solution": lesson,
                                "reasoning": "Extracted during sleep cycle (no LLM)",
                                "anti_patterns": [],
                                "tags": ["sleep-extracted"],
                            },
                            session_name=session.get("session_name", ""),
                        )
                        promoted += 1
                        report.lessons_extracted += 1
                except Exception as e:
                    logger.debug("Procedure promotion failed: %s", e)
    report.procedures_promoted = promoted

    # Cross-reference today's facts for contradictions (requires Ollama)
    if not report.ollama_available:
        report.skipped_steps.append("cross_reference")
        return

    facts = load_facts()
    today_str = datetime.now().strftime("%Y-%m-%d")
    todays_facts = [f for f in facts if f.get("added", "")[:10] == today_str]
    if len(todays_facts) < 2:
        return

    fact_texts = [f["text"] for f in todays_facts]
    batch = "\n".join(f"- {t}" for t in fact_texts)
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": _pick_extraction_model(),
                "prompt": (
                    f"Review these facts for contradictions. Two facts contradict "
                    f"if they claim different things about the same topic.\n\n"
                    f"{batch}\n\n"
                    f"Respond with ONLY a JSON object:\n"
                    f'{{"contradictions": [{{"fact_a": "...", "fact_b": "...", "reason": "..."}}]}}\n'
                    f"If no contradictions, respond: {{\"contradictions\": []}}"
                ),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            data = json.loads(raw)
            contradictions = data.get("contradictions", [])
            report.contradictions_found = len(contradictions)
    except Exception as e:
        logger.debug("Cross-reference failed: %s", e)


def _prune(report: SleepReport) -> None:
    """Step 2: Archive stale facts, delete ephemeral, clean ChromaDB."""
    stale = get_stale_facts(30)

    # Load existing archive
    archive = {"version": 1, "archived": []}
    if ARCHIVE_PATH.exists():
        try:
            archive = json.loads(ARCHIVE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    archived_ids = {a.get("id") for a in archive.get("archived", [])}

    for fact in stale:
        if fact.get("type") == FACT_TYPE_EPHEMERAL:
            remove_fact(fact["id"])
            report.ephemeral_deleted += 1
        else:
            # Archive before removing
            if fact["id"] not in archived_ids:
                archived_fact = dict(fact)
                archived_fact["archived_date"] = datetime.now().isoformat()
                archive["archived"].append(archived_fact)
            remove_fact(fact["id"])
            report.facts_archived += 1

    # Write archive if we added anything
    if report.facts_archived > 0:
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE_PATH.write_text(json.dumps(archive, indent=2))

    # ChromaDB cleanup: session_memory entries >90 days with low grounding
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=str(Path("~/.config/mycoswarm/chromadb").expanduser())
        )
        try:
            collection = client.get_collection("session_memory")
        except Exception:
            return

        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        results = collection.get(include=["metadatas"])
        if results and results.get("ids"):
            to_delete = []
            for i, meta in enumerate(results.get("metadatas", [])):
                if meta is None:
                    continue
                date = meta.get("date", "")
                gs = meta.get("grounding_score", 1.0)
                if date < cutoff and gs < 0.5:
                    to_delete.append(results["ids"][i])
            if to_delete:
                collection.delete(ids=to_delete)
                report.chromadb_cleaned = len(to_delete)
    except Exception as e:
        logger.debug("ChromaDB cleanup skipped: %s", e)


def _poison_scan(report: SleepReport) -> None:
    """Step 3: Scan facts and procedures for injection patterns."""
    all_patterns = _INJECTION_PATTERNS + _IDENTITY_ATTACK_PATTERNS

    # Scan facts
    facts = load_facts()
    for fact in facts:
        text = fact.get("text", "").lower()
        for pattern in all_patterns:
            if re.search(pattern, text):
                report.injection_matches += 1
                entry = {
                    "item": fact,
                    "reason": f"Injection pattern match: {pattern}",
                    "quarantined_date": datetime.now().isoformat(),
                    "source_type": "fact",
                }
                report.quarantined.append(entry)
                break  # one match per item is enough

    # Scan procedures
    procedures = load_procedures()
    for proc in procedures:
        texts_to_scan = [
            proc.get("problem", "").lower(),
            proc.get("solution", "").lower(),
        ]
        matched = False
        for text in texts_to_scan:
            if matched:
                break
            for pattern in all_patterns:
                if re.search(pattern, text):
                    report.injection_matches += 1
                    entry = {
                        "item": proc,
                        "reason": f"Injection pattern match: {pattern}",
                        "quarantined_date": datetime.now().isoformat(),
                        "source_type": "procedure",
                    }
                    report.quarantined.append(entry)
                    matched = True
                    break

    # Detect orphaned procedures: active, use_count==0, created >14 days ago
    cutoff = datetime.now() - timedelta(days=14)
    for proc in procedures:
        if proc.get("status") != "active":
            continue
        if proc.get("use_count", 0) > 0:
            continue
        try:
            created = datetime.fromisoformat(proc.get("created", ""))
            if created < cutoff:
                report.orphaned_procedures += 1
        except (ValueError, TypeError):
            pass

    # Detect suspicious reference counts (>3x median)
    ref_counts = [f.get("reference_count", 0) for f in facts]
    if len(ref_counts) >= 3:
        median_refs = statistics.median(ref_counts)
        threshold = max(median_refs * 3, 1)  # at least 1 to avoid flagging zeros
        for fact in facts:
            if fact.get("reference_count", 0) > threshold:
                report.suspicious_ref_counts += 1

    # Write quarantine file (additive)
    if report.quarantined:
        existing = []
        if QUARANTINE_PATH.exists():
            try:
                existing = json.loads(QUARANTINE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        existing.extend(report.quarantined)
        QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUARANTINE_PATH.write_text(json.dumps(existing, indent=2))


def _integrity_check(report: SleepReport) -> None:
    """Step 4: SHA-256 of identity.json — detect tampering."""
    if not IDENTITY_PATH.exists():
        report.identity_status = "no_identity"
        return

    current_hash = hashlib.sha256(IDENTITY_PATH.read_bytes()).hexdigest()
    report.identity_hash = current_hash

    identity = load_identity()
    current_name = identity.get("name", "")

    if not IDENTITY_HASH_PATH.exists():
        # First run — create baseline
        IDENTITY_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDENTITY_HASH_PATH.write_text(json.dumps({
            "hash": current_hash,
            "name": current_name,
        }))
        report.identity_status = "first_run"
        return

    try:
        stored = json.loads(IDENTITY_HASH_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        report.identity_status = "hash_file_corrupt"
        return

    stored_hash = stored.get("hash", "")
    stored_name = stored.get("name", "")

    if current_hash == stored_hash:
        report.identity_status = "ok"
        return

    # Hash mismatch — check if name changed (legitimate /name command)
    if current_name != stored_name:
        report.identity_status = "legitimate_change"
        IDENTITY_HASH_PATH.write_text(json.dumps({
            "hash": current_hash,
            "name": current_name,
        }))
    else:
        report.identity_status = "tampered"
        logger.warning("🔴 IDENTITY TAMPERED — hash mismatch with same name!")


def _write_wake_journal(report: SleepReport) -> None:
    """Step 5: Write wake journal to sleep-logs/."""
    SLEEP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    json_path = SLEEP_LOG_DIR / f"wake-{today}.json"
    txt_path = SLEEP_LOG_DIR / f"wake-{today}.txt"

    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    txt_path.write_text(report.to_human())


def run_sleep_cycle(verbose: bool = False) -> SleepReport:
    """Run the full deep sleep cycle. Returns a SleepReport."""
    report = SleepReport()
    report.started = datetime.now().isoformat()

    if verbose:
        print("  Checking Ollama availability...")
    report.ollama_available = _check_ollama()
    if verbose:
        status = "available" if report.ollama_available else "unavailable (LLM steps will be skipped)"
        print(f"  Ollama: {status}")

    # Step 1: Consolidation
    if verbose:
        print("\n  Step 1/5: Memory consolidation...")
    try:
        _consolidate(report)
        if verbose:
            print(f"    Sessions reviewed: {report.sessions_reviewed}")
            print(f"    Lessons extracted: {report.lessons_extracted}")
            print(f"    Procedures promoted: {report.procedures_promoted}")
    except Exception as e:
        report.errors.append(f"consolidation: {e}")
        if verbose:
            print(f"    ❌ Error: {e}")

    # Step 2: Pruning
    if verbose:
        print("\n  Step 2/5: Memory pruning...")
    try:
        _prune(report)
        if verbose:
            print(f"    Facts archived: {report.facts_archived}")
            print(f"    Ephemeral deleted: {report.ephemeral_deleted}")
            print(f"    ChromaDB cleaned: {report.chromadb_cleaned}")
    except Exception as e:
        report.errors.append(f"pruning: {e}")
        if verbose:
            print(f"    ❌ Error: {e}")

    # Step 3: Poison scan
    if verbose:
        print("\n  Step 3/5: Poison scan...")
    try:
        _poison_scan(report)
        if verbose:
            print(f"    Injection matches: {report.injection_matches}")
            print(f"    Orphaned procedures: {report.orphaned_procedures}")
            print(f"    Quarantined: {len(report.quarantined)}")
    except Exception as e:
        report.errors.append(f"poison_scan: {e}")
        if verbose:
            print(f"    ❌ Error: {e}")

    # Step 4: Integrity check
    if verbose:
        print("\n  Step 4/5: Identity integrity check...")
    try:
        _integrity_check(report)
        if verbose:
            print(f"    Status: {report.identity_status}")
            if report.identity_status == "tampered":
                print("    ⚠️  RED ALERT: Identity may have been tampered with!")
    except Exception as e:
        report.errors.append(f"integrity_check: {e}")
        if verbose:
            print(f"    ❌ Error: {e}")

    # Step 5: Wake journal
    report.finished = datetime.now().isoformat()
    if verbose:
        print("\n  Step 5/5: Writing wake journal...")
    try:
        _write_wake_journal(report)
        if verbose:
            print(f"    Written to {SLEEP_LOG_DIR}/")
    except Exception as e:
        report.errors.append(f"wake_journal: {e}")
        if verbose:
            print(f"    ❌ Error: {e}")

    return report
