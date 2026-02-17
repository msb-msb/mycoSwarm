# Fix CHANGELOG.md — Backfill All Missing Releases

## Context

CHANGELOG.md is stuck at v0.1.5. We need to add every release from v0.1.6 through the current version. Below are the reconstructed release notes from our dev sessions.

## Step 1: Check current state

```bash
cat CHANGELOG.md | head -20
git tag -l "v*" --sort=-v:refname
grep "version" pyproject.toml | head -1
```

## Step 2: Add these entries to CHANGELOG.md

Insert these ABOVE the existing v0.1.5 entry, newest first. Use the exact format already in the file. Keep the `*` bullet style matching the v0.1.5 entry.

---

## v0.2.4 — Wisdom Layer & Procedure Growth (2026-02-16)
* Procedure growth from experience — LLM extracts candidates from session lessons, human review via `/procedure review`
* Candidate quality gates: Jaccard dedup (0.6 threshold), max 3 per session, auto-expire after 14 days
* Ethical reasoning domain: 9 hand-curated wisdom procedures from Wu Wei, IFS, and Tai Chi
* Independent procedural retrieval: wisdom procedures surface on problem-pattern queries regardless of intent
* Fact attribution fix: model says "you teach Tai Chi" not "I teach Tai Chi"
* Session relevance filtering: RRF minimum threshold + word-overlap gate prevents irrelevant session bleed
* Regex inflection matching for broader problem pattern detection
* Conciseness prompt tuning
* 357 tests passing

## v0.2.3 — Procedure Growth from Experience (2026-02-16)
* LLM-powered procedure extraction from session lessons at end of session
* Candidates stored with status=candidate, NOT indexed until human review
* `/procedure review` interactive flow: approve, reject, skip, quit
* Improved lesson prompts capture principles not just actions
* 349 tests passing

## v0.2.2 — Chart Tool v3 + Procedural Retrieval Fix (2026-02-15)
* Chart tool v3: Graphviz engine for flow diagrams (proper layout, arrows connect to box edges)
* Matplotlib engine unchanged for bar, line, table, before/after charts
* InsiderLLM dark theme with brand colors and watermark
* Optional dependencies: `pip install mycoswarm[charts]` (matplotlib + graphviz)
* Procedural retrieval regex expanded: "ignored", "broken", "crash", "stuck", "slow", "missing", "unexpected", "weird" now trigger procedure search
* 9 Phase 20 debugging exemplars seeded as procedural memory
* 5 lessons promoted from episodic to procedural memory
* 337 tests passing

## v0.2.1 — Procedural Memory (2026-02-15)
* Phase 21d: Procedural memory store (procedures.jsonl + ChromaDB procedural_memory collection)
* `/procedure` CLI: list, add, remove, promote lesson to procedure
* `search_all()` returns 3-tuple with `[P1]`/`[P2]` procedure citations
* Intent-triggered retrieval on execute mode and problem-like queries
* Bridge between rich episodic lessons (29a) and reusable procedural knowledge
* 337 tests passing

## v0.2.0 — Cognitive Architecture Foundations (2026-02-15)
* Phase 29a: Rich Episodic Memory — sessions capture decisions, lessons, surprises, emotional tone
* Lessons indexed separately in ChromaDB for procedural retrieval
* "Reflecting on session..." replaces "Summarizing session..." on exit
* Phase 21a: Fact Lifecycle Tags — types (preference, fact, project, ephemeral), reference tracking, `/stale` command
* Phase 21b: Decay Scoring — exponential half-life (30 days), lessons decay slower (60 days)
* Phase 29b: Reflection prompt fix — subject-matter lessons, not self-referential observations
* ARCHITECTURE-COGNITIVE.md — IFS + CoALA framework documentation
* 22 smoke test checks passing across 5 test scripts
* All 5 swarm nodes updated
* 311 tests passing

## v0.1.9 — PDF Intelligence & Smoke Tests (2026-02-15)
* Phase 21g Step 3: Contradiction detection — pattern-matching drops session summaries that contradict documents
* PDF TOC/bookmark extraction via pymupdf for section headers
* Paragraph-aware chunking: splits on boundaries, not fixed character counts
* Heuristic heading fallback for PDFs without TOC
* Wu Wei book test: 168 chunks, 43 sections, 0 untitled, 38% mid-sentence (was 83%)
* Smoke test suite: 5 scripts (RAG grounding, poison resistance, memory priority, intent classification, swarm distribution)
* Book stress test for large-scale RAG validation
* 270 tests passing

## v0.1.8 — Worker Parity & Self-Correcting Memory (2026-02-14)
* Docs scope override (web_and_rag → rag) added to worker.py, matching solo.py behavior
* All peers now produce identical intent classification results
* Phase 21g Steps 1-2: Source priority tagging (user_document vs model_generated)
* 2x RRF boost for user_document hits when scope=all
* Confidence gating: grounding_score (0-1) computed before saving session summaries
* Low grounding (<0.3) entries excluded from retrieval index
* 251 tests passing

## v0.1.7 — RAG Grounding Pipeline (2026-02-14)
* Source filtering: named-file queries (e.g. "PLAN.md") return only chunks from that file
* Section header boost: +0.05 RRF for word-boundary section matches
* User-message injection: RAG context merged into user message instead of separate system message — fixes local model grounding
* `--debug` flag: full pipeline visibility (intent → retrieval → hits → prompt → messages)
* Debug sessions skip summarization to prevent hallucination feedback loops
* Intent override: docs scope downgrades web_and_rag → rag
* 247 tests passing

## v0.1.6 — Intent Classification Gate (2026-02-13)
* Intent classifier: classifies queries into {tool, mode, scope} before main inference
* Three-field schema: tool (answer/web_search/rag/web_and_rag), mode (recall/explore/execute/chat), scope (session/docs/facts/all)
* Gate model: gemma3:1b for fast classification on CPU nodes (~0.3s)
* Distributed gate: intent_classify dispatched across swarm via daemon to CPU workers
* Scope-driven retrieval: search_all() accepts intent dict — mode/scope adjusts candidates
* Embedding model exclusion: prevents nomic-embed-text from being selected for classification/reranking
* Chat loop unified: single search_all(intent=) call replaces separate search() + search_sessions()
* 232 tests passing

---

## Step 3: Add v0.2.5 entry

Add this ABOVE v0.2.4:

## v0.2.5 — Wisdom Layer & Procedure Growth (2026-02-16)
* Phase 21d complete: Procedure growth from experience — LLM extracts candidates from session lessons, human review via `/procedure review`
* Candidate quality gates: Jaccard dedup (0.6), stricter extraction prompt, max 3 per session, auto-expire after 14 days
* Ethical reasoning domain: 9 hand-curated wisdom procedures from Wu Wei, IFS, and Tai Chi shape how the system reasons
* Independent procedural retrieval: wisdom procedures surface on problem-pattern queries regardless of intent classification
* Fact attribution fix: model says "you teach Tai Chi" not "I teach Tai Chi"
* Session relevance filtering: RRF minimum threshold + word-overlap gate prevents irrelevant session bleed
* Conciseness prompt tuning
* Regex inflection matching for broader problem pattern detection
* 357 tests passing

Then check if there are any releases AFTER v0.2.5:

```bash
git tag -l "v*" --sort=-v:refname | head -5
grep "version" pyproject.toml | head -1
```

If there are any later releases, reconstruct their notes from `git log` and add entries for those too.

## Step 4: Update CLAUDE.md release checklist

Find the "Release Checklist" section and add after step 1 (bump version):

```
2. Update CHANGELOG.md with new version entry
```

Renumber remaining steps (old step 2 becomes 3, etc.)

## Step 5: Commit and push

```bash
git add CHANGELOG.md CLAUDE.md
git commit -m "docs: backfill CHANGELOG.md v0.1.6 through current, add changelog to release checklist"
git push
```
