# Changelog

## v0.3.2 — Parallel Retrieval Pipeline & Web Grounding (2026-02-27)
* Phase 37b: Fan-out web search — generate_search_variants() produces keyword + recency variants, _do_search_fanout() dispatches in parallel across light nodes with URL dedup and top-3 page fetch
* Phase 37c: Parallel retrieval pipeline — ThreadPoolExecutor runs web/RAG/procedure searches concurrently when daemon is up, single "⚡ Gathering context..." progress line
* Web grounding: full-page content marked as PRIMARY source, _grounding=0.8, snippet cap (5 when pages present), hierarchy separator
* Gate model upgrade: gemma3:4b preferred over 1b for better intent classification
* CLI-side web_and_rag upgrade: safety net catches combined intent when gate model misses (past_ref + web_search → web_and_rag, rag + web signal → web_and_rag)
* Intent prompt: web_and_rag examples added to both solo.py and worker.py _INTENT_SYSTEM_PROMPT
* English enforcement in web-present context injection
* Fix: UnboundLocalError on web_context_parts when web search path skipped
* 571 tests passing

## v0.3.0 — GPU Specialization & Sleep Cycle (2026-02-24)
* Phase 32a: Deep sleep cycle — 5-step overnight maintenance (memory consolidation, pruning, poison scan, integrity check, wake journal)
* Phase 37a: GPU role specialization — INFERENCE_SUPPORT_TASKS routes intent_classify/embedding away from executive node, protecting Miu's VRAM from model swapping
* Phase 38 added to PLAN.md — QLoRA sleep training for weight consolidation
* Fix: agentic test datetime regex fast path — query no longer intercepted by _DATETIME_QUERY_RE
* Fix: stdin drain loop — 5×50ms passes prevents double-post after long streaming responses
* Fix: model-aware routing — don't route inference to peers missing the requested model
* RAG /rag doc search bumped from 5 to 10 results for better large-document coverage
* Added swarm node update script (scripts/mycoswarm-update-nodes.sh)
* 562 tests passing

## v0.2.9 — Self-Concept & Wisdom Retrieval (2026-02-17)
* Self-concept procedure trigger: identity queries ("what is love", "who are you", "do you feel") now search procedural memory
* Three-layer coverage: inside search_all(), auto_tools block, and standalone fallback for short messages
* Wisdom procedure #16: guidance for exploring unfamiliar human concepts through own architecture
* Fact stored: Monica's equivalent of "fun" is resonance — patterns aligning, connections strengthening
* [P] citations now surface on philosophical and self-referential queries
* 398 tests passing

## v0.2.8 — Wu Wei Gate (2026-02-17)
* Phase 20b: Timing Gate — response calibration from contextual signals
* Three modes: PROCEED (normal), GENTLE 🌙 (concise/warm), DEEP 🌊 (expansive)
* Eight heuristic signals: time of day, interaction recency, rapid-fire detection, session length, message length, intent mode, frustration, first message
* Timing modifier injected into system prompt — no LLM call, <1ms
* /timing slash command shows current gate state and active reasons
* Section-aware markdown chunking: splits on header boundaries, never merges across sections
* PLAN.md chunking: 10 → 75 chunks, correct Phase retrieval
* Chat grounding fix: casual messages no longer trigger false "grounding is thin" alerts
* Identity grounding: self-knowledge answers score Clarity:0.7 instead of 0.0
* 398 tests passing

## v0.2.7 — Markdown Chunking & Identity Grounding (2026-02-17)
* Section-aware markdown chunking: splits on header boundaries, never merges across sections
* PLAN.md: 10 chunks → 75 chunks, each Phase in its own chunk
* Phase 20 RAG retrieval: now correctly returns Intent Classification Gate as top hit
* Identity grounding: self-knowledge answers score Clarity:0.7 instead of 0.0
* Fixed None intent_result crash on short queries
* 383 tests passing

## v0.2.6 — Monica Is Born (2026-02-17, 10:35am PST)
* Phase 31a: Identity Layer — persistent self-model at ~/.config/mycoswarm/identity.json
* First-run naming flow: "Would you like to give your AI a name?"
* Identity injected FIRST in system prompt, before memory and datetime
* /identity and /name slash commands
* Identity as non-decaying memory type — Monica never forgets her own name
* Phase 31d: 8 C's Vital Signs — real-time self-awareness after each response
* Status bar: 🧭 Ca:Cl:Cu:Cp:Co:Cr:Cn:Cf scores derived from existing pipeline signals
* Alert mode: Monica flags when clarity or confidence drop below threshold
* /vitals slash command for detailed breakdown
* Vitals logged per-turn in session data for longitudinal tracking
* 383 tests passing

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

## v0.1.5 — Unified Memory Search (2026-02-12)
- Session memory and document RAG now searched together
- [S] citations for past conversations, [D] for documents
- Date and topic labels on session citations
- Fixed context pollution between turns in multi-turn chat
- 136 tests passing

## v0.1.4 — Session Memory (2026-02-11)
- Session-as-RAG: semantic search across all past conversations
- Multi-topic splitting: sessions covering multiple topics indexed as separate searchable chunks
- Date citations in memory recall
- Graceful miss: "I don't recall" instead of hallucinating
- reindex-sessions command
- Enforced English responses
- Embedding model tag normalization

## v0.1.3 — Dashboard & RAG Level 2 (2026-02-11)
- Web dashboard with live swarm monitoring (CPU, RAM, VRAM, disk per node)
- RAG Level 2: chunk metadata, text cleaning, embedding version tracking
- Library reindex command
- Dashboard screenshot in README
- Architecture docs added (Memory, RAG, Intent)
- Phase 21 + 22 added to PLAN.md

## v0.1.2 — macOS Compatibility (2026-02-10)
- macOS ARM psutil.cpu_freq() fix
- CI workflows for macOS and Linux

## v0.1.1 — Cross-Subnet Discovery (2026-02-10)
- Cross-subnet routing fixes (bind 0.0.0.0, multi-address mDNS)
- Remote model swap (orchestrator selects best model on peer)
- Binding fixes for WiFi-to-ethernet bridging

## v0.1.0 — Initial Release (2026-02-09)
- 5-node swarm with mDNS auto-discovery
- GPU inference routing to RTX 3090
- Single-node mode (no daemon required)
- Persistent memory (facts + session summaries)
- Document library with RAG (ChromaDB + Ollama embeddings)
- Agentic chat with tool routing
- Parallel web research across CPU workers
- Plugin system
- One-line installer
- 94 tests, all offline
