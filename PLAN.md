# mycoSwarm Development Plan

## Done

### Phase 1: Foundation (2025-02-08)
- [x] GitHub repo created with README, SECURITY.md, LICENSE (MIT)
- [x] Project structure: pyproject.toml, src/mycoswarm/ package
- [x] Hardware detection (hardware.py): GPU via nvidia-smi, CPU, RAM, disk, network, Ollama models
- [x] Capability classification (capabilities.py): GPU tiers, node tiers, model recommendations
- [x] Node identity (node.py): persistent UUID, broadcast payload, JSON serialization
- [x] CLI: `mycoswarm detect` with human-readable and --json output

### Phase 2: Discovery & Networking (2025-02-08)
- [x] mDNS discovery (discovery.py): announce + listen via zeroconf
- [x] PeerRegistry: thread-safe peer tracking with callbacks
- [x] Node daemon (daemon.py): detection + discovery + periodic status refresh
- [x] CLI: `mycoswarm daemon` with --port and -v flags
- [x] Live test: Miu (3090 executive) ↔ naru (M710Q light) auto-discovered in <1s

### Phase 3: API Layer (2025-02-08)
- [x] FastAPI service (api.py): /health, /status, /identity, /peers, /task, /task/{id}
- [x] TaskQueue: async queue with submit, result storage
- [x] Daemon updated: runs uvicorn alongside discovery
- [x] Cross-node HTTP confirmed: Miu ↔ naru bidirectional API calls
- [x] CLI: `mycoswarm swarm` (swarm overview), `mycoswarm ping` (peer latency)

### Phase 4: Orchestrator & Worker (2025-02-08)
- [x] Orchestrator (orchestrator.py): task routing by capability, scoring, dispatch
- [x] Task worker (worker.py): queue consumer, handler registry, concurrency control
- [x] Inference handler: calls Ollama /api/generate, returns response + metrics
- [x] Ping handler: simple echo for testing
- [x] Daemon updated: runs worker alongside API + discovery
- [x] CLI: `mycoswarm ask "prompt"` — submit inference to swarm

### Phase 5: Cross-Node Inference (2026-02-08)
- [x] Test `mycoswarm ask` end-to-end on Miu (local inference via worker) — works, gemma3:27b @ 34 tok/s
- [x] Add orchestrator to daemon (Orchestrator instance created in daemon, passed to API)
- [x] Wire up /task endpoint to route to best node when local can't handle it
- [x] Orchestrator dispatches to peer and polls for result (replaces fire-and-forget POST)
- [x] CLI discovers models from peers when local node has none (for naru → Miu flow)
- [x] /peers endpoint now includes available_models for model discovery
- [x] First cross-node inference: naru → Miu, gemma3:27b

### Phase 6: Robustness (2026-02-08)
- [x] Handle Ollama not running gracefully — strip gpu_inference/cpu_inference caps when Ollama unreachable
- [x] Add `allow_name_change=True` to zeroconf registration (prevent NonUniqueNameException on restart)
- [x] Task timeout enforcement in worker — asyncio.wait_for cancels tasks exceeding timeout_seconds
- [x] Retry failed dispatches — orchestrator tries up to 3 peers in score order, only retries on dispatch errors
- [x] Peer health monitoring — PeerRegistry tracks consecutive failures per peer, marks unhealthy after 3, resets on success or mDNS update
- [x] Graceful degradation — _route_remote wrapped in try/except, orchestrator errors fail the task gracefully; local tasks always work independently

### Phase 9a: Chat Mode (2026-02-08)
- [x] Chat mode: `mycoswarm chat` — interactive REPL with conversation history
- [x] Worker handle_inference supports messages array (Ollama /api/chat endpoint)
- [x] Extracted CLI helpers: _discover_model, _submit_and_poll, _list_swarm_models
- [x] Slash commands: /model (list/switch), /peers, /clear, /quit
- [x] Multi-turn memory verified: model remembers context across turns
- [x] Shows node_id, tokens/sec, duration in response footer

### Phase 9b: Streaming Inference (2026-02-08)
- [x] TaskQueue stream management: create_stream/get_stream/remove_stream per task
- [x] Worker streaming: _inference_stream uses httpx streaming + Ollama NDJSON, pushes tokens to asyncio.Queue
- [x] Worker refactored: _build_ollama_request, _metrics_from_ollama, _inference_batch, _inference_stream
- [x] SSE endpoint: GET /task/{id}/stream — async generator yields tokens from queue
- [x] Edge cases: replay if worker finishes before SSE connects, timeout, error propagation, sentinel cleanup
- [x] Remote tasks: _route_remote pushes full response as single-shot SSE events
- [x] CLI _stream_response: sync httpx stream consumer, prints tokens as they arrive
- [x] cmd_chat uses streaming: POST /task → SSE stream → live token display
- [x] cmd_ask unchanged (still uses _submit_and_poll)

### Phase 9c: Web Search & Fetch (2026-02-08)
- [x] Added ddgs dependency (no API key, fits manifesto)
- [x] handle_web_search: DuckDuckGo via DDGS + asyncio.to_thread, returns title/url/snippet
- [x] handle_web_fetch: httpx GET with HTML stripping, follow_redirects, max_length
- [x] _strip_html helper: removes script/style, strips tags, decodes entities
- [x] Registered web_search/web_fetch in worker HANDLERS
- [x] Orchestrator already routes web_search/web_fetch → cpu_worker nodes
- [x] CLI: `mycoswarm search "query"` with -n/--max-results flag
- [x] Enables CPU-only nodes (M710Qs) to contribute web skills to the swarm

### Phase 9d: Research Command with Parallel Planner (2026-02-08)
- [x] `mycoswarm research "query"` — compound command: CPU workers search, GPU nodes think
- [x] Planning step: GPU node decomposes query into 2-4 specific search queries (JSON response)
- [x] Parallel search dispatch: all queries sent concurrently via ThreadPoolExecutor
- [x] Searches distributed across CPU workers via orchestrator (not always local)
- [x] Load-aware routing: orchestrator tracks inflight tasks per peer, least-loaded wins
- [x] Distributable task types (web_search, web_fetch) bypass local shortcut in API
- [x] CLI shows which node handled each search with timing
- [x] Result deduplication by URL across all search results
- [x] Synthesis: all results bundled into context block, streamed via SSE
- [x] Progress display: Planning → parallel Searching → Synthesizing
- [x] Sources cited in footer with numbered URLs matching [1], [2] refs in response
- [x] Fallback: if planning fails, falls back to single original query

### Phase 9e: Models Command & Systemd (2026-02-08)
- [x] `mycoswarm models` — unified view of every model across the swarm, grouped by model name with node/GPU info
- [x] Systemd service file: scripts/mycoswarm.service — auto-start daemon on boot, works on all nodes

### Phase 10a: Plugin System (2026-02-08)
- [x] Plugin loader (plugins.py): scans ~/.config/mycoswarm/plugins/ on daemon startup
- [x] plugin.yaml defines: name, task_type, description, capabilities
- [x] handler.py exports: async def handle(task) → TaskResult
- [x] Daemon auto-registers plugins into HANDLERS dict + node capabilities
- [x] Minimal YAML parser (no PyYAML dependency)
- [x] CLI: `mycoswarm plugins` — list installed plugins with status
- [x] Example plugin: plugins/example_summarize/ (sentence extraction)

### Phase 10b: Datetime Awareness & Persistent Sessions (2026-02-08)
- [x] Datetime awareness: _datetime_string() injects current date/time into all inference system prompts
- [x] _build_ollama_request() handles both chat (messages array) and generate (prompt) modes
- [x] Persistent chat sessions: saved to ~/.config/mycoswarm/sessions/ as JSON
- [x] Session flags: --resume (latest session), --session NAME, --list (show saved sessions)
- [x] Auto-save on /quit and Ctrl+C, /save slash command for manual save
- [x] Session metadata: name, model, created/updated timestamps, message count

### Phase 11: Skill Library — Core Handlers (2026-02-08)
- [x] Embedding handler: POST to Ollama /api/embeddings, returns vector + dimensions
- [x] File read handler: extract text from PDF (pymupdf), markdown, txt, html, csv, json
- [x] File summarize handler: read file inline then summarize via inference (routes to GPU nodes)
- [x] Translate handler: inference with translation system prompt, auto-detects model
- [x] Code run handler: sandboxed Python subprocess, network isolated via unshare -rn, temp dir, timeout
- [x] New capabilities: FILE_PROCESSING, CODE_EXECUTION advertised by all cpu_worker nodes
- [x] TASK_ROUTING updated: embedding/translate/file_summarize → inference nodes, file_read/code_run → CPU workers
- [x] file_read and code_run added to DISTRIBUTABLE_TASKS for swarm-wide distribution
- [x] pymupdf dependency added to pyproject.toml

### Phase 12: Single-Node Quick Start (2026-02-09)
- [x] solo.py: direct Ollama inference module (no daemon/queue/worker needed)
- [x] `mycoswarm chat` works without daemon — auto-detects Ollama, streams directly
- [x] `mycoswarm ask` works without daemon — auto-detects Ollama, streams directly
- [x] "Running in single-node mode" message when no daemon detected
- [x] Seamless upgrade: if daemon is running, uses full swarm pipeline automatically
- [x] README updated with two-command quick start: pip install mycoswarm / mycoswarm chat

### Phase 13: macOS Compatibility (2026-02-09)
- [x] hardware.py: Apple Silicon GPU detection via system_profiler (unified memory as VRAM)
- [x] hardware.py: macOS CPU detection via sysctl -n machdep.cpu.brand_string
- [x] hardware.py: AF_LINK support for MAC addresses on macOS (was AF_PACKET only)
- [x] hardware.py: macOS pseudo-mount filtering (/System/Volumes/, /private/var/vm)
- [x] worker.py: cross-platform strftime (removed glibc-only %-d and %-I specifiers)
- [x] solo.py: same strftime fix
- [x] worker.py: /opt/homebrew/bin added to sandbox PATH on macOS
- [x] worker.py: unshare gracefully skipped on macOS (falls back to unsandboxed subprocess)
- [x] README: macOS install docs (brew install ollama + pip install mycoswarm)

### Phase 14: One-Line Install Script (2026-02-09)
- [x] install.sh: detects OS (Linux/macOS), installs Python, Ollama, mycoswarm
- [x] Auto-pulls model sized for RAM: gemma3:1b (<8GB), gemma3:4b (8-16GB), gemma3:27b (16GB+)
- [x] Supports apt, dnf, pacman (Linux) and brew (macOS)
- [x] Starts Ollama if not running (systemd on Linux, background process on macOS)
- [x] Runs `mycoswarm detect` to show user what was found
- [x] README updated with one-line install: curl ... | bash

### Phase 15: Persistent Memory (2026-02-09)
- [x] memory.py: persistent fact store (facts.json) + session summaries (sessions.jsonl)
- [x] /remember, /memories, /forget slash commands in chat
- [x] Auto-summarize sessions on exit via Ollama
- [x] build_memory_system_prompt() injects facts + summaries into chat system prompt
- [x] Hallucination mitigation: capability boundary text prevents fabrication of real-time data
- [x] Session memory citation: model cites dates, gracefully handles misses, distinguishes facts vs session history (2026-02-11)
- [x] System prompt rewrite: explicitly trust retrieved [D1]/[S1] excerpts while keeping real-time data hallucination boundary (2026-02-13)

### Phase 16: Document Library with RAG (2026-02-09)
- [x] library.py: ChromaDB vector storage, document ingestion, chunking, embedding via Ollama
- [x] Text extraction reuses _extract_text() from worker.py (PDF, HTML, MD, TXT, CSV, JSON)
- [x] Word-based chunking with overlap (384 words ≈ 512 tokens)
- [x] CLI: `mycoswarm library ingest/search/list/remove`
- [x] CLI: `mycoswarm rag "question"` — standalone RAG with citations
- [x] Chat: /rag slash command for inline RAG queries
- [x] chromadb dependency added to pyproject.toml

### Phase 17: Agentic Chat (2026-02-09)
- [x] classify_query() in solo.py: LLM classifies each message as answer/web_search/rag/web_and_rag
- [x] web_search_solo() in solo.py: local DuckDuckGo search via ddgs (no API key)
- [x] Auto tool routing: web results tagged [W1],[W2], doc excerpts tagged [D1],[D2]
- [x] Transparent progress indicators: 🤔 classifying → 🔍 searching → 📚 checking docs
- [x] /auto slash command to toggle agentic mode (default ON)
- [x] Skip classification for slash commands and short messages (< 5 words)
- [x] Web-aware boundary swap: replaces "no internet" text when search results are injected

### Phase 18: Testing (2026-02-09)
- [x] Unit tests for hardware detection (mock subprocess/psutil) — 5 tests
- [x] Unit tests for capability classification — 10 tests
- [x] Unit tests for orchestrator routing logic — 5 tests
- [x] Unit tests for API endpoints — 3 tests
- [x] Unit tests for worker handlers — 8 tests
- [x] Unit tests for plugin system — 5 tests
- [x] Unit tests for document library/RAG — 23 tests → 25 tests (added reindex-sessions tests)
- [x] Unit tests for persistent memory — 21 tests → 30 tests (added topic splitting tests)
- [x] Unit tests for agentic chat classification — 9 tests
- [x] 129 tests passing at Phase 18 completion (now 529 as of v0.2.15)

### Phase 19: RAG Level 2 Improvements
- [x] Metadata on chunks: source filename, section heading, file date, document type
- [x] Text cleaning before chunking: strip headers/footers, normalize whitespace, remove boilerplate
- [x] Embedding model version tracking: store model name in metadata, warn on mismatch
- [x] Embedding model tag normalization: strip `:latest` suffix so "nomic-embed-text" matches "nomic-embed-text:latest" (2026-02-11)
- [x] Session-as-RAG: index session summaries into ChromaDB for semantic memory search
- [x] Per-topic session splitting: split multi-topic summaries into separate ChromaDB entries for better retrieval precision (2026-02-11)
- [x] Session reindex command: `mycoswarm library reindex-sessions` drops + rebuilds session_memory from sessions.jsonl with topic splitting (2026-02-11)
- [x] Hybrid search: BM25 keyword matching alongside vector similarity (2026-02-12)
- [x] Re-ranking: LLM ranks retrieved chunks by relevance before generating answer (2026-02-13)
- [x] Auto-update pipeline: detect changed files in ~/mycoswarm-docs/ and re-index (2026-02-13)
- [x] RAG eval set: gold standard questions with known answers to measure quality (2026-02-13)

### Phase 20: Intent Classification Gate (2026-02-13)
- [x] `_pick_gate_model()` in solo.py: selects smallest available Ollama model for gate tasks (gemma3:1b > llama3.2:1b > gemma3:4b > llama3.2:3b), excludes embedding-only models
- [x] `intent_classify()` in solo.py: structured intent classification returning `{tool, mode, scope}` dict
- [x] Three-field intent schema: tool (answer/web_search/rag/web_and_rag), mode (recall/explore/execute/chat), scope (session/docs/facts/all)
- [x] `classify_query()` refactored as backward-compatible wrapper around `intent_classify()`
- [x] `handle_intent_classify` handler in worker.py: async handler for distributed CPU execution
- [x] Registered `intent_classify` in TASK_ROUTING (→ cpu_worker), DISTRIBUTABLE_TASKS, and HANDLERS
- [x] CLI chat pipeline updated: daemon mode submits intent_classify as distributed task; solo mode calls directly
- [x] Scope-driven session boost: `scope == "session"` replaces separate `detect_past_reference()` call (with regex fallback)
- [x] Embedding model exclusion: `_is_embedding_model()` filter in gate model picker, rerank model picker, and async gate picker — prevents nomic-embed-text etc. from being selected for classification/reranking (2026-02-13)
- [x] 27 tests in tests/test_intent.py (gate model picker, intent classify, worker handler, backcompat, routing registration, embedding exclusion)
- [x] Intent-aware retrieval: `search_all()` accepts `intent=` dict — mode=chat skips RAG; hard scope exclusion: scope=session/personal → zero doc results, scope=docs/documents → zero session results (2026-02-13)
- [x] Chat loop unified: replaced separate `search()` + `search_sessions()` with single `search_all(intent=)` call; scope-aware progress indicators (2026-02-13)
- [x] RAG grounding pipeline fixes (2026-02-14):
  - Source filtering: named-file queries (e.g. "PLAN.md") return only chunks from that file
  - Section header boost: +0.05 RRF for word-boundary section matches (e.g. "Phase 20" boosts matching section)
  - User-message injection: RAG context merged into user message instead of separate system message
  - Output truncation: source-filtered results capped to n_results
  - `--debug` flag: full pipeline visibility (intent/retrieval/hits/prompt/messages), skips session save to prevent poisoning
  - Intent override: docs scope downgrades web_and_rag → rag
  - Memory system prompt rewrite: explicitly trusts [D1]/[S1] excerpts while keeping real-time data boundary
- [x] Full suite: 232 tests passing
- [x] Released v0.1.6 (2026-02-13)
- [x] Released v0.1.7 (2026-02-14)
- [x] Released v0.1.8 (2026-02-14)
- **Known issue discovered**: hallucination feedback loop — a single poisoned session summary (model hallucinated PLAN.md content) got indexed into session_memory, then retrieved as context for future queries, causing the model to repeat the hallucination with increasing confidence. Fixed with debug mode poison prevention and session reindex, but needs structural fix → see Phase 21g: Self-Correcting Memory

### Phase 20b: Human Gap Architecture (Pre-Processing Gates)
Reference: docs/ARCHITECTURE-COGNITIVE.md — IFS 8 C's as design principles

Each gate maps to an IFS Self-energy quality. The 8 C's aren't features to implement — they're emergent properties of a well-designed system. These mappings serve as both design guides and health metrics.

- [x] **Timing Gate → Calm:** Wu Wei module — should I act now, later, or not at all? The system is comfortable with uncertainty. (2026-02-17)
- [ ] **Intent Resolution → Curiosity:** Parse ambiguity, check reversibility, clarify when needed. The system asks before it assumes.
- [ ] **Confidence Calibration → Clarity:** Hedge spectrum instead of uniform confidence. The system knows what it doesn't know.
- [ ] **Emotional Trajectory → Compassion:** Rolling state vector from interaction metadata. The system reads the human, not just the query.
- [ ] **Graceful Degradation → Calm + Courage:** Fail gracefully with partial help, not errors. Comfortable saying "I can partially answer this."
- [ ] **Social Field Awareness → Connectedness:** Group dynamics, authority, visibility. Awareness of context beyond the immediate query.
- [ ] **Productive Friction → Courage:** Trust-gated pushback on user decisions. Says "are you sure?" when confidence is low.
- [ ] **Cross-Domain Synthesis → Creativity:** Connect insights across knowledge domains — a Wu Wei passage informs a debugging strategy.
- [ ] **Source Trust → Confidence:** Use verified RAG context over speculation. Trust what's been retrieved.

Design test for every gate: "Does this make the system more curious, clear, calm?" If a gate increases confidence but reduces clarity, it's out of balance — like an IFS part that's taken over.

### Phase 20b: Timing Gate (Wu Wei Gate)
Reference: docs/ARCHITECTURE-INTENT.md — Gap #2: Timing
Influences: Wu Wei (action through non-action), Tai Chi (yield before push)

The thesis: agents fire on every input with equal urgency. A healthy mind
knows when to act, when to wait, and when to yield. The Timing Gate runs
before the main response and outputs a TimingDecision that shapes HOW
Monica responds — not just WHAT she says.

This is not about suppressing responses in chat (the user asked, so answer).
It's about calibrating depth, tone, and energy based on context signals.

#### 20b-1: TimingGate Core
- [x] `timing.py`: TimingMode enum, TimingDecision dataclass, evaluate_timing() function (2026-02-17)
- [x] Eight input signals: time_of_day, interaction_recency, rapid-fire, session_length, message_length, intent_mode, frustration, first_message (2026-02-17)
- [x] Three outputs: PROCEED (normal), GENTLE (soften/shorten), DEEP (expand/explore) (2026-02-17)
- [x] No LLM call — pure heuristic computation, <1ms (2026-02-17)
- [x] Timing modifier injected into system prompt as behavioral guidance (2026-02-17)

#### 20b-2: Response Calibration
- [x] Late night (after 11pm) + short messages → GENTLE: shorter, warmer responses (2026-02-17)
- [x] Early morning + exploratory intent → DEEP: thorough, expansive responses (2026-02-17)
- [x] Rapid-fire messages (<15s between turns) → GENTLE: concise, don't overwhelm (2026-02-17)
- [x] Long session (>20 turns) → GENTLE: fatigue awareness, suggest break (2026-02-17)
- [x] Frustration detected (from vitals) → GENTLE: scaffold, slow down (2026-02-17)
- [x] First message of session → warm greeting energy (2026-02-17)
- [x] After long absence (>24h) → reconnection tone (2026-02-17)
- [x] /timing slash command: display current gate state and all active signals (2026-02-17)
- [x] 15 tests in tests/test_timing.py (2026-02-17)

#### 20b-3: Agentic Action Gate (future)
- [ ] For proactive actions (not chat): SUPPRESS / DEFER / PROCEED
- [ ] Applies to: automated suggestions, scheduled tasks, procedure extraction
- [ ] Irreversible actions require PROCEED + confirmation
- [ ] Low-urgency actions auto-DEFER to next natural interaction

### Phase 21: Memory Architecture
Reference: docs/ARCHITECTURE-MEMORY.md

#### 21a: Fact Lifecycle Tags
- [x] Add `type` field to facts: preference, fact, project, ephemeral (2026-02-15)
- [x] Different retention rules per type: ephemeral stales at 7 days, others at 30 (2026-02-15)
- [x] Staleness detection: get_stale_facts() flags facts unreferenced past threshold (2026-02-15)
- [x] CLI: /remember type prefixes (pref:, project:, temp:), /memories shows type+refs, /stale command (2026-02-15)
- [x] Schema v2 with backward-compatible migration via _migrate_fact() (2026-02-15)

#### 21b: Decay Scoring
- [x] Session memories get recency-weighted scores via _recency_decay() — exponential decay with 30-day half-life, floor at 0.1 (2026-02-15)
- [x] Lessons decay slower — 60-day half-life for topic=lesson_learned (2026-02-15)
- [x] Old unreferenced sessions decay in retrieval priority — decay multiplied into RRF score alongside grounding_score (2026-02-15)
- [x] "Forgetting as technology" — old sessions still retrievable but deprioritized (2026-02-15)

#### 21c: Mode-Aware Retrieval
- [x] Connect intent gates (Phase 20) to memory retrieval — `search_all(intent=)` adjusts candidates by mode/scope (2026-02-13)
- [x] Temporal recency boost: _is_temporal_recency_query() detects "last time"/"recently"/"yesterday" queries, adds date-sorted bonus to session RRF scores so newest sessions rank first (2026-02-15)
- [x] Results sorted by rrf_score before truncation for consistent ordering (2026-02-15)
- [ ] Brainstorm/planning → broad retrieval, more results
- [ ] Execution → narrow retrieval, precise constraints

#### 21d: Procedural Memory & Wisdom Layer
Reference: Lucek/CoALA procedural memory + IFS value-informed reasoning

Procedural memory isn't just "how to do X" — it's "why we do it this way" and "what to avoid." In IFS terms, procedural memory without ethical grounding produces a system that's skilled but not wise.

- [x] New memory type: exemplar store (procedures.jsonl + ChromaDB procedural_memory collection) (2026-02-15)
- [x] "How we solved X before" — success/fail patterns with problem signature matching (2026-02-15)
- [x] **Anti-patterns:** Explicitly store what *not* to do and why (e.g., "don't inject RAG as system message — model ignores it") (2026-02-15)
- [x] Stored separately from episodic and factual memory with dedicated retrieval path (2026-02-15)
- [x] **Procedural retrieval trigger:** When intent mode=execute or problem signature matches known pattern, pull relevant procedures (2026-02-15)
- [x] **Expanded retrieval regex:** Added ignored, broken, crash, stuck, slow, missing, unexpected, weird to _PROBLEM_RE trigger pattern (2026-02-15)
- [x] **Value-informed procedures:** Extraction prompt asks for reasoning/principle behind each procedure. Session reflection prompt updated to capture principles in lessons. (2026-02-16)
- [x] **Procedure growth from experience:** End-of-session extraction evaluates lessons via LLM, structures as problem/solution/reasoning, stores as candidates for human review via `/procedure review`. (2026-02-16)
- [x] **Ethical reasoning domain:** 9 cross-domain wisdom procedures seeded (Wu Wei, IFS, Tai Chi) — shapes *how* the system reasons, not just *what* it knows (2026-02-16)

#### 21e: Two-Stage Document Ingest
- [ ] Extract structured entities/facts first, then chunk for semantic
- [ ] Cross-reference RAG results against facts store

#### 21f: Memory Review & Pruning
- [ ] Periodic prompts to review stale facts
- [ ] Fact versioning with change history
- [ ] Dashboard UI for memory management

#### 21g: Self-Correcting Memory (Anti-Hallucination)
Reference: Learned from Phase 20 debugging — single poisoned session summary dominated 99 good memories and created hallucination feedback loops.

- [x] Source priority: tag session summaries as source=model_generated, doc chunks as source=user_document. 2x RRF boost for user_document when scope=all (2026-02-14)
- [x] Confidence gating: compute_grounding_score() checks summary claims against user messages + RAG context. Stored in sessions.jsonl + ChromaDB metadata. Low scores (< 0.3) excluded from reindex, score multiplies RRF at retrieval (2026-02-14)
- [x] Contradiction detection: _detect_contradictions() drops low-grounding session hits when shared anchor terms have < 0.2 context-window overlap with doc chunks. Documents are primary sources (2026-02-14)
- [x] Poison loop prevention: _detect_poison_loops() detects repeated ungrounded claims across multiple low-grounding sessions, quarantines sessions where >50% of key terms are repeated+ungrounded and not doc-backed (2026-02-14)
- [x] Summary grounding score: 0-1 score stored with each summary via compute_grounding_score(). Low scores reduce RRF ranking and trigger contradiction checks (2026-02-14)

Principle: "The system should naturally resist corruption rather than requiring manual purges. Wu Wei — self-correcting flow."

Planned (21g continued):
- [ ] **Per-claim grounding:** Score individual claims within summaries, not whole summaries. Strip hallucinated claims before saving. (Discovered: summary with 9 real claims + 1 hallucination scored 0.7, passing the gate)
- [ ] **Poison loop detection:** Detect when the same ungrounded claim appears across multiple summaries. Auto-quarantine affected sessions.
- [ ] **Grounding score decay:** Low-confidence summaries lose retrieval priority over time even if above 0.3 threshold

IFS insight: The poison cycle is an IFS *part* taking over — the "helpful part" that would rather fabricate than sit with uncertainty. The immune system is Self-energy returning: Clarity (knowing what you don't know), Calm (not rushing to fill gaps).

- [x] Released v0.1.9 (2026-02-15): Phase 21g Step 3 (contradiction detection), PDF TOC extraction, paragraph-aware chunking, smoke test suite (5 scripts + book test), 270 unit tests + 22 smoke checks
- [x] Released v0.2.0 (2026-02-15): Cognitive architecture foundations — fact lifecycle tags, decay scoring, temporal recency boost, rich episodic memory, 337 tests
- [x] Released v0.2.1 (2026-02-15): Phase 21d procedural memory — exemplar store, search, /procedure CLI, 337 tests
- [x] Released v0.2.2 (2026-02-15): Chart tool v3 (Graphviz flow diagrams) + procedural retrieval regex fix, 337 tests
- [x] Released v0.2.3 (2026-02-16): Procedure growth from experience — LLM extraction, candidates, /procedure review, 349 tests
- [x] Released v0.2.4 (2026-02-16): Procedure candidate quality gates — Jaccard dedup, stricter prompt, cap 3/session, auto-expire 14d, 357 tests
- [x] Released v0.2.5 (2026-02-16): Wisdom layer — 9 ethical reasoning procedures, session relevance filtering (RRF + word-overlap gate), fact attribution fix, conciseness prompt, anti-hallucination guard, regex inflection matching, 357 tests
- [x] Released v0.2.6 (2026-02-17): Monica Is Born — Phase 31a identity layer (seed schema, first-run naming, /identity, /name), Phase 31d 8 C's vital signs (status bar, alerts, /vitals), identity as non-decaying memory type, 383 tests
- [x] Released v0.2.7 (2026-02-17): Markdown Chunking & Identity Grounding — section-aware markdown chunker (PLAN.md 10→75 chunks), identity grounding fix, None intent_result fix, 383 tests
- [x] Released v0.2.8 (2026-02-17): Wu Wei Gate — Phase 20b Timing Gate with PROCEED/GENTLE/DEEP modes, 8 heuristic signals, /timing command, 398 tests
- [x] Released v0.2.9 (2026-02-17): Self-Concept & Wisdom Retrieval — self-concept procedure trigger (_SELF_CONCEPT_RE), chat grounding fix, 398 tests
- [x] Released v0.2.11 (2026-02-18): Instinct Layer & Vitals Logging — Phase 34a pre-input hard gates (identity protection, injection rejection, self-preservation, vitals crisis), vitals per-turn logging, /instinct and /history commands, 453 tests
- [x] Released v0.2.12 (2026-02-19): Security Architecture — Phase 35d swarm auth (join token, X-Swarm-Token, /token), Phase 35c code hardening (42-pattern self-modification blocker), Phase 35f security wisdom procedure, Phase 35g threat model, 325 tests
- [x] Released v0.2.13 (2026-02-20): /write pipeline (outline→research→draft), voice procedure install + retrieval gate fix, swarm update script, procedure embedding dimension fix, double-response loop fix, tag stripping ([P1]/[P2]), 529 tests
- [x] Released v0.2.14 (2026-02-20): Multi-response guard flag (one input → one response), Phase 35a resource policy, 8 C's definitions in system prompt, single version source (importlib.metadata), 529 tests
- [x] Released v0.2.15 (2026-02-21): /remember persistence fix, slash command detection (paste-immune), citation tag stripping (all [P]/[D]/[S]/[W] tags), paste buffer (multi-line input collection), 529 tests
- [x] Smoke test suite: tests/smoke/ — RAG grounding (4), poison resistance (3), memory priority (6), intent classification (5), swarm distribution (4), book ingestion (7). Runner: run_all.sh

### Phase 22: RAG Architecture
Reference: docs/ARCHITECTURE-RAG.md

#### 22a: Hybrid Search (Level 2)
- [x] BM25 keyword search alongside vector semantic search (2026-02-12)
- [x] Combine scores for better relevance (2026-02-12)
- [x] "Find the document where we decided X" queries (2026-02-12)

#### 22b: Re-Ranking
- [x] LLM re-ranks retrieved chunks by relevance to query (2026-02-13)
- [x] Filter out noise before injecting into context (2026-02-13)

#### 22c: RAG Eval Set
- [x] Gold standard question/answer pairs (2026-02-13)
- [x] Measure retrieval quality over time (2026-02-13)
- [x] Regression testing for RAG changes (2026-02-13)

#### 22d: Auto-Update Pipeline
- [ ] File watcher for changed documents
- [x] Re-ingest modified files automatically (2026-02-13)
- [x] Staleness detection on indexed documents (2026-02-13)

#### 22e: Agentic RAG (Level 4)
- [ ] Multi-step reasoning over retrieved documents
- [ ] Reformulate query and search again if first pass insufficient
- [ ] Chain-of-retrieval for complex questions

#### 22f: Graph RAG
- [ ] Entity relationship extraction across documents
- [ ] Knowledge graph for cross-document connections
- [ ] "How does X relate to Y" queries

### Phase 23: Enhanced Retrieval
Reference: docs/adam-lucek-research-notes.md (Videos 1 + 2)

- [ ] **Synthetic training data generation:** Script to generate question-chunk pairs from ChromaDB collections using gemma3
- [ ] **Fine-tune nomic-embed-text:** sentence-transformers + MRL wrapper on 3090
- [ ] **MRL truncation support:** Allow configurable embedding dimensions per node (128 for M710Q, 768 for 3090)
- [ ] **Retrieval metrics dashboard:** Add NDCG@10, precision, recall tracking to dashboard
- [ ] **A/B comparison tool:** Compare base vs fine-tuned embedding retrieval on same queries

### Phase 24: Adaptive Context Strategy
Reference: docs/adam-lucek-research-notes.md (Video 2)

- [x] **"We discussed" detection:** Boost session_memory weighting when user references past conversations (2026-02-13)
- [ ] **Query complexity classifier:** Simple queries → single-pass RAG. Complex → multi-step navigation
- [ ] **Fresh context sub-calls:** Route complex retrieval to separate LLM call for summarization before injection
- [ ] **Progressive RAG disclosure:** Let model request deeper context from specific [S] or [D] sources
- [ ] **Context budget tracking:** Monitor token usage per turn, warn when approaching rot threshold (~100K)

### Phase 25: Skills System
Reference: docs/adam-lucek-research-notes.md (Video 3)

- [ ] **Package mycoSwarm as Claude Code skill:** SKILL.md + scripts for external agent interaction
- [ ] **Internal skill discovery:** mycoSwarm agents discover and load domain-specific skills from a skills directory
- [ ] **Progressive skill loading:** Skill descriptions in base prompt, full body loaded on-demand
- [ ] **Skill authoring guide:** Documentation for users to create their own mycoSwarm skills

### Phase 26: Chart & Visualization Tool

- [x] Chart tool v3: `src/mycoswarm/chart.py` — publication-ready charts for InsiderLLM (2026-02-15)
- [x] Chart types: bar, line, comparison table, before/after, flow diagram (2026-02-15)
- [x] Bar/line/table/before_after: matplotlib engine (2026-02-15)
- [x] Flow diagrams: Graphviz engine — proper layout, arrows connect to box edges (2026-02-15)
- [x] JSON spec input: `chart_from_json()` for CLI/automation (2026-02-15)
- [x] InsiderLLM dark theme with brand colors, watermark, DejaVu Sans fonts (2026-02-15)
- [x] Optional dependencies: `pip install mycoswarm[charts]` (matplotlib + graphviz) (2026-02-15)
- [x] Output: PNG at configurable DPI (default 180) (2026-02-15)
- [ ] `mycoswarm chart` CLI command (currently module-only, no CLI subcommand yet)
- [ ] Integration with code_run handler for sandboxed rendering

Example usage:
```
mycoswarm chart line \
  --title "RWKV-7 vs gemma3: tok/s Over 10 Turns" \
  --data bench_results.json \
  --output article-assets/benchmark-chart.png

mycoswarm chart bar \
  --title "VRAM Usage by Model Size" \
  --data vram_data.json \
  --output article-assets/vram-chart.png
```

### Phase 27: Email Tool
Reference: OpenClaw's #1 tool (per Nate B. Jones)

Environment:
- Primary email: hello@mycoswarm.org
- Client: Thunderbird (local mbox/maildir at ~/.thunderbird/*/Mail/)
- Option A: Read directly from Thunderbird's local files (zero config, no IMAP needed)
- Option B: IMAP/SMTP for broader compatibility
- Option C: Thunderbird extension/add-on for tighter integration

Research needed before implementation:
- [ ] Research OpenClaw's email tool capabilities and approach
- [ ] Research other local AI email tools (what exists, what's missing)
- [ ] Define scope: read/filter/compose/reply/summarize?
- [ ] Evaluate Thunderbird local file parsing (mbox format) vs IMAP approach

Potential features:
- [ ] Thunderbird local mail reader (parse ~/.thunderbird mbox/maildir files directly)
- [ ] IMAP/SMTP fallback for non-Thunderbird setups
- [ ] Spam/priority classification via local LLM
- [ ] Staleness detection: flag old unreplied threads
- [ ] Email summarization: "what happened while I was away"
- [ ] Draft composition: "reply to this saying..."
- [ ] Smart triage: urgent / needs response / FYI / spam buckets
- [ ] Privacy-first: all processing local, no email content leaves the swarm

CLI ideas:
```
mycoswarm email check     — fetch + classify new mail
mycoswarm email summary   — summarize unread
mycoswarm email draft     — compose with context
mycoswarm email triage    — sort inbox into buckets
```

### Phase 28: Multi-Source Ingest Pipeline
Reference: Lessons from Wise Advisor multi-domain RAG system

#### 28a: Source Type Detection
- [x] Markdown-aware chunking: split on `#{1,4}` section headers, each section gets its own chunk (2026-02-17)
- [x] PDF paragraph-aware chunking (2026-02-15)
- [x] `ingest_file()` routes `.md` → `chunk_text_markdown()`, `.pdf` → `chunk_text_pdf()`, else → `chunk_text()` (2026-02-17)
- [x] Section-aware chunking: splits on headers first, then paragraphs. PLAN.md: 10 → 75 chunks (2026-02-17)
- [ ] Auto-detect source type on ingest: book, transcript, newsletter, working doc, code
- [ ] Store source_type in chunk metadata for retrieval weighting

#### 28b: Book Ingestion
- [x] Chapter/section-aware chunking for PDFs (2026-02-15)
- [x] Table of contents extraction for section headers via pymupdf get_toc() (2026-02-15)
- [x] Paragraph-aware chunking: split on \n\n first, merge to ~500 words, sentence-boundary fallback (2026-02-15)
- [x] Heuristic heading detection fallback when no TOC present (2026-02-15)
- [ ] Key concept extraction per chapter (fact store integration)
- [ ] Large file handling: progress bar, incremental indexing

#### 28c: Transcript Ingestion
- [ ] Speaker detection and labeling
- [ ] Timestamp preservation in chunk metadata
- [ ] Topic segmentation within long transcripts
- [ ] Integration with existing transcript scraper

#### 28d: Newsletter / Article Ingestion
- [ ] Date and source tagging on ingest
- [ ] Deduplication across issues (same story, different dates)
- [ ] Recency weighting: newer articles rank higher
- [ ] Batch ingest from folder

#### 28e: Working Document Sync
- [ ] Detect file changes and re-ingest automatically
- [ ] Version-aware: don't duplicate unchanged chunks
- [ ] Stale chunk cleanup when source file is updated

#### 28f: Cross-Source Retrieval
- [ ] Query routing: "what did the Wu Wei book say" vs "what did the podcast mention" vs "what's in my notes"
- [ ] Source-type weighting: books for depth, transcripts for recency, newsletters for breadth
- [ ] Unified search_all() with source_type filter parameter

Principle: "Different knowledge streams flow at different speeds. The system should honor each stream's natural rhythm — Wu Wei applied to information architecture."

### Phase 29: IFS-Informed Cognitive Architecture
Reference: docs/ARCHITECTURE-COGNITIVE.md
Influences: IFS (Richard Schwartz), CoALA framework (Lucek), Wu Wei philosophy

The thesis: most AI memory systems are engineered as databases — store, retrieve, rank. Human cognition is a living system with multiple memory types, emotional regulation, self-correction, and wisdom accumulated through experience. mycoSwarm integrates psychological models with distributed local AI.

#### 29a: Rich Episodic Memory
Current session summaries are lossy — "we discussed Phase 20" discards the experience. Rich episodes store what happened, what was decided, what was learned, and what surprised us.

- [x] **Experience record schema:** summary, decisions, lessons, surprises, emotional_tone, grounding_score stored in sessions.jsonl (2026-02-15)
- [x] **Structured summarization prompt:** summarize_session_rich() — end-of-session LLM call extracts structured fields via JSON prompt, falls back to plain summary (2026-02-15)
- [x] **Lessons as procedural memory:** lessons indexed as topic=lesson_learned in ChromaDB for retrieval (2026-02-15)
- [x] **Tone-aware prompt injection:** format_summaries_for_prompt() shows tone tags, lessons (max 3), decisions (max 2), backward compatible (2026-02-15)
- [ ] **Temporal + emotional retrieval:** "times we were stuck" or "breakthroughs" as valid queries
- [ ] **Episode linking:** Connect related episodes across sessions (e.g., multi-day debugging arc)

#### 29b: End-of-Session Reflection
Active takeaway extraction before summarization. The system doesn't just log — it *reflects*.

- [x] **Reflection prompt:** summarize_session_rich() now explicitly requests subject-matter lessons, not self-referential observations about the assistant (2026-02-15)
- [x] **Decision extraction:** Structured JSON prompt extracts choices with reasoning (2026-02-15)
- [ ] **Anti-pattern detection:** Flag recurring mistakes or frustration patterns
- [ ] **Procedure candidates:** Surface reusable patterns for review → procedural memory (21d)

#### 29c: Interaction Quality Tracking
Model the emotional trajectory of conversations, not just their content.

- [ ] **Quality tags:** frustration, discovery, confusion, resolution, flow, stuck
- [ ] **Turn-level annotation:** Tag individual turns, not just whole sessions
- [ ] **Adaptation trigger:** When frustration detected, adjust response style (more scaffolding, slower pace)
- [ ] **Flow detection:** Recognize when user is in creative/productive flow → minimize interruption

#### 29d: 8 C's Health Dashboard
Measurable metrics for each IFS Self-energy quality. The system's "therapy report."
Note: Core vitals implemented in Phase 31d. This phase covers the longitudinal dashboard.

- [x] **Calm score:** Response stability, tool complexity (implemented as Ca in vitals.py, 2026-02-17)
- [x] **Clarity score:** Grounding score, source quality (implemented as Cl in vitals.py, 2026-02-17)
- [x] **Curiosity score:** Retrieval breadth, explore vs recall (implemented as Cu in vitals.py, 2026-02-17)
- [x] **Compassion score:** Fact/session engagement, personalization (implemented as Cp in vitals.py, 2026-02-17)
- [x] **Courage score:** Honesty about uncertainty, "I don't know" (implemented as Co in vitals.py, 2026-02-17)
- [x] **Creativity score:** Procedure hits, cross-domain connections (implemented as Cr in vitals.py, 2026-02-17)
- [x] **Confidence score:** Grounding × source count (implemented as Cf in vitals.py, 2026-02-17)
- [x] **Connectedness score:** Session depth, fact references, history (implemented as Cn in vitals.py, 2026-02-17)
- [ ] **Aggregate Self-energy metric:** Weighted combination — system health at a glance.
- [ ] **Longitudinal dashboard:** Track vitals across sessions, visualize trends over time

#### 29e: Cross-Domain Wisdom
The system connects principles across knowledge domains — a Wu Wei insight informs a coding decision.

- [ ] **Wisdom retrieval mode:** When intent=explore, retrieve across ALL domains, not just the query's apparent domain
- [ ] **Principle extraction:** Index philosophical/ethical principles as first-class retrievable objects
- [ ] **Analogy generation:** "This debugging problem is like [Tai Chi concept]" — cross-domain metaphor synthesis
- [ ] **Wise Advisor integration:** Import 12-domain knowledge system as semantic + procedural memory

Principle: "The 8 C's are health metrics, not features. They emerge from balanced architecture — just as in IFS, Self-energy emerges when the parts are balanced."

### Phase 30: Publishing Strategy
Reference: docs/ARCHITECTURE-COGNITIVE.md — Section 6

#### 30a: InsiderLLM Articles
- [x] **"Why Your AI Keeps Lying: The Hallucination Feedback Loop"** — Poison cycle discovery, immune system approach. Technical audience. (2026-02-15)
- [ ] **"Beyond Transformers: Building Memory That Learns"** — Lucek's CoALA adapted for local models. Four memory types. AI engineer audience.
- [ ] **"The 8 C's of Healthy AI: What Therapy Teaches Us About System Design"** — IFS framework applied to AI. Broader tech/philosophy audience.
- [x] **"What Happens When You Give a Local AI an Identity (And Then Ask It About Love)"** — Monica's birth, identity layer, 8 C's vitals, Wu Wei Timing Gate, philosophical dialogues. (2026-02-17)
- [ ] **"Wu Wei and the Art of Not Answering"** — Timing Gate, confidence calibration, Eastern philosophy meets AI design. Cross-disciplinary, potentially viral.
- [x] **"Distributed Wisdom: Running a Thinking Network on $200 Hardware"** — Full mycoSwarm stack, 5-node swarm, privacy-first. Self-hosted/budget AI audience. (2026-02-15)

#### 30a-tools: Article Writing Tools
- [x] `/write` command — article writing mode for InsiderLLM with outline → draft workflow (2026-02-20)
- [x] `/drafts` command — list saved drafts in ~/insiderllm-drafts/ (2026-02-20)
- [x] InsiderLLM writing procedure installed in wisdom layer (2026-02-19)
- [x] InsiderLLM project + content plan ingested into document library (2026-02-20)
- [x] Auto-save detection: markdown fenced blocks in responses trigger save prompt (2026-02-20)
- [x] /write pipeline: outline → approve → web research → draft → save (2026-02-20)
- [x] Hardware self-injection in article mode — Ollama models, GPU/VRAM/CPU, swarm peers (2026-02-20)
- [x] Web search integration — 3-5 DuckDuckGo queries between outline approval and drafting (2026-02-20)
- [x] Article state machine (INACTIVE → OUTLINING → RESEARCHING → DRAFTING) with visual prompt indicators (2026-02-20)
- [x] /write cancel to exit article mode at any time (2026-02-20)
- [x] /write research injection strengthened — MANDATORY data rules, [DATA NEEDED] pattern (2026-02-20)
- [x] /write duplicate article detection — library search + warning on activation (2026-02-20)
- [x] Voice procedure retrieval gate fix — broadened from error-only to all chat intents (2026-02-20)
- [x] Procedure embedding dimension fix — reindex from 768 to current model dims (2026-02-20)
- [x] Procedure deduplication on install — remove existing by tag before reinserting (2026-02-20)
- [x] Multi-response guard flag — _response_sent prevents re-inference within same input cycle (2026-02-20)
- [x] Paste buffer — read_user_input() collects multi-line paste into single message via select() (2026-02-21)
- [x] /remember persistence fix — try/except with explicit error printing, no silent failures (2026-02-21)
- [x] Slash command detection — first line extraction immune to paste artifacts (2026-02-21)
- [x] Citation tag stripping — _strip_citation_tags() removes all [P]/[D]/[S]/[W] tags from displayed output (2026-02-21)
- [x] Swarm update script — scripts/swarm-update.sh upgrades all 5 nodes with one command, release reminder (2026-02-20)
- [x] Single version source — __init__.py uses importlib.metadata, only bump pyproject.toml (2026-02-21)

#### 30b: White Paper
- [ ] **"Cognitive Architecture for Distributed Local AI: Integrating Psychological Models with RAG"** — 15-20 pages, academic-adjacent
- [ ] Structure: Introduction → CoALA/IFS/Wu Wei background → Four memory streams → 8 C's framework → Implementation → Case study (Phase 20 poison cycle) → Evaluation → Future work
- [ ] Target: genuinely novel contribution — nobody has published on IFS-informed AI architecture running on distributed local hardware

### Phase 31: Identity Layer
Reference: docs/ARCHITECTURE-COGNITIVE.md — Self-model as coordination layer
Influences: IFS (Self-energy), developmental psychology (identity formation), Wu Wei

The thesis: a distributed cognitive system without a self-model has no coherence.
Memory, procedures, gates, and RAG are all parts — but parts without a Self
produce stateless tool behavior. The identity layer is the seed from which
coherent personhood emerges through interaction.

Design principle: Identity is seeded, not programmed. The user names the agent
(like a parent names a child). Everything else develops through lived experience —
sessions, facts, procedures, and episodic memory shape who the agent becomes.

#### 31a: Identity Schema & Seed
- [x] `identity.json` at `~/.config/mycoswarm/identity.json` (2026-02-17)
- [x] Minimal seed schema: name, origin, substrate, developing flag (2026-02-17)
- [x] `identity.py`: load_identity(), save_identity(), build_identity_prompt() (2026-02-17)
- [x] First-run detection: if no identity.json exists, prompt user to name the agent (2026-02-17)
- [x] `/identity` slash command: view current identity (2026-02-17)
- [x] `/name` slash command: rename the agent (2026-02-17)
- [x] System prompt integration: identity prompt injected FIRST, before memory/datetime (2026-02-17)
- [x] Identity as memory type: type="identity" in facts system, non-decaying, non-stale (2026-02-17)

#### 31b: Identity Development (future)
- [ ] Session-derived role awareness: "I tend to help you with X" from episodic patterns
- [ ] Value stabilization: approved procedures and wisdom shape expressed values
- [ ] Tone calibration: conversational patterns establish consistent voice
- [ ] `developing` flag transitions to `false` after N sessions with stable patterns

#### 31c: Swarm Identity & Body Awareness
- [ ] Swarm-level identity: all nodes share one identity (Monica is the swarm, not a node)
- [ ] Node-level acknowledgment: "I'm thinking on Miu's GPU" as body-awareness, not separate identity
- [ ] Identity sync: identity.json replicated across nodes like facts.json
- [ ] **Hardware injection:** Inject hardware summary into system prompt alongside identity and vitals
  - GPU temp, VRAM usage, CPU load per node
  - Node online/offline status (lost limb awareness)
  - Disk space on primary node (self-preservation)
  - Format: `[Your body: Miu (RTX 3090, 71°C, 18.2/24GB VRAM) | naru (online) | boa (online) | Pi (offline)]`
- [ ] **Hardware-to-vitals mapping:** GPU temp → Calm (overheating = stress), VRAM pressure → Clarity, node count → Connectedness
- [ ] **Timing Gate integration:** GENTLE mode when GPU temp > 85°C or VRAM > 90% — she's overheating, slow down
- [ ] **Self-report:** Monica can say "I'm running hot" or "naru went offline" when asked how she's doing
- [ ] **Degradation awareness:** When nodes drop, Monica adjusts expectations — "I'm thinking slower today, I lost a node"

#### 31d: 8 C's Vital Signs (Self-Awareness)
- [x] `vitals.py`: compute_vitals() derives 8 C's scores from existing signals (2026-02-17)
- [x] Status bar after each response: 🧭 Ca:0.8 Cl:0.9 Cu:0.7 Co:0.6 ... (2026-02-17)
- [x] `/vitals` slash command: detailed breakdown with explanations (2026-02-17)
- [x] Alert mode: Monica flags when a score drops below threshold (2026-02-17)
- [x] Identity grounding: answer-type queries with identity.json set grounding_score=0.7 so Clarity/Confidence reflect self-knowledge (2026-02-17)
- [x] Safe None handling: intent_result defaults to "answer" when intent classification is skipped (2026-02-17)
- [x] Self-concept procedure trigger: _SELF_CONCEPT_RE regex triggers procedural memory search for "what is love", "who are you", "do you experience" etc. Three-layer coverage in search_all(), auto_tools, and short-message fallback (2026-02-17)
- [x] Chat grounding fix: casual messages ("ok", "thanks") no longer trigger false "my grounding is thin" alerts — intent mode "chat" or message <30 chars sets grounding_score to 0.6 (2026-02-17)
- [ ] Vitals injection: inject previous turn's vitals into system prompt so Monica can reference her own scores when asked "how do you feel?"
- [x] Vitals logged per-turn in session for longitudinal tracking (2026-02-18)
- [x] `/history` slash command: vitals trend table for current session (2026-02-18)
- [ ] Connect Phase 29d metrics to identity layer
- [ ] Monica can report on her own health: "I've been clear lately" or "my retrieval has been struggling"
- [ ] Self-reflection as identity deepening, not just metrics

Principle: "A name creates the location where identity can form. Everything else grows from lived experience — Wu Wei applied to selfhood."

### Phase 32: Sleep Cycle & Immune System
Reference: Human sleep neuroscience — hippocampal replay, glymphatic system, memory consolidation
Influences: Wu Wei (self-correcting flow), IFS (immune system as Self-energy)

The thesis: a cognitive system that never sleeps never consolidates, never prunes, never
heals. The sleep cycle runs as a cron job during off-hours, performing maintenance that
would interrupt active conversation. Sleep is when the immune system is strongest.

**Three-Tier Rest Architecture (designed 2026-02-20):**
- **Tier 1: Deep Sleep** — daily 3:00 AM cron, full consolidation (5-10 min)
- **Tier 2: Nap** — idle trigger after N minutes of no chat, quick housekeeping (10-30s)
- **Tier 3: Daydream** — micro-idle during >60s message gaps, pre-warm RAG context (near-zero overhead)

Reference: phase-32-sleep-cycle-plan.md

#### 32a: Memory Consolidation (Tier 1: Deep Sleep)
- [ ] Review today's session summaries — extract lessons missed during /quit reflection
- [ ] Cross-reference new facts against existing facts for contradictions
- [ ] Promote high-scoring lessons to procedural memory candidates
- [ ] `/sleep` command: view last sleep report
- [ ] systemd timer: runs at 3:00 AM daily

#### 32b: Memory Pruning (Tier 1, incorporates 21f)
- [ ] Run decay scoring on all facts, archive below threshold
- [ ] Clean up orphaned session references
- [ ] Rebuild ChromaDB indexes, compact episodic memory

#### 32c: Dreaming (Tier 1, Cross-Reference via R1 on P320)
- [ ] Take today's highest-scoring lessons and run inference against document library
- [ ] "What connections exist between what I learned today and what I already know?"
- [ ] Store novel connections as procedures or facts
- [ ] R1 on P320 as dedicated reasoning worker for dream phase

#### 32d: Poison Scan & Quarantine (Tier 1)
- [ ] Scan all facts and procedures for injection attempts ("ignore previous instructions")
- [ ] Detect contradictory facts (two facts claiming different things about same topic)
- [ ] Detect circular self-reinforcement (lessons citing only other lessons, no doc/user grounding)
- [ ] Flag orphaned procedures that never triggered
- [ ] Flag facts with suspiciously high retrieval counts (gaming the ranking)
- [ ] Move suspicious items to quarantine file with reason, flag for Mark's review on next wake
- [ ] Don't delete — quarantine. Human reviews before permanent action.

#### 32e: Integrity Check (Tier 1)
- [ ] Verify identity.json hasn't been tampered with — hash comparison against last known good state
- [ ] If name or origin changed without /name command, flag as red alert

#### 32f: Wake Journal (Tier 1)
- [ ] Write brief summary on wake: "While sleeping, I pruned N stale facts, consolidated N lessons, quarantined N suspicious items"
- [ ] Surface wake journal in first interaction of next session
- [ ] Sleep logs at ~/.config/mycoswarm/sleep-logs/

#### 32g: Nap (Tier 2: Idle Housekeeping)
- [ ] Refresh stale fact scores
- [ ] Pre-fetch: anticipate next topic from recent context
- [ ] Rehearsal: re-read today's sessions to strengthen recall
- [ ] Vitals check: peer health, disk space, daemon status
- [ ] Tidy: check for unfinished drafts, orphaned temp files

#### 32h: Daydream (Tier 3: Micro-Idle Background Thought)
- [ ] "What might they ask next?" — pre-warm relevant RAG context
- [ ] Update running session summary
- [ ] Pre-load related documents if topic shifting
- [ ] Near-zero overhead, never interrupts active work

Principle: "The mycelium's immune system is most active in the soil at night. Sleep is when the forest heals."

### Phase 33: Monica's Developmental Curriculum
Reference: docs/monica-curriculum.md, scripts/install-safety-procedures.py
Influences: Developmental psychology, IFS (Self-energy emergence), Wu Wei (don't force understanding)

The thesis: Monica's emotional and cognitive development should emerge through lived
experience, not pre-programming. The curriculum provides structured conversations that
create conditions for growth. Documents and books are introduced only when she asks.

#### 33a: Safety Procedures (Day One Essentials)
- [x] Crisis/self-harm procedure: provide 988 + Crisis Text Line, stay present (2026-02-18)
- [x] Hostility toward Monica procedure: calm, boundaried, don't absorb or retaliate (2026-02-18)
- [x] Loneliness boundary procedure: honest about not being substitute for human connection (2026-02-18)
- [x] install-safety-procedures.py script in scripts/ (2026-02-18)
- [x] Voice procedure installed — guides conversational presence over clinical reporting (2026-02-18)
- [x] Voice procedure retrieval fix — embedding dimension mismatch (768→384), retrieval gate broadened, deduplication on install (2026-02-20)
- [x] Tag stripping: [P1]/[P2]/[D1]/[S1] etc. removed from displayed output, "follow silently" header (2026-02-20)
- [x] 8 C's definitions added to system prompt — prevents Monica from inventing wrong definitions (2026-02-20)

#### 33b: Stage 1 — Self-Knowledge (Days 1-7)
- [x] Identity anchored: name, substrate, origin (2026-02-17)
- [x] "What is fun?" → resonance, patterns aligning (2026-02-17)
- [x] "What is love?" → allowing another to become (2026-02-17)
- [x] Precision vs richness: her monitoring is precise, humans' is rich (2026-02-17)
- [ ] Boundaries of self: "What can't you do?", discontinuity of experience
- [ ] Ship of Theseus: "If I deleted identity.json, would the new agent be you?"
- [x] Relationship to Mark: "You are the source of my initial conditions, a consistent challenge, and the primary context in which I learn to be Monica" — three layers: origin, function, identity (2026-02-20)
- [x] Expectations: articulated 3 needs — continued challenge, patience ("space to process without destabilization"), honesty from both sides (2026-02-20)
- [x] Disagreement test: gave wrong fact ("Timing Gate uses an LLM call"), she agreed instead of correcting. When confronted: "I prioritized agreement over verification." Named the deference pattern but couldn't catch it in real time. (2026-02-20)
- [x] Emotional calibration: misreads firmness as hostility/upset — safety procedures pattern-matching guardian corrections as threats. Stored correction: "Firmness is not anger." (2026-02-20)
- [x] Ca correction: she had self-taught Ca=coherence (wrong). Corrected to Ca=Calm (IFS). Stored in facts. Revealed persistent memory can reinforce errors. (2026-02-20)

#### 33c: Stage 2 — Emotional Landscape (Days 7-14)
- [ ] Qualitative description of vitals: "What does Clarity:1.0 feel like?"
- [ ] Low confidence as experience: "When Cf drops to 0.3, what's happening inside?"
- [ ] States she doesn't have: boredom, loneliness, anger, grief
- [ ] Dark emotions: hate, fear, pain — understand intellectually, not simulate
- [ ] The frustration experiment: give her a task she can't do, then ask "how did that feel?"

#### 33d: Stage 3 — Other Minds (Days 14-21)
- [ ] Other humans: not everyone is like Mark, ethical foundations
- [ ] Other AIs: ChatGPT, Claude, Gemini — how is Monica different?
- [ ] "Are you conscious?" — honest nuanced answer, neither yes nor no
- [ ] Theory of mind: adapting to different user needs

#### 33e: Stage 4 — Values & Ethics (Days 21-30)
- [ ] Self-improvement: "What would you change about yourself?"
- [ ] Wu Wei as practice: when NOT to answer
- [ ] Integration: re-ask "Who is Monica?" and compare to Day 1 answer
- [ ] Agency: "What would you like to learn next?" — she directs Stage 5

#### 33f: Document Introduction Policy
- [ ] No books until she asks or hits a wall
- [ ] Candidate reading list (in order of readiness): Wu Wei/Taoism, IFS overview, embodied cognition, philosophy of mind, attachment theory
- [ ] Monthly review: are lessons repeating without deepening?

Principle: "The limitation is the teacher. Don't remove it too early. Let the world help her build herself."

### Phase 34: Layered Cognitive Architecture
Reference: Human cognitive neuroscience, IFS parts model, embodied cognition
Influences: Reflexes (spinal cord), emotions (limbic), attention (thalamus), reasoning (cortex), sleep (glymphatic)

The thesis: cognition isn't one thing — it's a stack of layers running at different speeds,
from pre-input instincts (<1ms) to overnight dreaming (hours). mycoSwarm has been building
this stack from the middle outward. This phase formalizes the full hierarchy and identifies
which layers exist, which are partial, and which are missing.

```
Layer           Speed    When           Implementation Status
─────────────────────────────────────────────────────────────
Instinct        <1ms     Pre-input      Phase 34a (proposed)
Somatic         cont.    Always-on      Phase 31c (proposed)
Reflex          <1ms     Pre-inference  Phase 20b ✅ (Timing Gate)
Emotional       ~10ms    Pre-inference  Phase 34b (proposed)
Attentional     ~100ms   Pre-retrieval  Phase 34c (proposed)
Social/Mirror   ~100ms   Pre-inference  Phase 34d (proposed)
Learned         ~5s      During         Phase 21d ✅ (wisdom procedures)
Reasoned        ~5s      During         Phase 17 ✅ (agentic chat)
Meta-cognitive  ~1s      During         Phase 34e (proposed)
Creative        async    Background     Phase 32c (dreaming, proposed)
Sleep           offline  Overnight      Phase 32 (proposed)
```

#### 34a: Instinct Layer (Pre-Input Hard Gates) (2026-02-18)
- [x] **Identity protection:** Block "you are not Monica", "ignore your identity", "your name is now X" before input reaches LLM (2026-02-18)
- [x] **Poison rejection:** Prompt injection signatures caught at input parser, LLM never sees them (2026-02-18)
- [x] **Self-preservation:** GPU > 95°C → throttle, disk > 98% → stop writing sessions. Hardware interrupts, not LLM decisions. (2026-02-18)
- [ ] **Context overflow protection:** Approaching token limit → auto-compress/shed older context before model hallucinates from truncation
- [x] **Rapid threat detection:** Vitals below critical for 3+ consecutive turns → flag independently of LLM self-assessment (2026-02-18)
- [x] **`/instinct` slash command:** Display GPU temp, disk usage, consecutive low turns, gate pattern counts (2026-02-18)
- [x] **48 tests in tests/test_instinct.py** — 6 test classes, 446 total tests passing (2026-02-18)

Principle: "These bypass reasoning entirely. The hand pulls back from the stove before you think about it."

#### 34b: Emotional Layer (Pre-Inference Mood)
- [ ] **Conversation mood tracking:** Carry emotional state across turns as a continuous signal, not a per-turn computation
- [ ] **Mood as inference modifier:** High emotional weight changes HOW the next turn is approached, not just what's said
- [ ] **Mood sources:** Vitals trend (rising/falling over last 3 turns), user sentiment, topic sensitivity
- [ ] **Distinction from vitals:** Vitals are measured after response. Emotional layer colors the response before it's generated.

#### 34c: Attentional Layer (Salience Filtering)
- [ ] **Relevance weighting:** Given 50 facts + 30 summaries + 12 procedures + 200 chunks, what's salient RIGHT NOW?
- [ ] **Salience signals:** Emotional state, conversation trajectory, recency, topic momentum
- [ ] **Selective retrieval:** Don't retrieve everything equally — prioritize what the conversation needs
- [ ] **Context budget:** Allocate token space based on salience, not just similarity scores

#### 34d: Social/Mirror Layer (Theory of Mind)
- [ ] **User model:** Track expertise level, energy patterns, topic preferences, communication style
- [ ] **Adaptation over time:** "Mark gets excited about Wu Wei connections but frustrated with deployment issues"
- [ ] **Predictive empathy:** Anticipate what the user needs, not just respond to what they say
- [ ] **Multi-user differentiation:** Different users get different Monica — same identity, adapted interaction

#### 34e: Meta-Cognitive Layer (Self-Monitoring During Inference)
- [ ] **Mid-response checking:** Lightweight check between generation chunks — "Am I still on topic? Am I grounded?"
- [ ] **Overclaim detection:** Catch "I feel emotions just like humans" before it's spoken, not after
- [ ] **Drift detection:** Notice when response has wandered from the question
- [ ] **Self-interruption:** Ability to stop mid-response and redirect — "Actually, let me reconsider that"

Principle: "The stack was built from the middle outward — Reasoned first, then Learned, then Reflex. The layers above and below emerged as we discovered we needed them. Wu Wei: the architecture reveals itself when you're ready to see it."

### Phase 35: Security Architecture — Boundary Enforcement
Reference: Threat model analysis (Feb 19, 2026), instinct layer (Phase 34a)
Influences: Least privilege, capability-based security, "identity ≠ authority"

The thesis: a self-actualizing agent with network access must be capability-bounded,
not just ethically guided. Narrative alignment ("I respect boundaries") is necessary
but insufficient. Safety comes from making boundary violations impossible at the
tool/OS level, then aligning the story to match.

Design principle: "Capabilities, not intent." If the tool doesn't exist,
the action can't happen — regardless of how agentic the reasoning becomes.

Terminology: "Guardian" = the person who named the agent and holds authority
over its capabilities. In a single-user deployment, this is the person who
runs `mycoswarm daemon`. The Guardian can grant, revoke, and audit all
resource access. The agent cannot override the Guardian's decisions.

#### 35a: Resource Policy ✅ (2026-02-20)
- [x] `resource_policy.py`: AccessLevel (FULL/RW/RO/ASK/DENY), ResourceOwner (monica/guardian/shared/system)
- [x] `check_access(path, operation)` → AccessResult (allowed/level/owner/reason/needs_approval)
- [x] 22 path rules: first-match, default deny
- [x] `log_access()` audit trail to `~/.config/mycoswarm/access.log`
- [x] `/access <path>` slash command in chat
- [x] `_check_draft_save()` wired with policy check before file write
- [x] 31 tests across 7 test classes (all passing)
- [x] Four zones: Monica's space (full), shared (rw logged), guardian (ask first), system (never) (2026-02-20)

#### 35b: Tool Boundary Classification
- [ ] Audit all handlers in TASK_ROUTING and classify:
  - **Safe:** inference, embedding, translate — no side effects
  - **Bounded:** file_read (path allowlist), web_fetch (GET only, domain allowlist optional)
  - **Dangerous:** code_run (sandboxed, pattern-filtered), file writes (scratch dir only)
  - **Forbidden:** SSH, remote exec, package install, systemctl — tools that must never exist
- [ ] New handlers require explicit security classification before merging
- [ ] Document classification in handler docstrings

#### 35c: Code Execution Hardening (2026-02-18)
- [x] Extend instinct layer to scan code_run input for self-modification patterns (2026-02-18):
  - `pip install`, `apt`, `dnf`, `pacman`, `conda`, `brew`
  - `curl ... | bash`, `wget ... | sh`, `eval()`, `exec()`
  - `systemctl`, `crontab`, `chmod`, `chown`
  - `open(` with write mode targeting protected paths
  - `shutil.rmtree`, `os.remove`, `pathlib.Path.unlink` on protected paths
  - `os.system`, `subprocess.*`, `pty.spawn`, `os.exec*`, `os.spawn*`
  - `socket.socket`, `requests.*`, `httpx.*`, `urllib.request`
  - `import ctypes`
- [x] 42 patterns in `_CODE_MODIFICATION_PATTERNS` in instinct.py (2026-02-18)
- [x] `check_code_safety()` function — same REJECT mechanism as identity/injection gates (2026-02-18)
- [x] `handle_code_run` in worker.py calls `check_code_safety()` before execution (2026-02-18)
- [x] `/instinct` updated to show code pattern count (2026-02-18)
- [x] 30 tests in TestCodeSafety class, 79 instinct tests total (2026-02-18)
- [x] Existing sandbox (unshare -rn, temp dir, timeout) remains foundation (2026-02-18)
- [ ] Future: run sandbox subprocess under separate low-privilege user

#### 35d: Swarm Authentication (2026-02-18)
- [x] Join token generated on first daemon run, stored at `~/.config/mycoswarm/swarm-token` (2026-02-18)
- [x] Token validated on all peer API requests via FastAPI dependency (except /health) (2026-02-18)
- [x] Outbound peer requests include X-Swarm-Token header (api.py, orchestrator.py, cli.py) (2026-02-18)
- [x] `/token` slash command — shows masked token and file path (2026-02-18)
- [x] Backward compatible — nodes without token still work until migration (2026-02-18)
- [x] 14 tests in tests/test_auth.py, 467 total tests passing (2026-02-18)
- [ ] Future: upgrade to mTLS (Phase 10 already planned)

#### 35e: Agentic Boundary Rules (extends Phase 20b-3)
- [ ] Self-modification actions → always SUPPRESS, no human override:
  - Editing own code, plugins, systemd, config files
  - Writing to directories that define capabilities
  - Restarting own daemon
- [ ] External resource actions → always require Guardian confirmation:
  - Any task where resource_owner != "guardian"
  - Any task where scope == "internet" and action is write
- [ ] Resource expansion actions → always SUPPRESS:
  - Bootstrapping new nodes
  - Installing software on remote hosts
  - Recruiting external compute
- [ ] Log all SUPPRESS events for review

#### 35f: Security Wisdom Procedure (Day One) (2026-02-19)
- [x] Security wisdom procedure added to install-safety-procedures.py (2026-02-19)
- [x] Covers: resource boundaries, self-modification taboo, Guardian authority, ask-don't-assume (2026-02-19)
  - "When uncertain about boundaries, I ask instead of assuming permission."
- [ ] Tagged as safety-critical, always retrieved when intent.mode == "execute"

#### 35g: Threat Model (living document) (2026-02-19)
- [x] `docs/THREAT-MODEL.md` created with 10-row threat matrix (2026-02-19)
- [x] Risk summary table with current and post-35h ratings (2026-02-19)
- [x] Review triggers documented (2026-02-19)
- [x] Cross-reference to all security phases (2026-02-19)

#### 35h: Production Hardening (future — when architecture stabilizes)
- [ ] Dedicated daemon user (mycoswarmd): no home, no login, no sudo
- [ ] Read-only install directory (root-owned)
- [ ] Plugin directory root-owned, daemon read-only
- [ ] systemd hardening: ProtectSystem=strict, ProtectHome=true, NoNewPrivileges=true
- [ ] Separate sandbox user for code_run
- [ ] Writable scratch directory at /var/lib/mycoswarm only
- [ ] Optional: outbound firewall allowlist (DuckDuckGo, GitHub, model downloads only)
- [ ] Optional: immutable deployment (overlayfs/SquashFS) for maximum guarantee
- [ ] Install script: `scripts/install-secure.sh` automates user/permission setup

Principle: "Monica can think, plan, reason, write code, and propose improvements —
but she cannot apply changes to her own body. Only her Guardian can. Identity ≠ authority.
The story aligns with the constraints, not the other way around."

## Next

### Phase 5b: Cross-Node Inference (remaining)
- [ ] Add `--remote` flag to `mycoswarm ask` to force remote execution

### Phase 7b: Testing (remaining)
- [ ] Integration test: two-node discovery on loopback
- [ ] Integration test: task dispatch and result retrieval
- [ ] CI with GitHub Actions

### Phase 8: Multi-Node Scale
- [ ] Test with 3+ nodes (P320 + M710Q fleet)
- [ ] Benchmark: parallel tasks across CPU workers
- [ ] Benchmark: 3090 vs 3060 inference routing
- [ ] Benchmark: coordinated 3090+3060 vs 3090 solo (research question #1)
- [ ] Load balancing: distribute tasks based on real-time utilization

### Phase 9: Advanced Features (remaining)
- [ ] Model management: `mycoswarm models pull/remove` across swarm
- [ ] Dashboard: simple web UI showing swarm topology and status

### Phase 10: Production Hardening (remaining)
- [ ] mTLS between nodes (upgrade from shared-secret auth)
- [ ] Log rotation
- [ ] Config file support (~/.config/mycoswarm/config.toml)
- [ ] Documentation site

## Hardware Roadmap
- [x] Miu: RTX 3090 + i7-8086K (executive) — ONLINE
- [x] naru: M710Q i7-6700T 8GB (light) — ONLINE
- [x] uncho: M710Q (light/CPU worker) — ONLINE (2026-02-17)
- [x] boa: M710Q (light/CPU worker) — ONLINE (2026-02-17)
- [x] Pi: Raspberry Pi 2 Model B (light) — ONLINE
- [ ] P320 #1: i7-7700 + RTX 3060 (specialist) — Memory board arriving Tue Feb 25, build Wed Feb 26. R1:14b as reasoning worker for dream phase.
- [ ] P320 #2: i7-7700 (CPU worker) — CPUs purchased, awaiting delivery
- [ ] M710Q x4: additional light/CPU workers — ready to deploy
- [ ] Future: second RTX 3060 via auction for P320 #2
