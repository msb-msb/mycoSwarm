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
- [x] 129 tests passing total

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

- [ ] **Timing Gate → Calm:** Wu Wei module — should I act now, later, or not at all? The system is comfortable with uncertainty.
- [ ] **Intent Resolution → Curiosity:** Parse ambiguity, check reversibility, clarify when needed. The system asks before it assumes.
- [ ] **Confidence Calibration → Clarity:** Hedge spectrum instead of uniform confidence. The system knows what it doesn't know.
- [ ] **Emotional Trajectory → Compassion:** Rolling state vector from interaction metadata. The system reads the human, not just the query.
- [ ] **Graceful Degradation → Calm + Courage:** Fail gracefully with partial help, not errors. Comfortable saying "I can partially answer this."
- [ ] **Social Field Awareness → Connectedness:** Group dynamics, authority, visibility. Awareness of context beyond the immediate query.
- [ ] **Productive Friction → Courage:** Trust-gated pushback on user decisions. Says "are you sure?" when confidence is low.
- [ ] **Cross-Domain Synthesis → Creativity:** Connect insights across knowledge domains — a Wu Wei passage informs a debugging strategy.
- [ ] **Source Trust → Confidence:** Use verified RAG context over speculation. Trust what's been retrieved.

Design test for every gate: "Does this make the system more curious, clear, calm?" If a gate increases confidence but reduces clarity, it's out of balance — like an IFS part that's taken over.

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
- [ ] Auto-detect source type on ingest: book, transcript, newsletter, working doc, code
- [ ] Apply type-specific chunking strategy per source
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

- [ ] **Curiosity score:** Intent classification accuracy. How often does the system ask vs. assume?
- [ ] **Clarity score:** Grounding score distribution. Hallucination rate over time.
- [ ] **Calm score:** Appropriate deferral rate. Partial-answer quality.
- [ ] **Compassion score:** Interaction quality trend. Frustration-to-resolution ratio.
- [ ] **Creativity score:** Cross-source retrieval diversity. Novel domain combinations in responses.
- [ ] **Confidence score:** Source priority adherence. Citation rate.
- [ ] **Courage score:** Productive friction events. Pushback appropriateness.
- [ ] **Connectedness score:** Session recall accuracy. Cross-session reference quality.
- [ ] **Aggregate Self-energy metric:** Weighted combination — system health at a glance.

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
- [ ] **"Wu Wei and the Art of Not Answering"** — Timing Gate, confidence calibration, Eastern philosophy meets AI design. Cross-disciplinary, potentially viral.
- [x] **"Distributed Wisdom: Running a Thinking Network on $200 Hardware"** — Full mycoSwarm stack, 5-node swarm, privacy-first. Self-hosted/budget AI audience. (2026-02-15)

#### 30b: White Paper
- [ ] **"Cognitive Architecture for Distributed Local AI: Integrating Psychological Models with RAG"** — 15-20 pages, academic-adjacent
- [ ] Structure: Introduction → CoALA/IFS/Wu Wei background → Four memory streams → 8 C's framework → Implementation → Case study (Phase 20 poison cycle) → Evaluation → Future work
- [ ] Target: genuinely novel contribution — nobody has published on IFS-informed AI architecture running on distributed local hardware

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
- [ ] P320 #1: i7-7700 + RTX 3060 (specialist) — CPUs purchased, awaiting delivery
- [ ] P320 #2: i7-7700 (CPU worker) — CPUs purchased, awaiting delivery
- [ ] M710Q x4: additional light/CPU workers — ready to deploy
- [ ] Future: second RTX 3060 via auction for P320 #2
