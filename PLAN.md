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
- [x] Unit tests for document library/RAG — 23 tests
- [x] Unit tests for persistent memory — 21 tests
- [x] Unit tests for agentic chat classification — 9 tests
- [x] 94 tests passing total

### Phase 19: RAG Level 2 Improvements
- [x] Metadata on chunks: source filename, section heading, file date, document type
- [x] Text cleaning before chunking: strip headers/footers, normalize whitespace, remove boilerplate
- [x] Embedding model version tracking: store model name in metadata, warn on mismatch
- [x] Session-as-RAG: index session summaries into ChromaDB for semantic memory search
- [ ] Hybrid search: BM25 keyword matching alongside vector similarity
- [ ] Re-ranking: LLM ranks retrieved chunks by relevance before generating answer
- [ ] Auto-update pipeline: detect changed files in ~/mycoswarm-docs/ and re-index
- [ ] RAG eval set: gold standard questions with known answers to measure quality

### Phase 20: Human Gap Architecture (Pre-Processing Gates)
- [ ] Timing Gate: Wu Wei module — should I act now, later, or not at all?
- [ ] Intent Resolution: parse ambiguity, check reversibility, clarify when needed
- [ ] Confidence Calibration: hedge spectrum instead of uniform confidence
- [ ] Emotional Trajectory: rolling state vector from interaction metadata
- [ ] Graceful Degradation: fail gracefully with partial help, not errors
- [ ] Social Field Awareness: group dynamics, authority, visibility
- [ ] Productive Friction: trust-gated pushback on user decisions

### Phase 21: Memory Architecture
Reference: docs/ARCHITECTURE-MEMORY.md

#### 21a: Fact Lifecycle Tags
- [ ] Add `type` field to facts: preference, fact, project, ephemeral
- [ ] Different retention rules per type
- [ ] Staleness detection: flag facts unreferenced in N sessions

#### 21b: Decay Scoring
- [ ] Session memories get recency-weighted scores
- [ ] Referenced-again sessions get boosted
- [ ] Old unreferenced sessions decay in retrieval priority
- [ ] "Forgetting as technology"

#### 21c: Mode-Aware Retrieval
- [ ] Connect intent gates (Phase 20) to memory retrieval
- [ ] Brainstorm/planning → broad retrieval, more results
- [ ] Execution → narrow retrieval, precise constraints

#### 21d: Procedural Memory
- [ ] New memory type: exemplar store
- [ ] "How we solved X before" — success/fail patterns
- [ ] Stored separately from episodic and factual memory

#### 21e: Two-Stage Document Ingest
- [ ] Extract structured entities/facts first, then chunk for semantic
- [ ] Cross-reference RAG results against facts store

#### 21f: Memory Review & Pruning
- [ ] Periodic prompts to review stale facts
- [ ] Fact versioning with change history
- [ ] Dashboard UI for memory management

### Phase 22: RAG Architecture
Reference: docs/ARCHITECTURE-RAG.md

#### 22a: Hybrid Search (Level 2)
- [ ] BM25 keyword search alongside vector semantic search
- [ ] Combine scores for better relevance
- [ ] "Find the document where we decided X" queries

#### 22b: Re-Ranking
- [ ] LLM re-ranks retrieved chunks by relevance to query
- [ ] Filter out noise before injecting into context

#### 22c: RAG Eval Set
- [ ] Gold standard question/answer pairs
- [ ] Measure retrieval quality over time
- [ ] Regression testing for RAG changes

#### 22d: Auto-Update Pipeline
- [ ] File watcher for changed documents
- [ ] Re-ingest modified files automatically
- [ ] Staleness detection on indexed documents

#### 22e: Agentic RAG (Level 4)
- [ ] Multi-step reasoning over retrieved documents
- [ ] Reformulate query and search again if first pass insufficient
- [ ] Chain-of-retrieval for complex questions

#### 22f: Graph RAG
- [ ] Entity relationship extraction across documents
- [ ] Knowledge graph for cross-document connections
- [ ] "How does X relate to Y" queries

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
