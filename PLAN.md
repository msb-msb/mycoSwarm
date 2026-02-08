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

### Phase 9d: Research Command (2026-02-08)
- [x] `mycoswarm research "query"` — compound command: CPU workers search, GPU nodes think
- [x] Flow: web_search task → build context from results → inference task with system prompt
- [x] Progress display: "Searching on naru..." → "Thinking with gemma3:27b..."
- [x] Streams final response with live tokens via SSE
- [x] Sources cited in footer with numbered URLs matching [1], [2] refs in response
- [x] Demonstrates parallel swarm advantage: two cheap boxes > one expensive box

## Next

### Phase 5b: Cross-Node Inference (remaining)
- [ ] Add `--remote` flag to `mycoswarm ask` to force remote execution

### Phase 7: Testing
- [ ] Unit tests for hardware detection (mock subprocess/psutil)
- [ ] Unit tests for capability classification (known inputs → expected tiers)
- [ ] Unit tests for orchestrator routing logic
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
- [ ] Embedding handler (for RAG pipelines)
- [ ] File processing handler (read, transform, summarize documents)
- [ ] Model management: `mycoswarm models` — list/pull/remove across swarm
- [ ] Dashboard: simple web UI showing swarm topology and status

### Phase 10: Production Hardening
- [ ] mTLS between nodes (upgrade from shared-secret auth)
- [ ] Systemd service file for daemon auto-start
- [ ] Log rotation
- [ ] Config file support (~/.config/mycoswarm/config.toml)
- [ ] Plugin system for custom task handlers
- [ ] Documentation site

## Hardware Roadmap
- [x] Miu: RTX 3090 + i7-8086K (executive) — ONLINE
- [x] naru: M710Q i7-6700T 8GB (light) — ONLINE
- [ ] P320 #1: i7-7700 + RTX 3060 (specialist) — CPUs purchased, awaiting delivery
- [ ] P320 #2: i7-7700 (CPU worker) — CPUs purchased, awaiting delivery
- [ ] M710Q x4: additional light/CPU workers — ready to deploy
- [ ] Future: second RTX 3060 via auction for P320 #2
