# Changelog

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
