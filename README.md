# 🍄 mycoSwarm

**Distributed AI for everyone. Turn forgotten hardware into a thinking network.**

mycoSwarm connects your machines — old laptops, mini PCs, Raspberry Pis, GPU workstations — into a single AI swarm. No cloud. No API keys. No data leaves your network.

```bash
curl -fsSL https://raw.githubusercontent.com/msb-msb/mycoSwarm/main/scripts/install.sh | bash
mycoswarm chat
```

That's it. Two commands. You're running local AI.

---

## What It Does

**One machine?** Chat with local models instantly — no daemon, no config.

**Multiple machines?** They find each other automatically via mDNS, share capabilities, and route tasks to the right hardware. A $50 mini PC can chat with a 27B model running on a GPU across the room.

The weakest machine in the swarm gets access to the strongest model.

### Real Example: 5-Node Swarm

| Node | Hardware | Cost | Role |
|------|----------|------|------|
| Miu | RTX 3090, 64GB RAM | ~$850 (used) | GPU inference — runs 27B models |
| naru | Lenovo M710Q, 8GB RAM | $50 | Web search, file processing |
| uncho | Lenovo M710Q, 8GB RAM | $50 | Web search, coordination |
| boa | Lenovo M710Q, 8GB RAM | $50 | Web search, code execution |
| raspberrypi | Raspberry Pi 2, 1GB RAM | $35 | Search, lightweight tasks |

Total: ~$1,035. Zero monthly fees.

---

## Features

**Chat with memory** — Persistent facts and session history across conversations. Your AI remembers what you tell it.

**Research** — Ask a question, the swarm plans multiple searches, distributes them across CPU workers in parallel, and synthesizes a cited answer on the GPU. Faster than any single machine.

**Document library (RAG)** — Drop files into `~/mycoswarm-docs/`. The swarm indexes them and answers questions about your documents with citations.

**Agentic tool routing** — The model automatically decides when it needs web search or document lookup, shows you what it's doing, and uses the results. No manual tool selection.

**Honest AI** — When it doesn't know something, it says so. No hallucinated weather forecasts or fabricated facts.

**Plugin system** — Drop a folder into `~/.config/mycoswarm/plugins/` and your node advertises a new capability. No core code changes.

---

## Install

### Quick Start (Linux or macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/msb-msb/mycoSwarm/main/scripts/install.sh | bash
mycoswarm chat
```

The installer detects your OS, installs Python and Ollama if needed, pulls a model sized for your RAM, and runs hardware detection.

### Manual Install

```bash
pip install mycoswarm
mycoswarm chat
```

Requires [Ollama](https://ollama.ai) running with at least one model pulled.

### macOS (Apple Silicon)

```bash
brew install ollama
ollama serve &
ollama pull gemma3:27b  # or gemma3:4b for 8GB Macs
pip install mycoswarm
mycoswarm chat
```

Apple Silicon unified memory is detected automatically — an M1 with 16GB can run 14B+ models.

### Raspberry Pi

Works on Pi 2 and newer. pymupdf (PDF support) is optional — if it fails to build on ARM, PDF reading is disabled but everything else works.

```bash
sudo apt install -y python3-venv git
git clone https://github.com/msb-msb/mycoSwarm.git
cd mycoSwarm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mycoswarm detect
```

Pi nodes can't run inference (no GPU, limited RAM) but contribute as web search workers, file processors, and coordinators.

---

## Growing the Swarm

Single-node mode works out of the box. When you're ready for more:

### Start the Daemon

```bash
mycoswarm daemon
```

Or install as a service (Linux):

```bash
sudo cp scripts/mycoswarm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mycoswarm
```

### Add Another Machine

Install mycoSwarm on the second machine, start the daemon. That's it. mDNS handles discovery — no IP addresses to configure, no config files to edit. Within seconds:

```bash
mycoswarm swarm
```

Shows both nodes, their capabilities, and available models.

### How Routing Works

The orchestrator scores each node for each task type:

- **Inference** → GPU nodes (highest VRAM wins)
- **Web search / file processing** → CPU workers (distributed round-robin)
- **Embeddings** → Nodes running Ollama with embedding models
- **Code execution** → CPU workers (sandboxed subprocess)

Tasks go to the best available node. If that node fails, the orchestrator retries on the next candidate. Executive (GPU) nodes are reserved for inference — they won't waste cycles on web searches when CPU workers are available.

---

## CLI Commands

| Command | What It Does |
|---------|-------------|
| `mycoswarm chat` | Interactive chat with memory, tools, and document search |
| `mycoswarm ask "prompt"` | Single question, streamed response |
| `mycoswarm research "topic"` | Parallel web search → synthesized answer with citations |
| `mycoswarm rag "question"` | Answer from your indexed documents |
| `mycoswarm search "query"` | Raw web search results |
| `mycoswarm library ingest [path]` | Index files for document search |
| `mycoswarm library list` | Show indexed documents |
| `mycoswarm detect` | Show hardware and capabilities |
| `mycoswarm swarm` | Swarm overview — all nodes and status |
| `mycoswarm models` | All models across the swarm |
| `mycoswarm plugins` | Installed plugins |
| `mycoswarm memory` | View and manage stored facts |
| `mycoswarm daemon` | Start the swarm daemon |

### Chat Slash Commands

| Command | What It Does |
|---------|-------------|
| `/remember <fact>` | Store a persistent fact |
| `/memories` | Show all stored facts |
| `/forget <n>` | Remove a fact by number |
| `/rag <question>` | Search documents and answer |
| `/library` | Show indexed documents |
| `/auto` | Toggle agentic tool routing on/off |
| `/model` | Switch model |
| `/clear` | Reset conversation |
| `/quit` | Save session and exit |

---

## Architecture

```
src/mycoswarm/
├── hardware.py      # GPU/CPU/RAM/disk/Ollama detection (Linux, macOS, ARM)
├── capabilities.py  # Node classification — tiers, capabilities, model limits
├── node.py          # Persistent node identity (UUID survives restarts)
├── discovery.py     # mDNS auto-discovery, peer health tracking
├── api.py           # FastAPI service — health, status, peers, tasks, SSE streaming
├── daemon.py        # Main daemon — detection + discovery + API + worker + orchestrator
├── worker.py        # Task handlers — inference, search, embedding, files, code, translate
├── orchestrator.py  # Task routing — scoring, retry, load balancing, inflight tracking
├── plugins.py       # Plugin loader — scan ~/.config/mycoswarm/plugins/
├── solo.py          # Single-node mode — direct Ollama, agentic classification
├── library.py       # Document library — chunking, embeddings, ChromaDB, RAG
├── memory.py        # Persistent memory — facts, session summaries, prompt injection
└── cli.py           # All CLI commands and interactive chat
```

### Node Tiers

| Tier | Example Hardware | Role |
|------|-----------------|------|
| **EXECUTIVE** | RTX 3090 workstation | GPU inference, orchestration |
| **SPECIALIST** | RTX 3060 desktop | GPU inference (smaller models) |
| **LIGHT** | Lenovo M710Q, Raspberry Pi | Web search, file processing, coordination |
| **WORKER** | Any CPU-only machine | Distributed task execution |

### Discovery

Nodes broadcast via mDNS (`_mycoswarm._tcp.local.`). No central server, no configuration. Plug in a machine, start the daemon, the swarm grows.

### Task Flow

```
User asks question on Node A
  → Node A checks: can I handle this locally?
    → Yes: execute locally
    → No: orchestrator scores all peers
      → Dispatch to best peer
      → Stream response back to Node A
```

---

## Plugins

Extend the swarm without touching core code. Drop a directory into `~/.config/mycoswarm/plugins/`:

```
~/.config/mycoswarm/plugins/
└── my_summarizer/
    ├── plugin.yaml
    └── handler.py
```

**plugin.yaml:**
```yaml
name: my_summarizer
task_type: summarize
description: Summarize text by extracting key points
capabilities: cpu_worker
```

**handler.py:**
```python
async def handle(task):
    text = task.payload.get("text", "")
    # Your logic here
    return {"summary": summarized_text}
```

Restart the daemon. The node advertises the new capability. Other nodes can route `summarize` tasks to it.

---

## Document Library

Drop files into `~/mycoswarm-docs/` and index them:

```bash
mycoswarm library ingest
```

Supports: PDF, Markdown, TXT, HTML, CSV, JSON.

Files are chunked, embedded (via Ollama), and stored in ChromaDB. Ask questions:

```bash
mycoswarm rag "what does the architecture section describe?"
```

Or use `/rag` in chat for inline document search.

---

## The Manifesto

Named after mycelium — the underground network connecting a forest. It doesn't centralize. It finds what's available and connects it.

**If a student in Lagos with two old laptops can't participate, the framework has failed.**

No cloud dependencies. No API keys. No expensive hardware requirements. Every node counts.

---

## What's Next

- **Dashboard** — Web UI showing swarm topology, active tasks, node health
- **Agentic planner** — LLM generates multi-step plans and executes them across the swarm
- **mTLS security** — Encrypted, authenticated inter-node communication
- **Config files** — `~/.config/mycoswarm/config.toml` for persistent settings
- **PyPI publishing** — `pip install mycoswarm` from anywhere
- **Mesh networking** — Connect swarms across the internet via VPN

---

## Contributing

mycoSwarm is MIT licensed. Contributions welcome.

```bash
git clone https://github.com/msb-msb/mycoSwarm.git
cd mycoSwarm
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v  # 94 tests, all offline
```

---

**Built with experience, not hype.** [InsiderLLM](https://insiderllm.com)
