# 🍄 mycoSwarm

**Distributed AI framework — grow your own cluster from whatever hardware you've got.**

mycoSwarm turns forgotten hardware into a collective intelligence. That old GPU in the closet, the retired office PC, the Raspberry Pi in the junk drawer — each one becomes a node in a living network that thinks together.

No cloud. No subscriptions. No data leaving your network. Just your hardware, your models, your rules.

---

## The Questions We're Asking

### Is the sum greater than the parts?

A single RTX 3060 is limited. But three of them on a LAN, coordinated by an intelligent orchestrator — can they outperform a single expensive GPU for real-world tasks? Not synthetic benchmarks. Real work: research, writing, coding, analysis. Nobody has seriously tested the swarm hypothesis with hardware regular people actually own. We intend to.

### Can a home cluster beat Claude Code?

Claude Code is a powerful agent — but it's trapped in a single-model, single-agent, serial execution box. It can't outsource to specialists. It can't parallelize research across multiple nodes. It can't route a coding task to a coding model and a writing task to a writing model simultaneously.

A home cluster has none of those constraints. Parallel execution, specialist routing, an executive brain orchestrating the whole thing — can it match or exceed a $200/month cloud subscription at zero marginal cost?

Let's find out.

### Can this be free and universal?

If a student with two old laptops can't participate, the framework has failed. If a maker in rural India needs a credit card and a cloud account, the framework has failed. Cloud AI requires credit cards, reliable internet, and USD pricing. mycoSwarm requires a computer and a LAN cable.

---

## Why

The AI revolution has a hardware problem. Frontier models need expensive subscriptions or enterprise GPUs. But most people already own enough compute — it's just scattered across devices collecting dust.

A 2021 GPU isn't obsolete — it's an underutilized specialist node. A retired office PC isn't e-waste — it's a web scraping pool, a vector database host, a document processor. A Raspberry Pi isn't a toy — it's a voice interface, a sensor hub, a discovery beacon.

mycoSwarm connects what you already have into something none of those devices could be alone.

---

## Principles

**Adapt to what you have.** The framework doesn't dictate your hardware — it discovers it, measures it, and puts it to work. One machine? Single-node agent. Ten machines? Distributed content factory. Same code, different scale.

**Secure by default.** AI agents that execute code are dangerous. mycoSwarm treats security as foundational, not optional. Every node runs sandboxed, every connection is authenticated, no data leaves your LAN unless you explicitly allow it. See [SECURITY.md](SECURITY.md).

**No center.** Any node can coordinate. If the biggest GPU goes down, a smaller one takes over. The system degrades gracefully, never fails completely.

**Minimum viable node.** A Raspberry Pi is a first-class citizen. Not every node needs a GPU. A CPU-only box hosting a vector database or running web searches is just as valuable as a GPU doing inference.

**Sovereignty.** Your prompts, your data, your models stay on your network. No API keys to revoke, no terms of service changes, no surprise bills.

**Built from scraps.** If it can't be assembled from used parts on eBay, it's not accessible enough.

---

## Architecture

mycoSwarm organizes nodes by what they can do, not what they are.

### Node Capabilities

Every node announces its capabilities to the swarm:

- **GPU Inference** — runs LLM models (reports VRAM, loaded models, utilization)
- **CPU Inference** — runs tiny models (1.5B-3B quantized)
- **CPU Worker** — web scraping, document processing, file conversion, parsing
- **Storage** — file serving, vector database (ChromaDB), artifact hosting
- **Coordinator** — discovery, health monitoring, task routing
- **Edge** — audio capture, sensor input, mobile client, physical I/O

A single machine can offer multiple capabilities. Roles aren't fixed — they're discovered and reassigned dynamically.

### Topology Patterns

**Solo** — One machine, everything local. Personal AI assistant, no network dependency.

**Desk** — 2-3 machines on a LAN. GPU nodes run models, CPU nodes handle research and storage. The sweet spot for home use.

**Lab** — 5-10 nodes. University lab, hackerspace, or small team pooling resources.

**Mesh** — Geographically distributed nodes over VPN. Communities sharing compute across locations.

### The Orchestrator

The orchestrator is a role, not a fixed service. Any capable node can fill it.

- Discovers peers via mDNS/Avahi on the LAN
- Collects capability announcements from each node
- Routes tasks based on current load and capability
- Redistributes work when nodes join or leave
- Migrates to another node if the current coordinator goes down

### Example: A Home Cluster

```
┌─────────────────────────┐
│   GPU Node (RTX 3090)   │ ← Executive: planning, synthesis
│   Qwen 32B, Gemma 27B  │ ← Runs orchestrator
└────────────┬────────────┘
             │ 1 GbE LAN
     ┌───────┼───────┐
     │       │       │
     ▼       ▼       ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│ GPU Node│ │ GPU Node│ │ CPU Pool  │
│ RTX 3060│ │ RTX 3060│ │ ThinkCntr│
│ 14B mdls│ │ 14B mdls│ │ scraping │
│ research│ │ drafting│ │ ChromaDB │
│ fact-chk│ │ coding  │ │ storage  │
└─────────┘ └─────────┘ └──────────┘
     ┌────────────┐
     │ Edge Nodes  │
     │ RPi: voice  │
     │ RPi: dash   │
     │ Phone: app  │
     └────────────┘
```

### Example Workflow: Parallel Article Engine

One of the first target workflows — distributed content creation vs serial cloud agents:

1. **User submits topic** → Orchestrator receives request
2. **Executive (3090) plans** → Outline, research queries, section assignments
3. **Parallel research (3060s + CPU pool)** → Multiple search/summarize tasks run simultaneously
4. **Parallel drafting (all GPU nodes)** → Each node drafts assigned sections
5. **Fact-checking (3060s)** → Claims verified against research corpus in parallel
6. **Synthesis (3090)** → Executive stitches sections, normalizes tone, polishes

A single cloud agent does steps 2-6 serially. mycoSwarm does them in parallel.

---

## What Can Your Hardware Run?

For model recommendations by GPU, VRAM tier, and budget, see [InsiderLLM.com](https://insiderllm.com) — practical local AI guides from someone who figured it out on real hardware.

- [VRAM Requirements for Local LLMs](https://insiderllm.com/guides/vram-requirements-local-llms/)
- [12GB VRAM — What Can You Run?](https://insiderllm.com/guides/what-can-you-run-12gb-vram/)
- [24GB VRAM — What Can You Run?](https://insiderllm.com/guides/what-can-you-run-24gb-vram/)
- [Best Local Coding Models](https://insiderllm.com/guides/best-local-coding-models-2026/)
- [CPU-Only LLMs: What Actually Works](https://insiderllm.com/guides/cpu-only-llms-what-actually-works/)
- [GPU Buying Guide for Local AI](https://insiderllm.com/guides/gpu-buying-guide-local-ai/)
- [Best GPU Under $300](https://insiderllm.com/guides/best-gpu-under-300-local-ai/)

---

## Requirements

**Minimum (Solo mode):**
- Any Linux machine (Ubuntu/Debian recommended)
- Python 3.11+
- 8GB RAM
- Any Ollama-supported model

**Recommended (Desk mode):**
- 1x GPU node (RTX 3060 or better, 12GB+ VRAM)
- 1-4x CPU nodes (any x86_64 with 8GB+ RAM)
- 1 GbE LAN connecting all nodes

**No cloud accounts, API keys, or subscriptions required.**

---

## Roadmap

### Phase 1 — Foundation
- [ ] Node daemon: capability announcement, health reporting
- [ ] Discovery: mDNS-based peer finding on LAN
- [ ] Orchestrator: basic task routing to available nodes
- [ ] Security baseline: sandboxing, authentication, LAN-only defaults
- [ ] Single-node mode: works as a standalone agent

### Phase 2 — Distributed Inference
- [ ] Multi-node model routing: send tasks to the right GPU
- [ ] Parallel research: distribute search + summarization across nodes
- [ ] Load balancing: route based on current VRAM/CPU usage
- [ ] Graceful degradation: reassign orchestrator if primary goes down
- [ ] Encrypted inter-node transport

### Phase 3 — Workflows
- [ ] Article engine: parallel research → outline → draft → fact-check → polish
- [ ] Model comparison: same prompt to multiple models, compare output
- [ ] Skill system: pluggable tools any node can execute
- [ ] Voice pipeline: mic → Whisper → agent → TTS across edge + GPU nodes
- [ ] Benchmarks: real-world task timing vs Claude Code / cloud agents

### Phase 4 — Community
- [ ] Topology templates: "I have X, Y, Z — here's your optimal config"
- [ ] Shared skill library
- [ ] Mesh networking over VPN
- [ ] Hardware compatibility database (community benchmarks)
- [ ] Auto-recommend models based on detected hardware

---

## Security

Security is a core principle. See [SECURITY.md](SECURITY.md) for the full model.

The short version:
- **Sandboxed by default.** Agents cannot access files outside their workspace.
- **LAN-only by default.** No public interfaces, no outbound internet without explicit config.
- **Authenticated peers.** Nodes verify each other before accepting work.
- **Unprivileged execution.** Agent processes run as dedicated users with minimal permissions.
- **Auditable.** Small, readable Python you can verify yourself. All tool calls logged.
- **No cloud dependency.** No API keys to leak, no tokens to revoke.

---

## Philosophy

mycoSwarm is named after mycelium — the fungal network that connects a forest underground. It doesn't force its way through the soil. It finds the path of least resistance, connects what's already there, and makes the whole system stronger than any individual part.

The project exists because AI should be accessible to anyone with a computer, not just those who can afford cloud compute. The hardware already exists — it just needs software that knows how to use it.

---

## License

MIT — because freedom means freedom.

## Contributing

This project is just getting started. If you've got old hardware and want to help build the future of distributed local AI, open an issue or submit a PR.

Every node counts. 🍄
