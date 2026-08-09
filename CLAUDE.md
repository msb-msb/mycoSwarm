# mycoSwarm — Project Instructions for Claude Code

## What This Is
mycoSwarm is a distributed AI framework for coordinating local hardware (GPUs, CPUs, Raspberry Pis) into a unified swarm. Zero cloud, zero API keys, zero config. MIT licensed.

## Repository: github.com/msb-msb/mycoSwarm

## Architecture

```
src/mycoswarm/
├── hardware.py       # System detection (GPU, CPU, RAM, disk, Ollama models)
├── capabilities.py   # Node classification (tiers, capabilities, model recommendations)
├── node.py           # Node identity (persistent UUID, broadcast payload)
├── discovery.py      # mDNS auto-discovery via zeroconf (_mycoswarm._tcp.local.)
├── api.py            # FastAPI per-node service (health, status, peers, tasks)
├── daemon.py         # Main daemon (detection + discovery + API + worker)
├── worker.py         # Task execution (pulls from queue, runs handlers)
├── orchestrator.py   # Task routing (matches task type → best node)
└── cli.py            # CLI entry point (detect, daemon, swarm, ping, ask)
```

## Project Philosophy
See MANIFESTO.md for the full argument. The short version: "open source" in AI
often means weights you can download but can't run without renting a datacenter.
mycoSwarm exists to prove that coordinated cheap hardware — the GPUs and mini PCs
people already own — can deliver real AI capability without cloud dependencies.

The key test: **if a student with two old laptops can't participate, the framework
has failed.** Every design decision filters through this. No heavy dependencies,
no mandatory GPU, no configuration that assumes you know networking. A ThinkCentre
from eBay and a borrowed gaming PC should be a working swarm.

## Key Design Principles
- **No center**: Any node can be orchestrator. Roles migrate.
- **Adapt to what you have**: Works from Raspberry Pi to RTX 4090.
- **Zero config**: mDNS discovery, no IP addresses to configure.
- **Security first**: Sandboxed by default, LAN-only, process isolation.
- **Minimum dependencies**: Only zeroconf, psutil, httpx, fastapi, uvicorn.

## Current Test Bed
- Miu: RTX 3090 + i7-8086K, 64GB RAM (executive tier, 19 Ollama models)
- naru: ThinkCentre M710Q, i7-6700T, 8GB RAM (light tier, CPU only)
- More nodes incoming: 2x P320 Towers, 4x more M710Q

## Coding Standards
- Python 3.12+
- Type hints everywhere
- Dataclasses over dicts for structured data
- asyncio for all I/O
- Logging with emoji prefixes (🎯 routing, ⚙️ executing, ✅ completed, ❌ failed, 📡 discovery)
- Keep dependencies minimal — don't add packages without good reason
- Tests go in tests/ directory (pytest)
- Human-readable CLI output with emoji, machine-readable with --json flag

## Important Patterns
- Hardware detection: subprocess for nvidia-smi, psutil for CPU/RAM/disk
- Ollama integration: HTTP to localhost:11434 (never assume it's running)
- Discovery: zeroconf AsyncZeroconf, service type _mycoswarm._tcp.local.
- API: FastAPI binds 0.0.0.0 (all interfaces); the LAN IP is only advertised to peers. Reachability is not interface-dependent — security comes from the swarm token, not the bind address.
- Task flow: CLI/API → TaskQueue → TaskWorker → Handler → TaskResult
- Node IDs: persistent in ~/.config/mycoswarm/node_id, format myco-{12hex}

## What NOT To Do
- Don't add cloud/API dependencies
- Don't require configuration files for basic operation
- Don't break the zero-config promise
- Don't import heavy frameworks (no torch, no transformers)
- Don't rely on the bind address for access control — the daemon intentionally binds 0.0.0.0 (all interfaces) and depends on the swarm token + LAN isolation for security. To confine swarm traffic to a specific fabric, set MYCOSWARM_SWARM_SUBNET (soft-prefer), don't change the bind.
- **Don't `ollama pull` a model just to try it.** Ollama's store is on the 916 GB root filesystem and is the space constraint (it hit 180 GB / 77% before the 2026-08-08 prune). Models go to the LIBRARY as raw GGUFs — see below.

## Model storage: library vs Ollama
**`/media/minotaur/Storage_Disk_1/LLM_repo` is the archive. Ollama is the working set.**
- Pull models there as raw GGUFs (3.2 TB free, one directory per model, README with source repo, quant, exact bytes, sha256, licence).
- Import into Ollama **only if the model will actually be routed** — a binding, a fallback, the gate, or the pipeline. Benchmarking with llama.cpp needs no import.
- **An Ollama build is often NOT the same file as the library GGUF.** Measured: Ollama's `gemma3:4b` is 849 MB larger than bartowski's and is the multimodal (`vision`) build. For benchmarks, reproducing a measurement, or node deployment, **the library copy is canonical** — it has a recorded sha256 and a named source.
- **`ollama rm` does not reclaim blobs for Modelfile-imported models** — only for pulled ones. Measured 2026-08-09: removing two imported tags left 33.82 GiB of orphaned blobs on disk. After removing any imported tag, compare `du -sh /usr/share/ollama/.ollama/models` against the `ollama list` total; if they diverge, find blobs unreferenced by any manifest and delete them by hand (root, they are `ollama`-owned).
- Full policy, the current working set, and the exact tags of everything pruned (for unambiguous recovery): `LLM_repo/README.md`.

## Updating PLAN.md
After completing work, update PLAN.md:
- Move completed items from "Next" to "Done" with date
- Add any new items discovered during implementation
- Keep the status accurate

## Release Checklist
Every version bump must include ALL of these steps:
1. Bump version in BOTH pyproject.toml AND src/mycoswarm/__init__.py (must match!)
2. Update CHANGELOG.md with new version entry
3. Run full test suite (pytest + smoke tests)
4. Verify version: python -c "from mycoswarm import __version__; assert __version__ == 'X.Y.Z'"
5. Build wheel: python -m build
6. Show the user the upload command: `twine upload dist/mycoswarm-X.Y.Z*` (user runs this manually)
7. Create GitHub release: gh release create vX.Y.Z --title "..." --notes "..."
8. Update all swarm nodes (naru, boa, uncho, pi)

Never skip step 4. v0.2.9 shipped with __version__="0.1.8" because __init__.py wasn't synced.
Never skip step 6. Releases without GitHub tags are invisible.

## Running the Project
```bash
cd ~/Desktop/mycoSwarm  # on Miu (workstation)
source .venv/bin/activate
pip install -e .
mycoswarm detect        # test detection
mycoswarm daemon        # run the full daemon
mycoswarm swarm         # check swarm status (daemon must be running)
mycoswarm ping          # ping peers (daemon must be running)
mycoswarm ask "prompt"  # inference via the swarm (daemon must be running)
```
