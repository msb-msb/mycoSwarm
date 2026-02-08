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
- API: FastAPI bound to LAN IP (not 0.0.0.0)
- Task flow: CLI/API → TaskQueue → TaskWorker → Handler → TaskResult
- Node IDs: persistent in ~/.config/mycoswarm/node_id, format myco-{12hex}

## What NOT To Do
- Don't add cloud/API dependencies
- Don't require configuration files for basic operation
- Don't break the zero-config promise
- Don't import heavy frameworks (no torch, no transformers)
- Don't use 0.0.0.0 for network binding — always LAN IP

## Updating PLAN.md
After completing work, update PLAN.md:
- Move completed items from "Next" to "Done" with date
- Add any new items discovered during implementation
- Keep the status accurate

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
