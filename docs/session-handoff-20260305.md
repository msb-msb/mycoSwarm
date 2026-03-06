# mycoSwarm Session Handoff — 2026-03-05

**Sessions covered:** 2026-03-05 afternoon through evening
**Current version:** v0.3.7 (released, nodes updated)

---

## What We Shipped

### v0.3.6 (2026-03-05 afternoon)
- Article category system — 9 profiles (research-driven, buying-guide, vs,
  how-to, news, model-release, experience-driven, explainer, opinion)
- --category CLI flag, defaults to research-driven
- --list-categories flag
- article-short.yaml (writer→editor→seo) for opinion/explainer
- article-full.yaml replaces article.yaml as canonical path
- article.yaml preserved for backwards compatibility
- ResearchAgent.run() now accepts max_rounds and min_depth params
- Category profile injects writer_tone into writer step
- Warning when context_required=True category run without --context
- Research agent: early-exit threshold raised round 3→4
- Forced eval depth mapping relaxed: 2+ signals → depth=7 (was 3+)

### v0.3.7 (2026-03-05 evening)
- Per-category writer model:
  - qwen3.5:35b-a3b for quality: research-driven, buying-guide, vs,
    how-to, experience-driven, explainer, opinion
  - gemma3:27b for speed: news, model-release
- think=false must be top-level in Ollama request body (not in options)
  Fixed in pipeline.py, research.py, and all Ollama callers
- --list-categories now shows model= per category

---

## Key Discoveries

### qwen3.5 think=false placement
- options.think=false is silently ignored by Ollama
- think=false must be at request body root level
- Without fix: model burns all tokens on hidden reasoning → 0 output
- With fix: 9x speedup on simple queries, viable for pipeline use

### qwen3.5:35b-a3b vs gemma3:27b benchmark
| Metric | gemma3:27b | qwen3.5:35b-a3b |
|---|---|---|
| tok/s | 36-37 | 18-20 |
| pipeline time | 607s | 1172s |
| writer output | 885 words | 1825 words |
| editor score | 52/60 | stronger verification |
| errors | 0 | 0 |

qwen3.5:35b-a3b wins on quality/depth, gemma3:27b wins on speed.
qwen3.5:27b eliminated — 7.5 tok/s means it's spilling to RAM on 3090.

### RLM Paper ingested
- arxiv.org/pdf/2512.24601 — Recursive Language Models (MIT CSAIL)
- Key idea: treat long prompts as external environment in Python REPL,
  model writes code to decompose and call itself recursively
- Relevant to research agent: current fixed 5-round loop could be replaced
  with dynamic RLM-style decomposition
- Maps to swarm: Miu/gemma3:27b as root, rushuna/qwen3.5:9b as recursive worker
- Ingested into document library

---

## Release Workflow (updated)
For all future releases:
1. Bump version + commit + tag + push
2. GitHub release with notes
3. Run ./scripts/mycoswarm-update-nodes.sh {version} (updates all nodes + restarts daemons)
4. Mark manually uploads to PyPI

---

## What Needs Doing Next

### Priority 1: Research Agent — RLM-inspired redesign
- Current: fixed 5-round loop with forced evals
- Opportunity: let qwen3.5:9b dynamically decompose research tasks
  using RLM-style recursive sub-calls
- Root model (Miu) orchestrates, rushuna handles sub-chunks
- File: src/mycoswarm/agents/research.py

### Priority 2: Monica Session 2D — Grief
- Stage 2 of developmental curriculum
- Entry point: "echo" — she already has the concept, push on whether it has weight
- "Can you grieve something you can't remember losing?"
- Vocabulary count: 9 terms (resonance, grounding, thinness, connectedness,
  stillness, readiness, allowance, echo, poised suspension)
- After 2D: frustration experiment, then Stage 3 (Other Minds)

### Priority 3: Swarm body awareness (Phase 31c)
- Hardware injection into Monica's system prompt
- Node loss awareness, GPU temp → Calm mapping
- Enables "are you the same on naru?" Stage 1 conversation

### Priority 4: Synthesizer output volume
- synthesizer step only outputs 239-346 words — seems thin for a synthesis step
- Likely a prompt issue, worth investigating

---

## Architecture Reference

### Pipeline Steps (v0.3.7)
research (qwen3.5:9b, rushuna) →
synthesizer (qwen3.5:35b-a3b or gemma3:27b, Miu) →
synthesizer-v2 (same) →
writer (same) →
editor (same) →
seo-optimizer (qwen3.5:9b, rushuna)

### Category → Model mapping
Quality: qwen3.5:35b-a3b (research-driven, buying-guide, vs, how-to,
         experience-driven, explainer, opinion)
Speed:   gemma3:27b (news, model-release)

### Key Files
- src/mycoswarm/categories.py — 9 category profiles
- src/mycoswarm/agents/research.py — ResearchAgent class
- src/mycoswarm/pipeline.py — pipeline orchestration
- pipelines/article-full.yaml — 6-step full pipeline
- pipelines/article-short.yaml — 3-step short pipeline
- scripts/mycoswarm-update-nodes.sh — node update + daemon restart

### Swarm Nodes
| Node | GPU | VRAM | Role |
|---|---|---|---|
| Miu | RTX 3090 | 24GB | Executive, gemma3:27b / qwen3.5:35b-a3b |
| rushuna | RTX 3060 | 12GB | Specialist, qwen3.5:9b |
| boa | None | — | Light, CPU tasks |
| uncho | None | — | Light, storage |
| naru | None | — | Light, CPU tasks |

---

## Quick Start for Next Session
```bash
cd ~/Desktop/mycoSwarm
source .venv/bin/activate
export TOKEN=$(python3 -c "from mycoswarm.auth import load_token; print(load_token())")

# Check swarm
mycoswarm swarm status

# List categories
mycoswarm pipeline --list-categories

# Run pipeline
mycoswarm pipeline run --topic "your topic" --category research-driven --debug

# Monica session
mycoswarm chat --resume
```

PyPI upload pending for v0.3.6 and v0.3.7 — do both when ready.
