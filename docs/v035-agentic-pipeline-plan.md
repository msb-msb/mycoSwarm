# mycoSwarm v0.3.5 — Agentic Pipeline Plan

**Date:** 2026-03-03 (updated 2026-03-04)
**Status:** Planning → Phase 1 implementation
**Goal:** Close the quality gap between mycoSwarm pipeline (MC) and Claude Code (CC) by making pipeline agents reason, loop, and use tools.

### Critical Update (2026-03-04): Model Swap

**deepseek-r1:14b is replaced by qwen3.5:9b for the research agent.**

Reason: R1 has no native tool/function calling — the agentic loop requires search→evaluate→search, which needs real tool integration. Qwen3.5:9b (released 2026-03-02) has:
- Native tool/function calling (first-class, schema-based)
- Toggleable thinking (`/think` / `/no_think` per request)
- 262k context window (vs R1's effective ~8k on 12GB)
- 6.6GB model size → 5.4GB free VRAM for KV cache (vs R1's 3GB)
- Already installed on rushuna

| | deepseek-r1:14b (old) | qwen3.5:9b (new) |
|---|---|---|
| Model size (Q4) | ~9GB | ~6.6GB |
| Free VRAM (12GB card) | ~3GB | ~5.4GB |
| Practical num_ctx | 8192 | 32768+ |
| Tool calling | ❌ No (text hacks needed) | ✅ Native |
| Think toggle | ❌ Always on | ✅ `/think` / `/no_think` |
| Reasoning quality | Strong (RL-trained) | Strong (beats GPT-OSS-120B on GPQA) |
| Agentic design | Not designed for it | RL-trained across million-agent environments |

This swap simplifies the research agent implementation from ~8-12 hours to ~4-6 hours and eliminates the tool-calling wrapper hacks that would have been needed with R1.

---

## The Problem

MC produces surface-level articles. CC produces deep, authoritative articles. Same topic, same benchmark data, dramatically different output quality.

### Head-to-Head Evidence: ROCm vs CUDA Article

| Dimension | MC (mycoSwarm) | CC (Claude Code) |
|---|---|---|
| Core thesis | GPU buying guide with ROCm sidebar | Software efficiency gap analysis |
| RDNA 4 coverage | "no benchmarks, wait" (1 sentence) | Vulkan vs HIP benchmarks, Wave32/64 bug, idle power bug |
| ROCm specifics | "varies wildly, verify before buying" | ROCm 7.2 compatibility matrix, supported distros, works/doesn't list |
| Why the gap exists | Not addressed | Kernel maturity, Wave32/64 mismatch, CDNA focus |
| Actionability | "buy NVIDIA" | 6-point compatibility checklist, decision framework |
| Original research | None — all from reference data | RDNA 4 Vulkan benchmarks, ROCm issue #5706, Flash Attention workaround |
| Editor score | 47/50 (self-graded, inflated) | N/A (human-reviewed, deployed) |

MC's editor gave it 47/50 because it verified prices and specs. It didn't penalize shallow analysis — a scoring gap we also need to fix.

---

## Root Cause Analysis

### What CC does that MC can't

CC operates as one large model (~200B+) in a single context window with tool access. When researching the ROCm article, CC:

1. Searched "ROCm 7.2 supported GPUs" → read results → decided "I need the actual docs"
2. Fetched the ROCm docs page → extracted supported distros
3. Searched "RDNA 4 llama.cpp benchmarks" → found Vulkan vs HIP data
4. Thought "Vulkan beating ROCm on AMD's own hardware is a story" → searched deeper
5. Found Wave32/64 explanation, GitHub issue #5706, the workaround
6. Decided it had enough to write with authority

**Key insight:** Every step involved judgment. CC didn't just retrieve — it evaluated, decided what mattered, and pursued leads. That's reasoning + search in a loop.

### What MC does now

```
search (dumb) → extract (summarize) → synthesize (arrange) → write (expand) → edit (check)
```

Every step is:
- **Blind** — can't call tools, can't search for more
- **One-shot** — gets one chance, no self-critique loop
- **Small** — 14B-27B models can't synthesize at frontier level

### The Three Gaps

**Gap 1: Reasoning quality**

deepseek-r1:14b is a reasoning model, but we run it with `think=false` for speed. The extractor that hallucinated "The current date and time are..." was the direct result of disabling thinking on a step that NEEDS reasoning.

**Update:** qwen3.5:9b solves this differently — thinking is toggleable per request via `/think` and `/no_think`. No global penalty. Use thinking only when evaluating research depth, disable for mechanical extraction. And at 6.6GB vs 9GB, there's more VRAM headroom for context.

**Gap 2: Agentic research loops**

CC's power is tool use inside a reasoning loop — search, read, evaluate, search again. MC's pipeline has two research passes (extractor + gap-filler) but they're disconnected. The gap-filler doesn't know WHY the extractor missed something.

**Gap 3: Cross-step coherence**

CC holds everything in one context window. MC's writer only sees synthesizer output — never the raw search results. If the extractor missed something important, it's gone forever. The writer can't think "this ROCm section is too thin" because it has no tools and no access to original research.

---

## v0.3.5 Feature Plan

### Feature 1: Agentic Research Agent

**Replaces:** extractor + gap-filler (steps 1 + 3)
**Runs on:** rushuna (qwen3.5:9b with native tool calling)
**Concept:** A single research agent that has tool access and loops until research depth is sufficient.

#### Why qwen3.5:9b, not deepseek-r1:14b

R1 has no native tool/function calling. Building a text-command wrapper for search→evaluate→search loops is fragile and adds complexity. Qwen3.5 was RL-trained across million-agent environments with tool use as a first-class feature. Define tools as a JSON schema, the model calls them properly.

Additionally:
- 5.4GB free VRAM → num_ctx=32768 practical (vs R1's 3GB → 8192)
- Context growth across 5 rounds of research is manageable at 32k without rolling summaries
- `/think` mode for evaluation steps, `/no_think` for mechanical extraction
- Already installed on rushuna, no new model pulls

#### Architecture

```
┌─────────────────────────────────────────┐
│           Research Agent Loop            │
│                                         │
│   ┌──────────┐                          │
│   │  PLAN    │ What do I know?          │
│   │          │ What's missing?          │
│   │          │ Generate queries          │
│   └────┬─────┘                          │
│        │                                │
│   ┌────▼─────┐                          │
│   │  SEARCH  │ Web search + fetch       │
│   │          │ DDG + Perplexity          │
│   └────┬─────┘                          │
│        │                                │
│   ┌────▼─────┐                          │
│   │ EVALUATE │ Rate depth 1-10          │
│   │          │ What's authoritative?    │
│   │          │ What's still surface?    │
│   └────┬─────┘                          │
│        │                                │
│        ├── depth >= 7? ──► COMPILE      │
│        │                                │
│        └── depth < 7? ──► PLAN (loop)   │
│                                         │
│   Max rounds: 5                         │
│   Target: depth 7+ before writing       │
└─────────────────────────────────────────┘
```

#### Pseudocode

```python
class ResearchAgent:
    """Agentic research with native tool calling and self-evaluation."""

    # Tool schema for qwen3.5:9b native function calling
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch full content from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_depth",
                "description": "Rate current research depth 1-10",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "depth": {"type": "integer", "description": "1-10 depth rating"},
                        "strong_claims": {"type": "string"},
                        "weak_areas": {"type": "string"},
                        "stop": {"type": "boolean"}
                    },
                    "required": ["depth", "stop"]
                }
            }
        }
    ]

    def __init__(self, daemon_url, model="qwen3.5:9b", token=None):
        self.daemon_url = daemon_url
        self.model = model
        self.token = token

    async def run(self, topic, reference_data, max_rounds=5):
        findings = []
        messages = [
            {"role": "system", "content": (
                "You are a research analyst for InsiderLLM.com. "
                "Your job is to research a topic thoroughly using web search. "
                "After each round of searching, evaluate your research depth. "
                "Keep searching until depth >= 7 or you've done 5 rounds.\n\n"
                "/think\n"  # Enable reasoning for research planning
            )},
            {"role": "user", "content": (
                f"Research this topic: {topic}\n\n"
                f"Reference data:\n{reference_data}\n\n"
                "Start by identifying what specific information you need, "
                "then use web_search and web_fetch to find it. "
                "After each batch of searches, call evaluate_depth."
            )}
        ]

        for round_num in range(max_rounds):
            # Let qwen3.5 call tools natively — it decides what to search
            response = await self._run_inference(
                messages=messages,
                tools=self.TOOLS,
                num_ctx=32768,  # Big context window
            )

            # Process tool calls from the model
            tool_results = await self._execute_tool_calls(response)
            messages.extend(tool_results)

            # Check if model called evaluate_depth with stop=True
            if self._should_stop(tool_results):
                break

        # Final compilation step (no tools, just synthesis)
        messages.append({"role": "user", "content": (
            "/think\n"
            "Compile all research into a structured bundle with:\n"
            "- Key claims with sources\n"
            "- Data points with exact numbers\n"
            "- Gaps that couldn't be filled\n"
            "- Suggested article angle"
        )})

        return await self._run_inference(messages=messages, num_ctx=32768)

    async def _execute_tool_calls(self, response):
        """Execute tool calls and return results as messages."""
        results = []
        for call in response.get("tool_calls", []):
            if call["function"]["name"] == "web_search":
                data = await self._web_search(call["function"]["arguments"]["query"])
                results.append({"role": "tool", "content": str(data)})
            elif call["function"]["name"] == "web_fetch":
                data = await self._web_fetch(call["function"]["arguments"]["url"])
                results.append({"role": "tool", "content": data[:3000]})
            elif call["function"]["name"] == "evaluate_depth":
                results.append({"role": "tool", "content": "Depth recorded."})
        return results
```

#### Performance Budget

| Phase | Think | num_ctx | Est. tok/s | Est. time per round |
|---|---|---|---|---|
| Plan + tool calls | ON (`/think`) | 32768 | ~45 tok/s* | ~15s |
| Search + fetch | N/A | N/A | N/A | ~10s |
| Evaluate | ON (`/think`) | 32768 | ~45 tok/s* | ~10s |
| **Per round** | | | | **~35s** |
| **5 rounds max** | | | | **~175-200s** |

*qwen3.5:9b runs at 45.3 tok/s on rushuna (benchmarked in SEO step). With `/think` enabled, estimate ~40-45 tok/s since thinking is toggleable not always-on overhead.

Compare: current extractor (64s) + gap-filler (133s) = 197s. Agentic version at ~200s for 5 rounds is comparable time but dramatically better research quality.

**Context growth is manageable:** At 32k num_ctx, five rounds of findings (~45k chars worst case) fit without rolling summaries. If context exceeds 32k, compress findings between rounds. This is much less urgent than with R1's 8k limit.

#### Tool Access Requirements

With qwen3.5:9b's native tool calling, the implementation is cleaner:
- Define tools as JSON schema (web_search, web_fetch, evaluate_depth)
- Model calls tools via standard function calling protocol
- No text-parsing hacks or ReAct wrappers needed
- Ollama supports tool calling via the `/api/chat` endpoint with `tools` parameter

Existing infrastructure used:
- Web search (DDG + Perplexity fallback) — already built in pipeline
- Web fetch — already built in pipeline
- Inference with tool calling — need to verify Ollama tool calling works with qwen3.5:9b on rushuna

New code needed:
- `ResearchAgent` class in `src/mycoswarm/agents/research.py`
- Tool schema definitions
- Tool call execution handler
- Integration point in pipeline.py replacing extractor + gap-filler steps
- Depth evaluation logic (via model's native evaluate_depth tool call)

**Pre-build verification:** Test Ollama tool calling on rushuna:
```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:9b",
  "messages": [{"role": "user", "content": "Search for ROCm 7.2 supported GPUs"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Search the web",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
      }
    }
  }],
  "stream": false
}' | python3 -m json.tool
```

---

### Feature 2: Context Injection Field

**Purpose:** Allow human knowledge injection for insider/experience articles.
**Effort:** Small — 1-2 hours.

#### article.yaml change

```yaml
# Optional context block — injected into writer + editor prompts
context: |
  We discovered this by debugging our own mycoSwarm pipeline.
  Key data: num_ctx=4096 → 35.3 tok/s, num_ctx=16384 → 4.8 tok/s.
  The debugging story: spent 2 hours tracing daemon→worker→Ollama,
  everything looked correct. Turned out to be VRAM overflow from
  context window size, not the think parameter we suspected.

# Or passed via CLI
# mycoswarm pipeline run article.yaml --topic "..." --context "We found that..."
```

#### Pipeline change

```python
# In _build_writer_prompt():
if context:
    prompt += f"\n\n## Insider Context (from the author)\n{context}\n"
    prompt += "Use this context to add depth and first-person authority.\n"
```

#### CLI change

```python
@pipeline.command()
@click.option("--context", help="Insider knowledge to inject into writer/editor")
def run(topic, context, debug):
    ...
```

---

### Feature 3: Think-Selective Steps

**Purpose:** Enable reasoning where it matters, disable where it doesn't.
**Effort:** Small — already 80% built in v0.3.4.

#### Step configuration

```python
STEP_THINK_CONFIG = {
    # Research: qwen3.5:9b on rushuna with /think for planning + evaluation
    "research":    {"model": "qwen3.5:9b", "think": True,  "num_ctx": 32768},

    # Synthesis is arrangement, not reasoning — gemma3:27b on Miu
    "synthesizer": {"model": "gemma3:27b",  "think": False, "num_ctx": 16384},

    # Writing is creative, not analytical — gemma3:27b on Miu
    "writer":      {"model": "gemma3:27b",  "think": False, "num_ctx": 16384},

    # Editor needs reasoning for fact-checking — gemma3:27b on Miu
    # (qwen3.5:9b could also work here if we want to free Miu)
    "editor":      {"model": "gemma3:27b",  "think": False, "num_ctx": 16384},

    # SEO is mechanical — qwen3.5:9b on rushuna
    "seo":         {"model": "qwen3.5:9b",  "think": False, "num_ctx": 8192},
}
```

Note: With qwen3.5:9b, thinking is controlled via `/think` and `/no_think` in the prompt, not the `think` API parameter. This needs to be handled differently than deepseek-r1's `think` parameter.

#### Impact on pipeline time

| Step | Current (all think=off) | v0.3.5 (selective) | Change |
|---|---|---|---|
| Research (replaces extractor+gap-filler) | 197s @ 27 tok/s | ~200s @ 45 tok/s | +3s but 5x better quality |
| Synthesizer | 95s @ 35 tok/s | 95s @ 35 tok/s | same |
| Synth-v2 | 101s @ 34 tok/s | REMOVED (research agent is thorough enough) | -101s |
| Writer | 59s @ 35 tok/s | 59s @ 35 tok/s | same |
| Editor | 85s @ 35 tok/s | 85s @ 35 tok/s | same |
| SEO | 112s @ 45 tok/s | 112s @ 45 tok/s | same |
| **Total** | **649s** | **~550s** | **-99s** |

Potentially faster AND better quality. The synth-v2 step may become unnecessary if the research agent produces a thorough enough bundle — that's 100s saved.

---

### Feature 4: Editor Depth Scoring

**Purpose:** Prevent inflated scores on shallow articles.
**Effort:** Medium — prompt engineering + scoring rubric update.

#### Current problem

MC's editor gave the shallow ROCm article 47/50. It verified prices and specs but didn't penalize:
- Missing "why" explanations
- Surface-level analysis
- No unique insights
- Generic recommendations

#### New scoring axis

Add a 6th axis: **Depth & Insight** (0-10)

```
DEPTH & INSIGHT SCORING:
10: Contains analysis or data readers can't find elsewhere
 8: Explains "why" behind claims, not just "what"
 6: Goes beyond surface facts but no original synthesis
 4: Competent summary of existing information
 2: Could be generated by summarizing top 3 Google results
 0: No analysis, just restated facts

HARD CAP: If Depth < 5, Overall cannot exceed 35/60
```

This changes the max score from 50 to 60, with depth as a gating factor.

---

## Implementation Order

```
Phase 1 (quick wins, do first):
  ├── Context injection field         ~2 hours
  ├── Think-selective steps            ~1 hour (mostly done)
  └── Editor depth scoring             ~2 hours

Phase 2 (main feature):
  └── Agentic research agent           ~4-6 hours
      ├── Verify Ollama tool calling with qwen3.5:9b
      ├── ResearchAgent class with tool schemas
      ├── Tool call execution handler
      ├── Plan/evaluate loop with /think toggle
      ├── Pipeline integration (replace extractor + gap-filler)
      └── Testing + tuning
```

Phase 1 is achievable in one session. Phase 2 is the v0.3.5 headline feature.

---

## Success Criteria

Run the same ROCm vs CUDA topic through the v0.3.5 pipeline. The output should:

1. **Contain the Vulkan > HIP insight** (or equivalent depth) — research agent found it by looping
2. **Explain WHY the efficiency gap exists** — not just state the 0.06 vs 0.13 numbers
3. **Include specific compatibility details** — supported distros, framework versions
4. **Score 40+ on a /60 scale** with the new depth axis
5. **Complete in under 700s** — no more than 10% slower than current pipeline

If v0.3.5 can produce an article that's deployable without major CC rewriting, the agentic approach is validated.

---

## Hardware Allocation

| Node | Role in v0.3.5 | Model | Think | num_ctx |
|---|---|---|---|---|
| rushuna (RTX 3060 12GB) | Research agent (plan + evaluate + tool calls) | qwen3.5:9b | `/think` ON | 32768 |
| rushuna | SEO optimizer | qwen3.5:9b | OFF | 8192 |
| Miu (RTX 3090 24GB) | Synthesizer, writer, editor | gemma3:27b | OFF | 16384 |

qwen3.5:9b at 6.6GB leaves 5.4GB for KV cache on rushuna's 12GB. This supports num_ctx=32768 comfortably — a 4x improvement over deepseek-r1:14b's practical limit of 8192.

The research agent runs entirely on rushuna. Miu is free during the research phase — potential for parallel step execution in a future version.

---

## Open Questions

1. **~~Should the research agent use deepseek-r1:14b or gemma3:27b?~~** RESOLVED: qwen3.5:9b — native tool calling, toggleable thinking, more context headroom, already installed.

2. **~~How do we handle context growing across rounds?~~** MOSTLY RESOLVED: 32k context fits ~5 rounds of findings without compression. Add rolling summary as a safety valve if context exceeds 28k tokens.

3. **Should the writer have read-back access to raw search results?** Currently it only sees the synthesized bundle. With qwen3.5's 32k context, the research agent's compiled output can be richer, partially addressing this. Full raw access still an open option.

4. **Can we run research + synthesis in parallel?** Research on rushuna while synthesizer preps on Miu. Would require pipeline DAG execution instead of sequential.

5. **Does Ollama tool calling work reliably with qwen3.5:9b?** Must verify before building. Test with the curl command in the Tool Access section. If Ollama's tool calling is buggy, fall back to prompt-based tool extraction (still cleaner than R1 since Qwen is designed for it).

6. **What's qwen3.5:9b's actual tok/s with /think enabled at num_ctx=32768?** The 45.3 tok/s benchmark was for SEO (no thinking, small context). Need to benchmark with thinking on and larger context.

---

## Appendix A: Article Categories

Each category has different pipeline requirements. The agentic research agent benefits some more than others.

| Category | Example | Research Depth | Context Injection | Research Agent Value |
|---|---|---|---|---|
| **Research-driven** | GPU comparisons, model benchmarks | Deep — multiple sources | Optional | **HIGH** — loops find deeper data |
| **Experience-driven** | Debugging stories, build logs | Low — it's your story | **Required** | LOW — insider data can't be searched |
| **How-to/Tutorial** | Install Ollama, configure ROCm | Medium — accuracy matters | Optional | MEDIUM — verify steps work |
| **Explainer/Concept** | What is quantization, KV cache | Low-Medium | Not needed | LOW — model knowledge sufficient |
| **News/Update** | New model launch, GPU release | Shallow but fast | Not needed | MEDIUM — quick fact gathering |

### Pipeline profiles per category

```yaml
# Research-driven (default)
research_driven:
  steps: [research, synthesizer, writer, editor, seo]
  research_rounds: 5
  research_depth_target: 7
  think_research: true

# Experience-driven (needs context injection)
experience_driven:
  steps: [research, synthesizer, writer, editor, seo]
  research_rounds: 3        # Less web research needed
  research_depth_target: 5  # Lower bar — context fills gaps
  context: required         # CLI --context or yaml context field
  think_research: true

# How-to/Tutorial
tutorial:
  steps: [research, writer, editor, seo]
  research_rounds: 3
  research_depth_target: 6
  think_research: false     # Accuracy over depth

# Explainer/Concept
explainer:
  steps: [writer, editor, seo]  # Skip research entirely
  research_rounds: 0
  context: optional

# News/Update (fast turnaround)
news:
  steps: [research, writer, seo]  # Skip synthesizer + editor
  research_rounds: 2
  research_depth_target: 4
  think_research: false
```

### Future: category auto-detection

The pipeline could infer category from the topic string:
- "How to..." / "Guide to..." → tutorial
- "Why..." / "What is..." → explainer
- Contains a date or "just released" → news
- Default → research-driven

Or just add `--category` to the CLI and let the user choose.

---

## Appendix B: Tonight's Debugging Timeline

The num_ctx discovery that inspired this plan:

```
17:55 — Pipeline run 7 starts, extractor at 27.2 tok/s (was 4.8)
18:12 — Pipeline complete: 643s (was 742s)
18:13 — MC starts ROCm article
18:30 — MC finishes: 565s, 47/50, surface-level
18:31 — CC starts same topic
19:00 — CC finishes: deep analysis with RDNA 4 Vulkan data
19:15 — Head-to-head comparison reveals the gap
19:30 — v0.3.5 plan drafted
```

The 100-second speedup from num_ctx was a performance win.
The quality gap revealed by the head-to-head is the strategic win.
v0.3.5 addresses the strategic gap.

---

## Appendix C: Model Selection Research (2026-03-04)

Late-night conversation identified critical issue: deepseek-r1:14b has **no native tool calling**. Building the agentic research loop on R1 would require fragile text-parsing hacks.

qwen3.5:9b (released 2026-03-02) was identified as the replacement:
- Native tool/function calling (RL-trained across million-agent environments)
- Toggleable thinking (`/think` / `/no_think`)
- 262k model context window
- 6.6GB model size → more VRAM headroom on 12GB card
- Already installed on rushuna
- Benchmarks: beats GPT-OSS-120B on GPQA Diamond (81.7 vs 71.5)

This discovery cut estimated implementation time from 8-12 hours to 4-6 hours and eliminated the biggest technical risk in the plan.
