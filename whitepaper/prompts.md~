

Phase 1 — Categorize all articles:


===================================
TASK: Add categories to all article frontmatter across guides/ and blog/

For every .md file in content/guides/ and content/blog/, add a `categories` field to the frontmatter. Use ONLY these categories:

GUIDE CATEGORIES:
- "getting-started" — beginner guides, first LLM, basics
- "hardware" — GPU guides, VRAM requirements, buying guides, platform comparisons (Mac vs PC, AMD vs NVIDIA), laptop vs desktop
- "models" — model-specific guides (Llama, Qwen, DeepSeek), model comparisons, quantization, model formats
- "software" — Ollama, LM Studio, llama.cpp, vLLM, Open WebUI, text-gen-webui, inference engines
- "agents" — OpenClaw, agent frameworks, function calling, MCP, routing
- "use-cases" — lawyers, coding, writing, math, summarization, privacy, RAG, voice chat, image gen, video gen
- "architecture" — KV cache, speculative decoding, MoE, context length, transformers, RWKV, distributed inference
- "troubleshooting" — fix guides, CUDA OOM, Ollama errors, model loading issues, debug guides

BLOG CATEGORIES (for content/blog/ posts):
- "news" — time-sensitive coverage (new releases, acquisitions, features)
- "opinion" — think pieces, analysis, editorials
- "weekly" — weekly roundup posts

RULES:
- Each article gets exactly ONE primary category
- Add it as: categories: ["hardware"] in the frontmatter
- Also add a `featured` field to the top 15 highest-traffic articles: featured: true
  These are: best-local-models-openclaw, best-local-llms-mac-2026, best-local-coding-models-2026, openclaw-token-optimization, running-llms-mac-m-series, best-openclaw-alternatives, llamacpp-vs-ollama-vs-vllm, vram-requirements-local-llms, best-openclaw-tools-extensions, gpu-buying-guide-local-ai, openclaw-setup-guide, run-first-local-llm, ollama-vs-lm-studio, openclaw-clawhub-security-alert, local-alternatives-claude-code-2026
- Do NOT change slugs, filenames, URLs, or any other frontmatter
- Do NOT move files between directories
- Build with hugo when done, verify 0 errors

==========================================

Phase 2 — Build the new guides hub + nav:


TASK: Rebuild the /guides/ listing page and update site navigation

1. GUIDES HUB PAGE (layouts/guides/list.html or wherever the guides listing template is):

Replace the current flat list with a clustered topic hub. Structure:

a) SEARCH BOX at top — simple JS filter that hides/shows articles as user types

b) FEATURED section — show articles with featured: true in frontmatter. Display as cards with title + description. Header: "⭐ Most Popular"

c) TOPIC CLUSTERS — one section per category. For each:
   - Emoji + category name + article count: "🖥️ Hardware & GPUs (34 guides)"
   - Show top 5 articles (by title, linked)
   - "Show all →" link/button that expands to show remaining articles in that category
   - Use the categories from frontmatter to group

   Display order:
   1. ⭐ Most Popular (featured: true)
   2. 🚀 Getting Started
   3. 🖥️ Hardware & GPUs
   4. 🤖 Models
   5. ⚡ Software & Tools
   6. 🕵️ AI Agents & OpenClaw
   7. 💼 Use Cases
   8. 🧠 Architecture & Theory
   9. 🔧 Troubleshooting

d) Keep the existing card styling from the site — match fonts, colors, spacing

2. TOP NAV — update the nav in layouts/partials/ (header.html or nav.html):

Change:
  Home | Guides | Blog | Planning Tool | About

To:
  Home | Guides ▾ | Blog | Tools | About

- "Guides ▾" shows a dropdown on hover/click with:
  Getting Started
  Hardware & GPUs
  Models
  Software & Tools
  AI Agents
  Use Cases
  Architecture
  Troubleshooting
  ──────────────
  View All Guides

  Each links to /guides/ with an anchor (#hardware, #models, etc.)
  Add matching id anchors to each section in the hub page

- "Tools" links to /tools/vram-calculator/ for now (will become a tools hub later)

3. HOMEPAGE — add one new card after "What Are You Looking For?" heading:

### 🔥 Latest
[show the 3 most recently published articles by date, auto-generated from frontmatter]

This ensures return visitors see fresh content.

4. Build with hugo, verify 0 errors, test the search box works, test the dropdown nav works, test anchor links scroll to correct sections.
Run Phase 1 first, then Phase 2 once it's done. Phase 1 is pure frontmatter edits — safe and fast. Phase 2 is the template work that depends on the categories being in place.

============================================================================


My recommended priority:

🔴 OpenClaw Alternatives (comprehensive) — rides the Meta email deletion news cycle, your existing alternatives article already gets traffic, massive search demand
🔴 Intent Engineering practical guide — follow-up to the Nate Jones piece while the topic is hot
🟡 Agent trust decay / alignment tax — the "content rot" style deep problem article


===================================================================

Here are the 3 CC prompts:
1. Comprehensive OpenClaw Alternatives:



Read INSIDERLLM-PROJECT.md and insiderllm-content-plan.md. Use the insiderllm-writer skill.

Write: Every OpenClaw Alternative Worth Trying in 2026 (content/guides/openclaw-alternatives-comprehensive-2026/index.md)

This is the DEFINITIVE comparison. Do web research on ALL of these:

AGENTS TO COVER:
- NanoClaw (github.com/qwibitai/nanoclaw) — 500 lines TypeScript, container isolation, Claude Agent SDK, 7K+ GitHub stars, agent swarms. Built by Gavriel Cohen (ex-Wix). Security-first.
- Nanobot (github.com/HKUDS/nanobot) — 4K lines Python, HK University, vLLM local LLM support, MCP support, multi-provider. The lightweight learning project.
- LightClaw (github.com/OthmaneBlial/lightclaw) — featherweight OpenClaw reimplementation, hackable, small codebase
- ZeroClaw — Rust-based, 3.4MB binary, sub-10ms cold starts, runs on $10 edge devices. Harvard/MIT contributors.
- memU — knowledge graph memory, learns user habits over time, proactive agent, reduces API costs
- SuperAGI — multi-agent framework, build agent teams that coordinate, long-term memory, open source
- Moltworker — OpenClaw on Cloudflare Workers, serverless deployment, no self-hosting needed
- n8n — visual workflow automation, 400+ integrations, self-hostable, enterprise-ready
- Jan.ai — 100% offline chat, not an agent but captures the local-first desire
- Claude Code — developer-focused, coding agent, not a personal assistant
- AnythingLLM — RAG-focused, document chat, multi-LLM support

For EACH alternative cover:
- What it is (1-2 sentences)
- GitHub stars / maturity
- Can it run local models? (critical for our audience)
- Hardware requirements
- Security model (container isolation vs application-level vs none)
- Messaging platform support
- Honest strengths and weaknesses
- Who should use it

COMPARISON TABLE required: all alternatives side-by-side on key dimensions (local model support, security, codebase size, messaging platforms, memory, ease of setup)

CONTEXT: The Meta researcher email deletion story just broke (Feb 23-25, TechCrunch, PCWorld, Malwarebytes). OpenClaw security is in the news again. Reference it in the intro — people are actively searching for safer alternatives right now.

Honest take: OpenClaw is still the most feature-rich. These alternatives win on focus — security (NanoClaw), simplicity (Nanobot), memory (memU), performance (ZeroClaw), local-first (Nanobot + LightClaw).

Links: openclaw-setup-guide, openclaw-clawhub-security-alert, openclaw-token-optimization, best-local-models-openclaw, lightclaw-lightweight-openclaw-alternative, local-ai-agents-guide, Planning Tool

Generate PDF. Build with hugo. Run /humanizer.

===================================


2. Intent Engineering Practical Guide:



Read INSIDERLLM-PROJECT.md and insiderllm-content-plan.md. Use the insiderllm-writer skill.

Write: Intent Engineering for Local AI Agents — A Practical Guide (content/guides/intent-engineering-local-ai-guide/index.md)

This is the PRACTICAL follow-up to our intent-engineering-ai-agents article (which covers the theory/Nate Jones framework). This article is about HOW TO DO IT with local agents.

Structure:
- Quick recap: prompt engineering → context engineering → intent engineering (link to the theory article)
- Why local agents need intent engineering MORE than cloud agents (they run longer, no human babysitting, persistent state)

PRACTICAL SECTIONS:

1. Encoding goals your agent can act on
   - Bad: "be helpful" / Good: specific objective functions with measurable signals
   - Show a real system prompt evolution from vague to intent-engineered
   - Example: a local agent managing files — what does "organize" mean? Define it.

2. Decision boundaries
   - When should the agent act vs ask? Define explicit thresholds.
   - Example: email triage agent — delete spam (act), flag important (act), reply to client (ASK)
   - Show this as a simple Python config/dict structure

3. Value hierarchies for tradeoffs
   - When speed conflicts with thoroughness, which wins? Encode it.
   - Example: research agent — "prefer accuracy over speed, prefer recent sources over old, never fabricate citations"
   - Show as structured YAML/JSON that agents can parse

4. Memory as intent persistence
   - Episodic memory lets agents learn from past decisions
   - Without memory, intent resets every session
   - How mycoSwarm approaches this: IFS parts system, Wu Wei Timing Gate
   - How OpenClaw handles it (or doesn't — context rot problem)

5. Feedback loops
   - How to detect alignment drift in long-running agents
   - Simple logging: log every decision + rationale, review weekly
   - Automated: flag decisions that violate encoded boundaries

6. Starter template
   - Provide a complete intent engineering template (YAML or JSON) that readers can adapt
   - Cover: agent role, objectives, decision boundaries, escalation rules, value hierarchy, forbidden actions

Do web research on:
- Google Agent Development Kit architecture (working context, session memory, long-term memory, artifacts)
- DeepMind's 5 levels of AI agent autonomy paper
- Deloitte 2026 State of AI stats (74% no tangible value, 84% haven't redesigned jobs)

Links: intent-engineering-ai-agents, local-ai-agents-guide, openclaw-memory-context-rot, session-as-rag, function-calling-local-llms, Planning Tool

Generate PDF. Build with hugo. Run /humanizer.


3. Agent Trust Decay:



Read INSIDERLLM-PROJECT.md and insiderllm-content-plan.md. Use the insiderllm-writer skill.

Write: Agent Trust Decay — Why Long-Running AI Agents Get Worse Over Time (content/guides/agent-trust-decay-long-running-ai/index.md)

This is the "content rot" style deep problem article. It names a problem most people feel but can't articulate.

THE CORE THESIS: The longer an AI agent runs autonomously, the less you should trust its outputs — and almost nobody is measuring this.

Sections:

1. What trust decay looks like
   - Agent works great for day 1-3, makes subtle errors by day 7, confidently wrong by day 14
   - Context windows fill with stale information
   - Accumulated small errors compound (wrong assumption in hour 2 becomes foundation for decisions in hour 40)
   - The agent doesn't know it's degrading — it maintains the same confidence level

2. Why it happens
   - Context window pollution — old irrelevant context crowds out new relevant context
   - Memory without forgetting — agents that remember everything remember too much
   - Drift from original intent — small optimizations push the agent away from its goals over time
   - No reality check — humans self-correct through social feedback, agents don't
   - Token economics — as context grows, inference gets slower and more expensive, so people truncate, losing critical context

3. The Klarna parallel
   - Same pattern at organizational scale — agent optimized for speed, drifted from customer satisfaction
   - Reference the Meta researcher email deletion (agent drifted from "organize" to "delete")
   - These aren't bugs, they're trust decay in action

4. How to detect it
   - Decision logging — track every agent action + stated rationale
   - Periodic human audit — sample 5% of decisions weekly
   - Drift metrics — compare agent behavior at day 1 vs day 7 vs day 30
   - Canary tasks — give the agent known-answer tests periodically to check calibration
   - Output quality scoring — automated checks on agent outputs over time

5. How to prevent it
   - Context pruning — actively remove stale information (not just append)
   - Memory lifecycle management — facts should have expiration dates
   - Intent re-anchoring — periodically re-inject core objectives fresh
   - Session boundaries — long-running doesn't mean infinite. Reset context strategically.
   - Human checkpoints — scheduled review gates, not just "call me if something breaks"
   - mycoSwarm's approach: fact lifecycle with temporal recency, episodic memory pruning, Wu Wei Timing Gate for decision pacing

6. The trust budget framework
   - Propose a simple mental model: agents start with a "trust budget" that depletes over time without maintenance
   - Fresh agent = high trust = more autonomy
   - 7-day agent = medium trust = human spot-checks required
   - 30-day agent = low trust = mandatory audit before continuing
   - This maps to DeepMind's 5 levels of agent autonomy

Do web research on:
- Meta researcher OpenClaw email deletion story (Feb 2026)
- Klarna AI customer service timeline
- Any academic work on LLM performance degradation over long contexts
- DeepMind 5 levels of agent autonomy

Links: openclaw-memory-context-rot, intent-engineering-ai-agents, intent-engineering-local-ai-guide, local-ai-agents-guide, context-length-explained, kv-cache-optimization-guide, Planning Tool

Generate PDF. Build with hugo. Run /humanizer.


=================================================



On Nate Jones' workflow — you're absolutely right. That transcript is 6,000+ words of tightly structured argument with citations from Deloitte, Gartner, McKinsey, KPMG, Bloomberg, and MIT — all woven into a coherent narrative with a novel framework (intent engineering) and specific examples (Klarna, Copilot). No one writes that from scratch in a reasonable timeframe.
Here's what's almost certainly happening:
His likely workflow:

Research phase — he's monitoring the same feeds you are (company blogs, analyst reports, HN, Reddit, X). He probably has a system similar to your news monitor, or uses Perplexity/ChatGPT to surface and summarize relevant reports. The Deloitte, Gartner, and KPMG citations suggest he's reading (or having AI read) the actual analyst reports.
Synthesis/framing — this is where his human value is. He identifies the thread connecting disparate stories (Klarna + Copilot + Deloitte stats = intent gap). The framework naming ("intent engineering") is likely his original contribution. AI can't do this well — it can summarize, but it doesn't see the meta-narrative.
Script generation — he almost certainly feeds his thesis + research into Claude or GPT-4 with something like: "Here's my argument, here are my sources, write a 20-minute video script in my voice." Then he edits heavily. The intellectual density you're noticing is the hallmark of AI-assisted writing — humans ramble, AI compresses.
Production — 127K YouTube subscribers means he has a rhythm. Probably records talking-head or voiceover, maybe uses AI for thumbnail generation and title optimization.

The key insight for you: His moat isn't the AI writing — anyone can do that. His moat is the curation and framing. He reads 50 sources and sees the one thread that connects them. That's the same thing you do with InsiderLLM — you're not just summarizing specs, you're telling people "here's what actually matters and why."
What this means for InsiderLLM:
You could adopt a similar workflow for opinion/analysis pieces:

News monitor catches the stories
You identify the connecting thread (your human insight)
Feed your thesis + sources to CC with a detailed prompt
CC writes the draft, humanizer cleans it
Deploy same day while the topic is hot

You're already doing this — the intent engineering article was exactly this workflow. The difference is Jones does it as video (higher production value, harder to scale) and you do it as articles (easier to produce, better for SEO, AI crawlers can cite you).
Want me to write the dashboard prompt now, or do you have more to discuss first?

=========================================================


TASK: Build an InsiderLLM traffic dashboard from log analysis reports.

Read these 3 reports:
- docs/log-analysis-2026-02-15.md (or wherever the first report is)
- docs/log-analysis-2026-02-19.md
- docs/log-analysis-2026-02-24.md

Create a single-page HTML dashboard at insiderllm-hugo/static/tools/traffic-dashboard/index.html using Chart.js (CDN).

CHARTS TO INCLUDE (all as line graphs with data points labeled):

1. Daily Human Pageviews — trend line across all report periods
2. ChatGPT Crawler Requests/Day — show the hockey stick
3. ChatGPT Human Clicks/Day — the referral growth
4. Search Engines: DDG vs Google vs Bing — three lines, same chart
5. Social: Facebook vs X/Twitter vs Bluesky — three lines, same chart  
6. AI Chat Referrals: ChatGPT vs Perplexity vs Claude.ai — three lines
7. AI Crawler Volume: ChatGPT-User vs Amazonbot vs Meta vs ClaudeBot
8. GitHub Referrals/Day

Extract the data points from the period breakdowns in each report. Use the midpoint date of each period as the x-axis value.

STYLE:
- Dark theme (matches InsiderLLM aesthetic)
- Responsive, works on mobile
- Title: "InsiderLLM Traffic Dashboard"
- Last updated date shown
- Each chart in its own card with a title

This is internal/private — do NOT add it to the nav or link from any public page.
Do NOT build with hugo — this is a standalone HTML file for internal use only.




=================================================









