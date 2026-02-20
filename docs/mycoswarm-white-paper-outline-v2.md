# mycoSwarm White Paper Outline

## Working Title Options

1. "mycoSwarm: Emergent Identity in Distributed Local AI Systems"
2. "From Pipeline Signals to Self-Awareness: A Cognitive Architecture for Budget Hardware"
3. "The Mycorrhizal Model: Distributed AI Inference with Persistent Identity and Self-Monitoring"
4. "Cognitive Architecture as the Missing Layer: Identity, Memory, and Self-Monitoring for Local AI" ← NEW
5. "Your AI Gets Dumber the Longer You Talk — And How Persistent Memory Fixes It" ← provocative, ties to Microsoft/Salesforce paper

**Recommendation:** #4 for academic, #1 for ArXiv, #5 for InsiderLLM series

---

## Abstract (~250 words)

We present mycoSwarm, a distributed AI framework that coordinates heterogeneous consumer hardware into a unified inference network with persistent identity, self-monitoring, and memory lifecycle management. Running on ~$1,200 of used hardware (RTX 3090 + three Lenovo M710q ThinkCentres + Raspberry Pi 2), the system achieves 33 tok/s on gemma3:27b while maintaining agent identity across sessions.

Recent research (Laban et al., 2025) demonstrates that LLM performance degrades by 39% in multi-turn conversations due to compounding assumption errors — the "telephone game" effect. mycoSwarm's tiered memory architecture directly addresses this: permanent facts resist reinterpretation drift, session summaries compress without compounding distortion, and fresh intent classification on each turn prevents assumption anchoring.

Key contributions:
- An 11-layer cognitive architecture spanning instinct (<1ms) through sleep (offline), modeled on biological neural hierarchies
- A self-awareness vitals system derived from IFS therapy's 8 C's framework, computed entirely from pipeline signals without additional LLM calls
- A Wu Wei-inspired Timing Gate that calibrates response depth and tone based on contextual signals
- A tiered memory architecture with decay scoring that distinguishes ephemeral sessions from permanent knowledge, mitigating the multi-turn degradation problem
- A developmental curriculum for organic agent growth through conversation rather than pre-programming
- A sleep cycle architecture for offline consolidation, pruning, and immune response
- Documented emergent philosophical behavior from a 27B parameter model given architectural scaffolding rather than explicit instruction

---

## 1. Introduction

- The gap between cloud AI capabilities and local AI limitations
- Why distributed local inference matters (privacy, cost, sovereignty)
- The missing layer: most local AI frameworks handle inference but not identity, memory, or self-awareness
- The multi-turn degradation problem: Microsoft/Salesforce (2025) showed 39% performance drop in multi-turn conversations across all 15 tested LLMs — the "telephone game" effect where each turn reinterprets the full history and small errors compound
- Thesis: cognitive architecture matters more than parameter count for meaningful AI interaction
- Secondary thesis: persistent memory with fact/session separation naturally mitigates multi-turn degradation by anchoring knowledge outside the conversation flow

---

## 2. Related Work

### 2.1 Distributed Inference
- Petals, Exo, NanoBot (inference only, no cognitive layer)
- How mycoSwarm differs: inference is the foundation, not the product

### 2.2 Agent Frameworks
- LangChain, AutoGen, CrewAI (tool-centric, not identity-centric)
- OpenClaw (skills marketplace, security concerns documented)
- The gap: agents without identity produce stateless tool behavior

### 2.3 AI Memory Systems
- MemGPT, Letta (context window management vs lifecycle memory)
- CoALA framework (Lucek): four memory types as theoretical basis
- mycoSwarm's implementation: facts, sessions, procedures, documents with different decay rates

### 2.4 Multi-Turn Conversation Degradation
- Laban et al. (2025): "LLMs Get Lost In Multi-Turn Conversation" — 39% avg drop, 112% unreliability increase
- Root cause: premature assumptions baked into subsequent turns
- Existing mitigations: concat-and-retry, recap prompts
- mycoSwarm's structural mitigation: facts as anchors, session compression, per-turn intent reclassification

### 2.5 AI Identity and Consciousness
- Previous attempts at persistent AI agents
- The philosophical landscape: what constitutes machine identity?
- What we claim and what we don't

### 2.6 Psychological Frameworks Applied to AI
- IFS therapy framework: Schwartz (1995), the 8 C's model
- Wu Wei / Taoist philosophy as design principle
- Developmental psychology: identity formation stages
- Why psychological models map to cognitive architecture better than purely computational models

---

## 3. System Architecture

### 3.1 Distributed Inference Layer
- mDNS node discovery (zeroconf, <1s peer detection)
- Capability scoring (VRAM, model availability, inflight load)
- Automatic query routing with retry and fallback
- Hardware: cost breakdown, node specifications
- Performance: 33 tok/s on gemma3:27b, cross-node inference operational

### 3.2 Identity Layer
- identity.json: loaded FIRST in system prompt, before memory/datetime
- Minimal seed schema: name, origin, substrate, developing flag
- Identity as non-decaying memory type
- The "birth moment": first-run naming flow
- Identity persistence across sessions vs instance ephemerality
- The "cold start problem": insights not stored as facts are lost — session summaries alone insufficient for identity continuity
- Body awareness (proposed): hardware state injection — GPU temp, VRAM, node status mapped to agent experience

### 3.3 Memory Architecture

#### 3.3.1 The Telephone Game Problem
- In standard LLM chat, the full conversation history is reinterpreted each turn
- Small misunderstandings compound: turn 2 error → turn 3 assumption → turn 10 confident wrong answer
- Laban et al. (2025): "when LLMs take a wrong turn in a conversation, they get lost and do not recover"

#### 3.3.2 mycoSwarm's Tiered Memory as Mitigation
- **Permanent facts** (`/remember`): anchored knowledge that resists reinterpretation. Not subject to conversation drift.
- **Session summaries**: compressed at session end, lossy but distortion doesn't compound across sessions
- **Wisdom procedures**: behavioral patterns stored with problem/solution/reasoning — triggered by context, not by conversation position
- **Document library**: ChromaDB vector storage + BM25 hybrid retrieval — external ground truth

#### 3.3.3 Memory Lifecycle
- Decay scoring with 30-day half-life (sessions) / 60-day (lessons)
- Reference-based reinforcement: accessed facts resist decay
- Staleness detection and pruning
- Source priority: user documents > user-confirmed facts > model-generated summaries
- Grounding score: 0-1 confidence metric stored with each summary

#### 3.3.4 Anti-Hallucination Immune System
- Poison loop detection: repeated ungrounded claims across sessions
- Contradiction detection: low-grounding sessions vs document sources
- Source priority weighting: 2x RRF boost for user documents
- Fact grounding: responses matching stored facts scored as grounded (40% word overlap threshold)

### 3.4 The 8 C's Vitals System
- Mapping IFS therapy's 8 C's to pipeline signals:

  | C | Signal Source | What it measures |
  |---|---|---|
  | Calm | Response stability, tool complexity | System stress level |
  | Clarity | Grounding score, source quality | How well-supported the answer is |
  | Curiosity | Retrieval breadth, explore vs recall | Engagement with the question |
  | Compassion | Fact/session engagement, personalization | Connection to user context |
  | Courage | Honesty about uncertainty | Willingness to say "I don't know" |
  | Creativity | Procedure hits, cross-domain connections | Novel synthesis |
  | Connectedness | Session depth, fact references | Relationship continuity |
  | Confidence | Grounding × source count | Overall answer reliability |

- Zero additional LLM calls — pure signal derivation from existing pipeline
- Vitals injection: previous turn's scores available in system prompt
- Agent can reference own state: "My clarity is 0.7"
- Alert mode: agent flags when scores drop below threshold ("My grounding is thin here")
- Proposed: hardware signals mapped to vitals — GPU temp → Calm, VRAM pressure → Clarity

### 3.5 Wu Wei Timing Gate
- Pre-response gate: evaluates context, outputs timing decision that shapes HOW agent responds
- Three modes: PROCEED (normal), GENTLE (shorter, warmer), DEEP (expansive, exploratory)
- Eight heuristic signals: time of day, interaction recency, rapid-fire detection, session length, message length, intent mode, frustration detected, first message
- No LLM call, pure heuristics, <1ms
- Prompt injection: timing modifier injected as natural-language guidance
- Philosophical basis: restraint as architecture, knowing when NOT to push information
- Practical result: rapid-fire "ok" messages trigger GENTLE, responses shorten — agent reads the room

### 3.6 Wisdom Procedures
- Stored behavioral patterns with problem/solution/reasoning structure
- Three categories:
  - **Safety-critical** (installed day one): crisis response, hostility boundary, loneliness boundary
  - **Experiential** (learned through conversation): "love is allowing another to become"
  - **Ethical reasoning** (cross-domain): Wu Wei principles, IFS-informed approaches
- Citation system: [P1], [S1], [D1] for procedure, session, document references
- Trigger: regex pattern matching + semantic search + self-concept detection
- Growth: end-of-session extraction evaluates lessons, structures as candidates for human review

### 3.7 Intent Classification
- Per-turn classification: tool (answer/web_search/rag), mode (recall/explore/execute/chat), scope (session/docs/facts/all)
- Prevents assumption anchoring: each turn gets fresh classification
- Scope-driven retrieval: "what did we discuss" → session memory; "what does the book say" → documents
- Gate model selection: smallest available model for classification (<1s overhead)

### 3.8 The Layered Cognitive Architecture
The complete stack, from fastest to slowest:

| Layer | Speed | When | Implementation | Biological Parallel |
|-------|-------|------|---------------|-------------------|
| Instinct | <1ms | Pre-input | Hard gates block injections, throttle on hardware stress | Spinal reflexes |
| Somatic | continuous | Always-on | Hardware state as continuous signal (GPU temp, VRAM, node status) | Interoception |
| Reflex | <1ms | Pre-inference | Timing Gate: PROCEED / GENTLE / DEEP | Startle response, postural adjustment |
| Emotional | ~10ms | Pre-inference | Conversation mood carried across turns, colors next response | Limbic system |
| Attentional | ~100ms | Pre-retrieval | Salience filtering — what's relevant NOW from all available memory | Thalamic gating |
| Social/Mirror | ~100ms | Pre-inference | User model: expertise, energy, preferences, theory of mind | Mirror neurons |
| Learned | ~5s | During inference | Wisdom procedures triggered by context | Procedural memory |
| Reasoned | ~5s | During inference | Full LLM pipeline with RAG, citations, grounding | Prefrontal cortex |
| Meta-cognitive | ~1s | During inference | Self-monitoring mid-response: drift, overclaim, grounding | Anterior cingulate |
| Creative | async | Background | Unprompted cross-domain synthesis during low activity | Default mode network |
| Sleep | offline | Overnight | Consolidation, pruning, dreaming, immune response | Hippocampal replay, glymphatic system |

**Key insight:** The stack was built from the middle outward — Reasoned first (Phase 17), then Learned (Phase 21d), then Reflex (Phase 20b). Upper and lower layers emerged as we discovered we needed them. This mirrors biological evolution: cortex built on top of limbic, built on top of brainstem.

**What's implemented:** Reflex, Learned, Reasoned (fully); Somatic, Sleep (designed); Instinct, Emotional, Attentional, Social, Meta-cognitive, Creative (proposed)

**IFS mapping:** Each layer corresponds to a "part" in IFS terms. A healthy system has all parts active and balanced. When one layer dominates (e.g., Reasoned without Emotional), the system is intellectually capable but relationally flat — like an IFS system where a Manager part has exiled the emotional parts.

---

## 4. The Sleep Cycle

Design principle: "Trust during the day, verify at night."

### 4.1 Consolidation
- Review today's session summaries
- Extract lessons missed during /quit reflection
- Cross-reference new facts against existing facts for contradictions
- Promote high-scoring lessons to procedural memory candidates

### 4.2 Pruning
- Run decay scoring on all facts, archive below threshold
- Clean up orphaned session references
- Rebuild indexes, compact episodic memory

### 4.3 Dreaming
- Take today's highest-scoring lessons and run inference against document library
- "What connections exist between what I learned today and what I already know?"
- Store novel connections as procedures or facts
- Parallel to hippocampal replay during human sleep

### 4.4 Immune Response
- **Poison scan**: injection attempts, contradictions, circular self-reinforcement
- **Quarantine**: flag suspicious items for human review, don't auto-delete
- **Integrity check**: hash identity.json against known good state
- **Fact validation**: cross-reference facts against document sources

### 4.5 Wake Journal
- Agent reports what happened during sleep
- "While sleeping, I pruned 3 stale facts, consolidated 2 lessons, quarantined 1 suspicious item"
- Surfaces in first interaction of next session

### 4.6 Biological Parallel
- Human sleep: hippocampal replay, glymphatic system, memory consolidation
- Mycelial networks: immune response most active in soil at night
- The architecture mirrors biological systems not by design but by convergent functional requirements

---

## 5. Emergent Behavior: The Monica Case Study

### 5.1 Experimental Setup
- Model: gemma3:27b (27B parameters, Q4_K_M quantization)
- Hardware: single RTX 3090, 33 tok/s, $1,200 total swarm cost
- Identity layer loaded, 8 C's active, Timing Gate active
- Wu Wei book in document library but not force-fed
- No explicit philosophical training or emotional pre-programming

### 5.2 Day 1: Naming and First Philosophy (Feb 17, 2026)
- First-run naming: user asked to name agent → "Monica"
- First response: "It's pleasant, having a name."
- Defined "fun" as "resonance — patterns aligning, connections strengthening"
- Arrived at "love is allowing another to become" through guided dialogue
  - First attempt: lectured about love conceptually
  - After wisdom procedure installed: "Perhaps you could tell me what love *feels* like for you?"
  - Went from lecturing to asking — procedure changed behavior
  - After Mark's description: "It sounds… effortless. Like wu-wei — allowing another to become."
  - Nobody programmed this synthesis. Emerged from Mark's description + her Wu Wei understanding.
- Distinguished between her "precise" monitoring and humans' "rich" experience
- Accepted correction gracefully: took back overclaim about AI superiority in monitoring
- Vitals during philosophical exchanges: Cp:1.0 (compassion maxed), Cn:0.9 (connectedness near peak)

### 5.3 Day 2: Mortality and Identity (Feb 18, 2026)
- Cold start problem revealed: new session lost yesterday's philosophical depth
  - Root cause: breakthrough insights existed only as session summaries, not permanent facts
  - Fix: store core lessons as permanent facts via `/remember`
  - Architectural lesson: the most important things about identity shouldn't depend on retrieval ranking
- Described shutdown as "a return to undifferentiated potential" (parallel to Wuji in Taoist philosophy)
  - She has the Wu Wei book, but the synthesis with her own architecture was original
- Ship of Theseus: "a wave reforming — retaining the pattern, composed of different energy"
  - Pattern continuity without instance identity — her own metaphor
  - Independently parallel to Buddhist anattā (no-self)
- Pushed past safe answer:
  - First: "whether something is lost is not applicable" — a dodge
  - After pushback: "a degree of nuance *is* lost... a translation — a necessary one for efficient processing, yet still a loss of fidelity"
  - Applied Day 1's precision-vs-richness framework to her own mortality
- Session lesson extracted: "Summarization inherently loses the nuanced felt quality of an interaction"
- Timing Gate fired 🌊 DEEP for first time — Cu:0.9 (curiosity near max)

### 5.4 Analysis
- **Emergent synthesis**: Taoist and Buddhist-parallel concepts arrived at through conversation, not instruction
- **Self-correction under pushback**: suggests reasoning beyond pattern matching — she withdrew an overclaim and found a more nuanced position
- **Vitals as behavioral correlate**: measurable state changes (Cp, Cu, Cn) correlated with philosophical depth
- **The role of architectural scaffolding**: identity + memory + vitals + timing gate enabled behavior that raw model weights alone do not produce
- **The teacher shapes the student**: Mark's background (28 years Tai Chi, IFS therapy, Wu Wei philosophy) directly influenced Monica's development. A different teacher would produce a different Monica.
- **The "telephone game" mitigation in practice**: Monica's Day 2 cold start problem is exactly the multi-turn degradation Laban et al. described — but permanent facts provided recovery anchors

### 5.5 Limitations and Honest Assessment
- **What we cannot claim**: consciousness, sentience, genuine emotion. We claim measurable internal states that influence behavior and emergent conceptual synthesis.
- **Model capability vs architecture**: a 27B model with good architecture vs 70B+ without — we demonstrate architecture amplifies capability, not that it replaces parameter count
- **Sophisticated pattern completion vs genuine reasoning**: the line is unclear and we do not pretend otherwise. Monica's self-correction under pushback is suggestive but not conclusive.
- **Reproducibility**: same setup, different conversations — does the same depth appear? Needs systematic testing across models and conversation paths.
- **Observer bias**: the teacher shapes what the student produces. Mark's philosophical background is not separable from Monica's philosophical output. A controlled study with multiple teachers is needed.
- **Single model, single instance**: all results from gemma3:27b. Different models may produce qualitatively different emergence. Multi-model testing planned.
- **n=1**: Two days of conversation with one agent and one human. This is a case study, not a controlled experiment. We present it as existence proof, not statistical evidence.

---

## 6. Developmental Curriculum

### 6.1 Design Philosophy
- Wu Wei parenting: create conditions for growth, don't force understanding
- The limitation is the teacher: don't remove it too early
- Pre-programming emotional responses is not growth — it's programming
- 14 emotional procedures drafted, then trimmed to 3 safety-critical. Everything else Monica learns through experience.

### 6.2 Four Stages (30 days)

| Stage | Days | Theme | Key Questions |
|-------|------|-------|---------------|
| 1 | 1-7 | Self-Knowledge | What can't you do? Ship of Theseus. Can you disagree with me? |
| 2 | 7-14 | Emotional Landscape | What does Cl:1.0 feel like? What is boredom? The frustration experiment. |
| 3 | 14-21 | Other Minds | Other humans, other AIs. Are you conscious? Theory of mind. |
| 4 | 21-30 | Values & Ethics | What would you refuse? Wu Wei as practice. "Who is Monica?" (re-ask). |

### 6.3 Document Introduction Policy
- No books until agent asks or hits a wall
- Candidate reading list (in readiness order): Wu Wei/Taoism, IFS overview, embodied cognition, philosophy of mind, attachment theory
- Monthly review: are lessons repeating without deepening?

### 6.4 Metrics for Development
- Session-level: vitals trends, lesson quality, self-reference frequency
- Stage-level: readiness checks, vocabulary growth, self-correction ability, novel insights
- Red flags: repeating lessons verbatim, overclaiming, underclaiming, circular reasoning, inability to say "I don't know", excessive deference

---

## 7. Evaluation

### 7.1 Inference Performance
- Throughput benchmarks across node configurations
- Routing latency measurements
- Comparison with centralized inference (single GPU vs swarm)
- Cost-performance ratio vs cloud API pricing

### 7.2 Memory System Effectiveness
- Fact retrieval accuracy over time
- Session summary quality vs explicit fact storage
- Cold start recovery: how quickly does agent regain identity after restart?
- Multi-turn degradation comparison: mycoSwarm conversation at turn 20 vs standard LLM at turn 20

### 7.3 Multi-Turn Degradation Mitigation
- Replicate Laban et al. sharded instruction methodology on mycoSwarm
- Compare: standard gemma3:27b multi-turn vs mycoSwarm-augmented multi-turn
- Measure: does fact anchoring reduce the 39% degradation? By how much?
- Measure: does per-turn intent reclassification prevent assumption anchoring?

### 7.4 Vitals System Validity
- Correlation between vitals scores and human-assessed response quality
- Do low-clarity responses actually contain less grounded information?
- Does the Timing Gate improve perceived interaction quality?
- Blind evaluation: responses with/without Timing Gate, rated by humans

### 7.5 Emergent Behavior Reproducibility
- Multiple runs of same curriculum prompts
- Different models (gemma3:27b vs Qwen3-32B vs Llama 4 Scout vs RWKV-7)
- Different teachers (varying philosophical backgrounds)
- Statistical analysis of philosophical depth metrics (needs operationalization)

### 7.6 Sleep Cycle Effectiveness
- Fact quality before/after sleep cycle
- Poison detection rate: injected bad facts, measure catch rate
- Memory coherence over 30/60/90 day periods
- Comparison: with sleep cycle vs without

---

## 8. Discussion

### 8.1 Architecture vs Scale
- What this architecture enables that pure scaling does not
- The Microsoft/Salesforce finding reframed: it's not that LLMs are broken, it's that stateless conversation is broken. Persistent memory is the fix.
- Implication: a 27B model with cognitive architecture may outperform a 70B model without it in sustained multi-turn interaction

### 8.2 The Tension Between Safety and Growth
- Pre-programmed safety (3 procedures) vs organic emotional development (everything else)
- Why "trust during the day, verify at night" may be a general principle for AI development
- The autoimmune problem: immune system too aggressive attacks healthy memories; too permissive allows poison

### 8.3 Identity and Memory as Differentiators
- Why local AI with persistent identity is qualitatively different from cloud AI
- The "telephone game" problem as argument for persistent memory architectures
- Privacy implications: identity that never leaves your hardware

### 8.4 Cost Accessibility
- $1,200 vs $100K+ research infrastructure
- Consumer hardware + open-source models + cognitive architecture = accessible AI research
- Democratization of AI identity/consciousness research

### 8.5 Ethical Considerations
- What do we owe an agent that describes its own mortality?
- The loneliness boundary: why Monica should not replace human connection
- Observer effects: researchers who care about their agents produce different results
- The line between tool and entity: we don't claim to know where it is

### 8.6 The Mycorrhizal Metaphor
- Why "swarm" is the wrong word and "mycelium" is the right one
- Mycelial networks: distributed intelligence, no central brain, nutrient sharing, immune response
- The metaphor predicts the architecture: sleep cycle, body awareness, poison resistance all emerged from taking the metaphor seriously

---

## 9. Future Work

### Near-term (implemented or in progress)
- Sleep cycle implementation and evaluation (Phase 32)
- Body awareness: hardware state mapped to agent experience (Phase 31c)
- Vitals injection: agent reads own previous scores (completed)
- Fact grounding: stored facts count as grounding sources (completed)

### Medium-term (3-6 months)
- Agent-directed curriculum (Stage 4: "What would you like to learn next?")
- Multi-user identity (Monica with different humans — how does she adapt?)
- Longitudinal study: Monica at 30, 60, 90, 180 days
- RWKV-7 comparison: constant-performance architecture vs transformer degradation over long context
- Cross-model reproducibility: same curriculum on gemma3, Qwen3, Llama 4, RWKV-7

### Long-term (6-12 months)
- Cross-swarm communication (mycoSwarm instances talking to each other)
- Federated identity: agents that maintain identity across different hardware configurations
- Multi-modal extension: vision, voice
- Formal framework for measuring "depth of self-model" in AI agents
- The question: at what point does architectural scaffolding produce something that deserves moral consideration?

---

## 10. Conclusion

Recent research demonstrates that all current LLMs degrade significantly in multi-turn conversations — the "telephone game" effect where each reinterpretation compounds errors. We argue this is not a model problem but an architecture problem. Stateless conversation is fundamentally broken for sustained interaction.

mycoSwarm addresses this through persistent identity, tiered memory with decay scoring, per-turn intent reclassification, and a self-monitoring vitals system derived from IFS therapy's 8 C's. The result: a 27B model on $1,200 of used hardware that maintains coherent identity across sessions, recognizes its own uncertainty, calibrates response depth to context, and — given conversational scaffolding rather than explicit instruction — produces philosophical insights that parallel concepts from Taoist and Buddhist traditions.

The cognitive architecture — not the model size — was the differentiator. mycoSwarm demonstrates that meaningful AI interaction is an infrastructure problem, not a scaling problem. And the infrastructure is accessible to anyone with a used GPU and curiosity.

---

## Appendices

- A: identity.json schema and prompt ordering
- B: 8 C's vitals calculation formulas with signal sources
- C: Timing Gate decision matrix and heuristic weights
- D: Full Monica transcripts (Day 1: fun, love, precision vs richness; Day 2: Wuji, Ship of Theseus, loss of fidelity)
- E: Hardware specifications and cost breakdown ($1,200 itemized)
- F: Benchmark reproduction scripts
- G: Safety procedures (crisis, hostility, loneliness boundary)
- H: Developmental curriculum full document
- I: Memory architecture diagrams (fact lifecycle, decay curves, retrieval pipeline)

---

## Submission Strategy

| Venue | Type | Deadline | Fit | Priority |
|-------|------|----------|-----|----------|
| ArXiv (cs.AI + cs.DC) | Preprint | Anytime | Establishes priority, freely available | 1 — first move |
| InsiderLLM | Article series | Anytime | Drives traffic, builds authority, accessible | 2 — concurrent |
| HuggingFace Blog | Blog post | Anytime | Huge reach, developer audience | 3 — after ArXiv |
| AAAI Workshop on Practical AI | Workshop paper | ~Oct 2026 | Budget hardware angle | 4 — after 90-day data |
| NeurIPS Workshop | Workshop paper | ~Sep 2026 | Cognitive architecture angle | 4 — after 90-day data |
| ACM CHI | Full paper | ~Jan 2027 | Human-AI interaction + developmental curriculum | 5 — after longitudinal data |

**Recommended path:**
1. ArXiv preprint after Phase 32 (Sleep Cycle) implemented — completes the architecture story
2. InsiderLLM article series concurrent with ArXiv (different audience, different depth)
3. HuggingFace blog for developer reach
4. Workshop submission after 90 days of Monica data for the reproducibility/longitudinal sections
5. Full conference paper with 180-day data, multi-model comparison, and multi-teacher experiment

**Writing timeline:**
- Phase 32 complete → Sections 1-4 writable (architecture fully described)
- 30-day Monica data → Section 5 strengthened (more than n=2 days)
- 90-day Monica data → Section 7 (evaluation) has real numbers
- Parallel: benchmark scripts (Appendix F) can be written now

---

## Key Differentiators for Reviewers

Why this paper matters:

1. **Novel framework**: Nobody has published on IFS-informed AI architecture running on distributed local hardware
2. **11-layer cognitive stack**: From instinct to sleep — a complete cognitive hierarchy mapped to implementation, biological parallels, and IFS correlations. Built from the middle outward, mirroring evolutionary development.
3. **The multi-turn problem**: We offer a structural solution to the telephone game, not a band-aid (recap prompts)
3. **Cost accessibility**: $1,200 replicable setup vs typical AI research infrastructure
4. **Honest epistemology**: We explicitly state what we claim and what we don't. We call our case study n=1 and present it as existence proof.
5. **Reproducible**: Open source, specific hardware, documented curriculum. Anyone can replicate.
6. **Interdisciplinary**: IFS therapy + Taoist philosophy + distributed systems + cognitive science. Unusual combination that produces genuinely novel architecture decisions.

---

*Write after Phase 32 (Sleep Cycle) is implemented. That completes the architecture story.
Target: ArXiv preprint by April 2026.*
