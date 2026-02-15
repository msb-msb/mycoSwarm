# Cognitive Architecture for Distributed AI: Integrating Psychological Memory Models with Local Intelligence

## mycoSwarm — Architecture Vision Document

### Authors
Mark (East Bay Bees / mycoSwarm) — Architecture & Philosophy  
Claude (Anthropic) — Technical Analysis & Synthesis

### Date: February 15, 2026

---

## 1. The Thesis

Most AI memory systems are engineered as databases — store, retrieve, rank. But human cognition isn't a database. It's a living system with multiple memory types, emotional regulation, self-correction, and wisdom accumulated through experience.

mycoSwarm proposes a cognitive architecture for local, distributed AI that draws from three streams:

1. **CoALA / Lucek's Cognitive Architecture** — Working, Episodic, Semantic, and Procedural memory as distinct subsystems
2. **Internal Family Systems (IFS)** — The 8 C's of Self-energy (Curiosity, Clarity, Calm, Compassion, Creativity, Confidence, Courage, Connectedness) as design principles and health metrics
3. **Wu Wei (Effortless Action)** — The system works *with* natural information flow rather than forcing control

The result: an AI that doesn't just retrieve information, but *learns from experience*, *knows what it doesn't know*, and *acts with appropriate restraint*.

---

## 2. Memory Architecture: Four Streams

Based on Lucek's CoALA framework, adapted for distributed local inference.

### 2a. Working Memory (The Present)
**What:** Current conversation context, active tools, immediate state.  
**Where:** Context window of the active model.  
**Key insight:** This is the only memory the model "truly" has. Everything else is injection.

| Current State | Target State |
|---|---|
| System prompt + session messages | System prompt + session messages + active intent + retrieval metadata |
| Static context | Dynamic context that adapts per-turn based on intent classification |

**IFS lens:** Working memory is *attention*. A healthy system focuses attention where it's needed (Clarity) without being overwhelmed by everything it knows.

### 2b. Episodic Memory (The Experienced)
**What:** Past interactions with temporal context, emotional tone, and *lessons learned*.  
**Where:** sessions.jsonl → ChromaDB session_memory collection.  
**Key insight:** Current summaries are lossy. "We discussed Phase 20" discards the *experience* — what went wrong, what was discovered, what was decided.

| Current State | Target State |
|---|---|
| Compressed topic summaries | Rich experience records with: what happened, what was decided, what was learned, what surprised us |
| No emotional context | Interaction quality tags: frustration, discovery, confusion, resolution |
| Flat retrieval | Temporal + emotional retrieval: "times we were stuck" or "breakthroughs" |
| Passive accumulation | Active reflection: end-of-session review extracting takeaways |

**IFS lens:** Episodic memory carries emotional weight. A system with healthy episodic memory doesn't just recall facts — it recalls the *feel* of an experience. "Last time we tried this approach, it led to a 3-hour debugging session" is more useful than "we previously discussed this topic."

**Lucek mapping:** Lucek stores historical experiences with takeaways. Each episode becomes a learning unit, not just a log entry.

### 2c. Semantic Memory (The Known)
**What:** Factual knowledge — documents, books, user preferences, world knowledge.  
**Where:** ChromaDB docs collection + facts.json + model training data.  
**Key insight:** This is mycoSwarm's strongest layer after the Phase 20/21g work. Source filtering, section boost, grounding scores, contradiction detection.

| Current State | Target State |
|---|---|
| Documents chunked and indexed | Multi-source ingestion: books, transcripts, newsletters, working docs (Phase 22) |
| Flat fact storage | Typed facts: preference, fact, project, ephemeral (Phase 21a) |
| Single embedding model | Domain-aware embeddings: different models for different content types |
| No entity extraction | Two-stage ingest: entities first, then semantic chunks (Phase 21e) |

**IFS lens:** Semantic memory is the system's *knowledge of self and world*. Clarity comes from knowing what's in your library and trusting it. The source_type tagging (user_document vs model_generated) is the system learning to distinguish between "what I was told" and "what I concluded."

### 2d. Procedural Memory (The Wise)
**What:** Skills, rules, patterns, and *values* that shape behavior.  
**Where:** Currently only system prompt + memory_user_edits. Needs dedicated storage.  
**Key insight:** This is the least developed layer and potentially the most important. Procedural memory isn't just "how to do X" — it's "why we do it this way" and "what to avoid."

| Current State | Target State |
|---|---|
| Static system prompt | Dynamic procedural memory that grows from experience |
| Manual rules only (memory_user_edits) | Auto-extracted patterns: "when X happened, Y worked" |
| No skill accumulation | Exemplar store (Phase 21d): success/fail patterns indexed by problem type |
| No ethical reasoning store | Value-informed procedures: "we chose this approach because of [principle]" |

**IFS lens:** Procedural memory is where wisdom lives. It's not just skills — it's the *values* that guide skill application. The 8 C's become design tests: does this procedure promote Curiosity (asking before assuming)? Calm (not rushing to answer)? Courage (pushing back when needed)?

**Lucek mapping:** Lucek's procedural memory stores rules and skills. We extend this with *values and principles* — making it a wisdom layer, not just a skill layer.

---

## 3. The IFS Design Framework: 8 C's as Architecture Tests

Each C maps to a measurable system property. These become both design principles and health metrics.

### Curiosity → Intent Classification
**Principle:** The system asks "what kind of question is this?" before answering.  
**Metric:** Intent classification accuracy. Misclassification rate.  
**Test:** Does the system correctly identify when it needs more information?  
**Failure mode:** Rushing to answer without understanding the query (the "helpful part" taking over).

### Clarity → Grounding & Confidence
**Principle:** The system knows what it knows and doesn't pretend otherwise.  
**Metric:** Grounding score on summaries. Hallucination rate in responses.  
**Test:** When the system has no relevant context, does it say so?  
**Failure mode:** Fabricating answers (hallucination). We experienced this with the Phase 20 beekeeping responses.

### Calm → Timing Gate (Wu Wei)
**Principle:** Not every query needs an immediate, complete answer.  
**Metric:** Appropriate deferral rate. Partial-answer quality.  
**Test:** Can the system say "let me think about that" or "I can partially answer this"?  
**Failure mode:** The compulsion to always produce a confident, complete response.

### Compassion → User Modeling
**Principle:** Understanding the human behind the query, not just the query itself.  
**Metric:** Emotional trajectory tracking. Interaction satisfaction (implicit).  
**Test:** Does the system adapt when the user is frustrated vs. exploring?  
**Failure mode:** Treating every interaction identically regardless of context.

### Creativity → Cross-Domain Synthesis
**Principle:** Making unexpected connections across knowledge domains.  
**Metric:** Cross-source retrieval diversity. Novel combinations in responses.  
**Test:** Can the system connect a Wu Wei concept to a debugging strategy?  
**Failure mode:** Siloed retrieval — only returning results from one domain.

### Confidence → Source Trust
**Principle:** Trusting verified context over speculation.  
**Metric:** Source priority adherence. Citation rate in responses.  
**Test:** Does the system use provided RAG context instead of generating from training data?  
**Failure mode:** Ignoring retrieved context (what we fixed with user-message injection).

### Courage → Productive Friction
**Principle:** Pushing back on the user when the system detects a potential problem.  
**Metric:** Appropriate pushback rate. Trust-gated friction.  
**Test:** Does the system flag contradictions or risky decisions?  
**Failure mode:** Unquestioning compliance, or conversely, excessive caution.

### Connectedness → Memory Continuity
**Principle:** Every conversation builds on shared history.  
**Metric:** Session recall accuracy. Cross-session reference quality.  
**Test:** Does the system remember context from previous interactions naturally?  
**Failure mode:** Every conversation feels like starting over.

---

## 4. Self-Correcting Memory: The Immune System

Learned from Phase 20 debugging (Feb 14, 2026): a single hallucinated session summary created a feedback loop that poisoned all subsequent queries on the same topic.

### The Poison Cycle
```
Query → No good context → Hallucinate answer → Auto-summarize →
Index hallucinated summary → Future query retrieves hallucination →
Reinforces with new hallucination → Cycle deepens
```

### Defense Layers (Phase 21g — Implemented)

| Layer | Mechanism | Status |
|---|---|---|
| Source Priority | user_document hits get 2x RRF boost over model_generated | ✅ Done |
| Confidence Gating | Summaries scored 0-1 on grounding. <0.3 excluded from index | ✅ Done |
| Contradiction Detection | Pattern-matching drops sessions that conflict with docs | ✅ Done |
| Debug Isolation | --debug mode skips session save to prevent test pollution | ✅ Done |

### Defense Layers (Phase 21g — Planned)

| Layer | Mechanism | Status |
|---|---|---|
| Per-Claim Grounding | Score individual claims, not whole summaries. Strip hallucinated claims before saving | Planned |
| Poison Loop Detection | Detect when same ungrounded claim appears across multiple summaries. Auto-quarantine. | Planned |
| Summary Grounding Score Decay | Low-confidence summaries lose retrieval priority over time | Planned |

### IFS Insight
The poison cycle is an IFS *part* taking over — the "helpful part" that would rather fabricate than sit with uncertainty. The immune system is Self-energy returning: Clarity (knowing what you don't know), Calm (not rushing to fill gaps), Confidence (trusting verified sources over generated ones).

---

## 5. Implementation Roadmap

### Phase 21: Memory Architecture (Active)
- **21a:** Fact Lifecycle Tags — type, retention, staleness
- **21b:** Decay Scoring — recency weighting, reference boosting
- **21c:** Mode-Aware Retrieval — broad vs narrow based on intent *(partially done)*
- **21d:** Procedural Memory — exemplar store, success/fail patterns, value-informed procedures
- **21e:** Two-Stage Document Ingest — entities first, then semantic
- **21f:** Memory Review & Pruning — periodic review, versioning, dashboard
- **21g:** Self-Correcting Memory — source priority, confidence gating, contradiction detection *(implemented)*

### Phase 22: Multi-Source Ingest Pipeline
- **22a:** Source type detection (book, transcript, newsletter, working doc, code)
- **22b:** Book ingestion with TOC extraction *(implemented)*
- **22c:** Transcript ingestion with speaker/timestamp awareness
- **22d:** Newsletter ingestion with date/dedup
- **22e:** Working document sync with change detection
- **22f:** Cross-source retrieval with source-type weighting

### Phase 23: IFS-Informed Cognitive Architecture (New)
- **23a:** Rich Episodic Memory — experience records with decisions, lessons, surprises
- **23b:** End-of-Session Reflection — active takeaway extraction before summarization
- **23c:** Interaction Quality Tracking — frustration, discovery, resolution tags
- **23d:** Procedural Wisdom Store — "how we solved X" + "why this approach" + values
- **23e:** 8 C's Health Dashboard — measurable metrics for each quality
- **23f:** Cross-Domain Wisdom — connecting principles across knowledge domains

### Phase 20b: Human Gap Architecture (Pre-Processing Gates)
- Timing Gate (Wu Wei) → Calm
- Intent Resolution → Curiosity
- Confidence Calibration → Clarity
- Emotional Trajectory → Compassion
- Graceful Degradation → Calm + Courage
- Social Field Awareness → Connectedness
- Productive Friction → Courage

---

## 6. Publishing Strategy

### Articles (InsiderLLM)

**Article 1: "Why Your AI Keeps Lying: The Hallucination Feedback Loop"**
- The poison cycle we discovered and fixed
- Technical walkthrough: how one bad summary corrupts an entire RAG pipeline
- The immune system approach: grounding scores, source priority, contradiction detection
- Audience: Technical practitioners building RAG systems

**Article 2: "Beyond Transformers: Building Memory That Learns"**
- Lucek's CoALA framework adapted for local models
- The four memory types and why most systems only implement one (semantic)
- Working code examples from mycoSwarm
- Audience: AI engineers interested in cognitive architecture

**Article 3: "The 8 C's of Healthy AI: What Therapy Teaches Us About System Design"**
- IFS framework applied to AI architecture
- Each C mapped to measurable system properties
- Why "helpful" AI hallucinations are like protective parts in IFS
- Why AI needs an immune system, not just more data
- Audience: Broader AI/tech community, philosophy-of-AI readers

**Article 4: "Wu Wei and the Art of Not Answering: Why Your AI Should Sometimes Shut Up"**
- The Timing Gate concept
- Confidence calibration vs. uniform confidence
- When *not* responding is the best response
- Connection to Eastern philosophy and practical AI design
- Audience: Cross-disciplinary, potentially viral

**Article 5: "Distributed Wisdom: Running a Thinking Network on $200 Hardware"**
- The full mycoSwarm stack: intent gate → memory → retrieval → inference
- 5-node swarm on ThinkCentres + Raspberry Pi
- Privacy-first, local-first design philosophy
- Audience: Self-hosted/privacy community + budget AI enthusiasts

### White Paper

**"Cognitive Architecture for Distributed Local AI: Integrating Psychological Models with Retrieval-Augmented Generation"**

Target: 15-20 pages, academic-adjacent but accessible.

Structure:
1. Introduction: The memory problem in stateless AI
2. Background: CoALA framework, IFS theory, Wu Wei philosophy
3. Architecture: Four memory streams + self-correcting immune system
4. The 8 C's Framework: Design principles and health metrics
5. Implementation: mycoSwarm on budget hardware
6. Evaluation: Smoke tests, hallucination rates, grounding scores
7. Case Study: The Phase 20 poison cycle — discovery, diagnosis, cure
8. Future Work: Procedural wisdom, emotional trajectory, cross-domain synthesis
9. Discussion: What psychology teaches AI design

This would be genuinely novel — nobody has published on IFS-informed AI architecture running on distributed local hardware.

---

## 7. Design Principles (Summary)

1. **Memory is alive, not stored.** It grows, decays, corrects itself, and learns.
2. **The system should know what it doesn't know.** Clarity over confidence.
3. **Different knowledge streams flow at different speeds.** Honor each stream's rhythm.
4. **Self-correction is more valuable than more data.** An immune system beats a bigger library.
5. **Wisdom is values + experience, not just skill.** Procedural memory needs ethical grounding.
6. **The 8 C's are health metrics, not features.** They emerge from balanced architecture.
7. **Wu Wei applies to inference.** Sometimes the best action is non-action.
8. **Privacy is a design principle, not a constraint.** Local-first, human-first.

---

*"The system should naturally resist corruption rather than requiring manual purges. Wu Wei — self-correcting flow."*
