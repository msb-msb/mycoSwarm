# Research Notes: Memory Architecture References

## Date: 2026-02-15
## Source: Adam Lucek's Agentic Memory video resources

---

## 1. CoALA: Cognitive Architectures for Language Agents
**Authors:** Sumers, Yao, Narasimhan, Griffiths (Princeton)
**Published:** TMLR, February 2024
**PDF:** https://arxiv.org/pdf/2309.02427
**Repo:** https://github.com/ALucek/agentic-memory

### Key Concepts
- Framework organizing language agents along three dimensions: **memory**, **action space**, **decision-making**
- Memory divided into **working** (context window) and **long-term** (episodic, semantic, procedural)
- Action space divided into **internal** (reasoning, retrieval, learning) and **external** (tools, environment)
- Decision-making as interactive loop: retrieve → reason → act
- Production systems analogy: LLMs as probabilistic production rules, cognitive architectures as control flow

### Relevance to mycoSwarm
- Our intent classifier = their "decision procedure"
- Our search_all() = their "retrieval" internal action
- Our summarize_session_rich() = their "learning" internal action
- We're ahead: self-correcting memory (21g) is described as needed but not implemented in CoALA
- Gap: they distinguish internal vs external actions more cleanly than we do

### For White Paper
- Primary theoretical framework citation
- Figure 1C is essentially what we're building
- We extend CoALA with IFS design principles and Wu Wei timing — novel contribution

---

## 2. A Survey on the Memory Mechanism of LLM-based Agents
**Authors:** Zhang, Bo, Ma, Li, Chen, Dai, Zhu, Dong, Wen (Renmin University + Huawei)
**Published:** April 2024
**PDF:** https://arxiv.org/pdf/2404.13501
**Repo:** https://github.com/nuster1128/LLM_Agent_Memory_Survey

### Key Taxonomy
- **Memory Sources:**
  - Inside-trial (within conversation) — context window management
  - Cross-trial (across conversations) — persistent memory
  - External knowledge — RAG, knowledge bases
- **Memory Forms:**
  - Textual — stored as text, retrieved via embedding/BM25 (what we do)
  - Parametric — fine-tuned into model weights (what Phase 23 moves toward)
- **Memory Operations:**
  - Writing — how memories are created (our summarize_session_rich)
  - Management — how memories are organized/pruned (our 21a/21b/21f)
  - Reading — how memories are retrieved (our search_all with RRF)
- **Evaluation:**
  - Direct: subjective + objective evaluation of memory quality
  - Indirect: conversation quality, QA accuracy, long-context performance

### Relevance to mycoSwarm
- We're purely textual memory — parametric (fine-tuning) is Phase 23
- Inside-trial memory management = Phase 24 (Adaptive Context Strategy)
- Their "memory management" maps to our 21a (lifecycle), 21b (decay), 21f (pruning)
- Their evaluation framework useful for 29d (8 C's Dashboard)

### For White Paper
- Comprehensive survey citation for background section
- Their taxonomy validates our architecture choices
- We extend with self-correction and IFS — not covered in survey

---

## 3. LangChain Blog: Memory for Agents
**Author:** Harrison Chase (LangChain)
**Published:** December 4, 2024
**URL:** https://blog.langchain.com/memory-for-agents/

### Key Insights
- "Memory is application-specific" — what a coding agent remembers differs from a research agent
- Validates typed facts (21a) and domain-aware retrieval
- Two update approaches: "in the hot path" (during response) vs "in the background" (async)
- CoALA mapping: procedural = LLM weights + agent code, semantic = facts/knowledge, episodic = past interactions
- User feedback as memory trigger — relevant to 29b reflection

### Relevance to mycoSwarm
- We do "in the hot path" (summarize on exit) — could add background processing later
- Their procedural memory = system prompt + code. Ours extends to exemplar store (21d)
- Validates our approach of separating memory types

---

## 4. LangMem SDK
**Published:** May 13, 2025
**URL:** https://blog.langchain.com/langmem-sdk-launch/

### Key Concepts
- Prompt optimizer: literally rewrites system prompt based on accumulated lessons
- Three optimization algorithms: metaprompt, gradient, simple prompt_memory
- More aggressive than our Phase 29 approach
- Episodic = recalling specific experiences (our rich episodes)
- Procedural = learning HOW to do things (our Phase 21d)
- Semantic = facts and knowledge (our facts.json + ChromaDB)

### For mycoSwarm
- Their prompt optimizer concept could inform a future phase
- We could auto-update the system prompt based on accumulated lessons
- More aggressive than storing lessons in ChromaDB — directly modifies behavior

---

## 5. Psychology Today: Types of Memory
**URL:** https://www.psychologytoday.com/us/basics/memory
(General reference on human memory types — semantic, episodic, procedural, working)

### For White Paper
- Background citation for human memory types
- Maps to CoALA framework and our architecture

---

## 6. Adam Lucek's Agentic Memory Repo
**URL:** https://github.com/ALucek/agentic-memory
**Implementation:** Jupyter notebook demonstrating 4 memory types with LangGraph

### Architecture
- Working Memory: conversation context
- Episodic Memory: historical experiences + takeaways (what we built in 29a)
- Semantic Memory: RAG knowledge base (what we have)
- Procedural Memory: rules file loaded into system prompt (simpler than our 21d plan)

---

## mycoSwarm's Unique Contributions (for white paper)

What nobody else has:

1. **Self-correcting memory immune system** (Phase 21g)
   - Grounding scores, contradiction detection, poison loop prevention
   - CoALA describes the need but doesn't implement it
   - LangChain/LangMem don't address hallucination feedback loops

2. **IFS-informed design principles** (Phase 29)
   - 8 C's as measurable system health metrics
   - Psychological framework applied to AI architecture
   - No precedent in literature

3. **Wu Wei timing gate** (Phase 20b)
   - "Should I act now, later, or not at all?"
   - Eastern philosophy applied to inference timing
   - No precedent in literature

4. **Distributed local-first architecture**
   - 5-node swarm on $200 budget hardware
   - Privacy-first: no data leaves the network
   - Intent classification distributed across CPU nodes
   - Nobody else runs cognitive architecture on ThinkCentres

5. **Rich episodic memory with emotional tone** (Phase 29a)
   - Structured experience records: decisions, lessons, surprises, tone
   - Lessons indexed separately for procedural retrieval
   - Goes beyond LangChain's episodic memory (which is just conversation logs)

6. **Fact lifecycle management** (Phase 21a)
   - Typed facts with reference tracking and staleness detection
   - Ephemeral facts with shorter decay windows
   - More granular than LangChain's key-value store approach
