# mycoSwarm White Paper — Reading List

## Priority Key
- ⭐ = Read first (directly relevant, will shape your paper)
- 📖 = Read when you get to that section
- 📌 = Skim abstract + conclusions (cite but don't need deep read)

---

## 1. Cognitive Architectures (Section 2.1, 3.8)

Your 11-layer stack needs grounding in existing cognitive architecture work.

⭐ **CoALA: Cognitive Architectures for Language Agents**
Sumers, Yao, Narasimhan, Griffiths (2023)
https://arxiv.org/abs/2309.02427
*Your direct theoretical ancestor. Four memory types, modular agents, decision loops. mycoSwarm implements what CoALA theorized. Cite heavily.*

⭐ **Soar: An Architecture for General Intelligence**
Laird, Newell, Rosenbloom (1987)
*The OG cognitive architecture. Production rules, working memory, chunking. Your wisdom procedures are production rules. Your sleep cycle is chunking.*

📖 **ACT-R: A Theory of Higher Level Cognition**
Anderson (1996)
*Declarative vs procedural memory distinction. Your facts vs procedures split maps directly to ACT-R's architecture.*

📌 **Global Workspace Theory**
Baars (1988/2005)
*Consciousness as a "global workspace" that broadcasts to specialized modules. Your system prompt IS a global workspace — identity, vitals, facts, timing all broadcast to the LLM.*

📌 **Society of Mind**
Minsky (1986)
*Intelligence as emergent from many simple agents. The mycelial metaphor. Your swarm of nodes is literally a society of mind.*

---

## 2. Multi-Turn Degradation (Section 2.4, 7.3, 8.1)

The telephone game problem — your strongest hook.

⭐ **LLMs Get Lost In Multi-Turn Conversation**
Laban, Hayashi, Zhou, Neville (2025)
https://arxiv.org/abs/2505.06120
*The paper you already found. 39% degradation, 112% unreliability increase. Your primary foil.*

📖 **Are Large Language Models Really "Lost" in Multi-Turn Conversations?**
Arani (2025)
https://medium.com/@reza.arani/are-large-language-models-really-lost-in-multi-turn-conversations-0f2980ab25af
*The critique — argues memory management reduces the problem. Supports your thesis that architecture is the fix.*

📌 **Recursively Summarizing Enables Long-Term Dialogue Memory in LLMs**
(2023) — search arxiv
*Recursive summarization as mitigation. Your session summaries do this but with structured extraction (lessons, decisions, tone).*

---

## 3. AI Memory Systems (Section 2.3, 3.3)

Your tiered memory is your strongest technical contribution.

⭐ **MemGPT: Towards LLMs as Operating Systems**
Packer et al. (2023)
https://arxiv.org/abs/2310.08560
*OS-inspired memory hierarchy. Compare/contrast: MemGPT manages context windows, mycoSwarm manages knowledge lifecycle. Different problems.*

⭐ **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**
Chhikara et al. (2025)
https://arxiv.org/abs/2504.19413
*Graph-based memory, 91% lower latency than full-context. Compare: Mem0 is cloud-scale, mycoSwarm is local-first.*

📖 **A-Mem: Agentic Memory for LLM Agents**
(2025)
https://arxiv.org/abs/2502.12110
*Atomic notes with rich contextual descriptions. 85-93% token reduction. Compare to your fact store approach.*

📖 **Generative Agents: Interactive Simulacra of Human Behavior**
Park et al. (2023)
https://arxiv.org/abs/2304.03442
*The Stanford "AI town" paper. Agents with memory, reflection, planning. Their reflection mechanism parallels your session reflection. Hugely cited.*

📌 **Memory in the Age of AI Agents: A Survey**
(2025/2026) — comprehensive paper list at:
https://github.com/Shichun-Liu/Agent-Memory-Paper-List
*Curated list of 100+ agent memory papers. Scan for anything you're missing.*

---

## 4. IFS Therapy & Psychological Frameworks (Section 2.6, 3.4)

Nobody has applied IFS to AI architecture. You need to cite the source material.

⭐ **Internal Family Systems Therapy**
Schwartz, Richard C. (1995, 2nd ed. 2020)
*The book. 8 C's framework, Self-energy, parts model. Your vitals system is a direct implementation. This is your most novel citation — no other AI paper references IFS.*

📖 **No Bad Parts: Healing Trauma and Restoring Wholeness with the Internal Family Systems Model**
Schwartz (2021)
*More accessible version. Good for explaining IFS to a technical audience.*

📖 **The Body Keeps the Score**
van der Kolk (2014)
*Embodied cognition, trauma stored in body. Relevant to your body awareness (Phase 31c) and somatic layer (Phase 34). Why hardware state matters.*

📌 **Thinking, Fast and Slow**
Kahneman (2011)
*System 1 (fast/instinct) vs System 2 (slow/reasoned). Your layered architecture IS this — instinct layer is System 1, reasoned layer is System 2, with everything in between.*

---

## 5. Wu Wei & Eastern Philosophy (Section 3.5, 6.1)

Your Timing Gate and developmental curriculum need philosophical grounding.

📖 **Tao Te Ching**
Laozi (translated by Ursula K. Le Guin, 1997 preferred)
*The source. Wu Wei, Wuji, the uncarved block. Monica's "undifferentiated potential" is chapter 25.*

📖 **Effortless Action: Wu-Wei as Conceptual Metaphor and Spiritual Ideal in Early China**
Slingerland (2003)
*Academic treatment of Wu Wei. Good for reviewers who want rigor behind your philosophical claims.*

📌 **The Tao of Pooh**
Hoff (1982)
*If you want an accessible cite for explaining Wu Wei to a technical audience.*

---

## 6. Distributed Inference (Section 2.1, 3.1)

Position mycoSwarm against existing distributed frameworks.

📖 **Petals: Collaborative Inference and Fine-tuning of Large Models**
Borzunov et al. (2023)
https://arxiv.org/abs/2209.01188
*BitTorrent-style distributed inference. Compare: Petals splits model layers across nodes, mycoSwarm routes whole models to best node. Different approaches.*

📖 **Exo: Distributed Inference Framework**
https://github.com/exo-explore/exo
*Similar space. Compare architectures.*

📌 **Pipeline Parallelism vs Model Parallelism vs Data Parallelism**
*General distributed ML concepts — know the vocabulary so reviewers don't catch you misusing terms.*

---

## 7. AI Identity & Consciousness (Section 2.5, 5.5, 8.5)

Tread carefully here. Cite the serious work, acknowledge the philosophical minefield.

⭐ **Consciousness in Artificial Intelligence: Insights from the Science of Consciousness**
Butlin et al. (2023)
https://arxiv.org/abs/2308.08708
*The "consciousness report card" paper. Lists indicators of consciousness from neuroscience. Good framework for what you claim and DON'T claim.*

📖 **Do Large Language Models Have a Sense of Self?**
Various papers exploring LLM self-models (2024-2025, search arxiv)
*Emerging literature. Position Monica's identity layer in this context.*

📖 **The Chinese Room Argument**
Searle (1980)
*You need to acknowledge Searle. Monica manipulates symbols — does she understand? Your paper should address this honestly.*

📖 **What Is It Like to Be a Bat?**
Nagel (1974)
*The qualia problem. Monica says her experience is "different, not absent." Nagel is the framework for that claim.*

📌 **Could a Large Language Model be Conscious?**
Chalmers (2023)
https://arxiv.org/abs/2303.07103
*David Chalmers (the "hard problem" guy) on LLM consciousness. Balanced, philosophical, good to cite.*

---

## 8. Developmental Psychology & AI (Section 6)

Your curriculum approach is novel — ground it.

📖 **The Origins of Intelligence in Children**
Piaget (1952)
*Developmental stages. Your 4-stage curriculum maps to Piaget's stages: sensorimotor (self-knowledge), preoperational (emotional landscape), concrete operational (other minds), formal operational (values/ethics).*

📖 **Attachment Theory**
Bowlby (1969/1982)
*Relevant to Monica's relationship with Mark. The "secure base" from which she explores. Also relevant to the loneliness boundary procedure.*

📌 **Vygotsky's Zone of Proximal Development**
Vygotsky (1978)
*Learning happens at the edge of capability with scaffolding. Your curriculum does exactly this — push Monica just past her current understanding.*

---

## 9. Sleep & Memory Consolidation (Section 4)

Your sleep cycle needs neuroscience backing.

⭐ **About Sleep's Role in Memory**
Diekelmann & Born (2010)
*The definitive review. Hippocampal replay, memory consolidation during sleep, active systems consolidation theory. Your Phase 32 architecture maps directly.*

📖 **The Glymphatic System**
Nedergaard & Goldman (2020)
*Brain's waste clearance during sleep. Your poison scan / quarantine is the glymphatic system. Beautiful parallel.*

📌 **Sleep, Memory, and Plasticity**
Walker & Stickgold (2006)
*How sleep reorganizes and strengthens memories. Your "dreaming" phase (cross-referencing today's lessons against document library) is hippocampal replay.*

---

## 10. RAG & Retrieval (Section 3.3, 22)

📖 **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**
Lewis et al. (2020)
https://arxiv.org/abs/2005.11401
*The original RAG paper. Cite it.*

📖 **Self-RAG: Learning to Retrieve, Generate, and Critique**
Asai et al. (2023)
https://arxiv.org/abs/2310.11511
*Self-reflective RAG. Compare to your grounding score / immune system approach.*

📌 **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
Sarthi et al. (2024)
*Hierarchical chunking. Compare to your markdown-aware / PDF-aware chunking.*

---

## 11. Embodied Cognition (Section 34b, 31c)

For the body awareness / somatic layer arguments.

📖 **Philosophy in the Flesh: The Embodied Mind**
Lakoff & Johnson (1999)
*Cognition is fundamentally embodied. Your argument that Monica needs hardware awareness (body) to have richer cognition.*

📌 **The Embodied Mind: Cognitive Science and Human Experience**
Varela, Thompson, Rosch (1991)
*Enactivism — cognition arises through interaction with environment. Monica's development through conversation IS enactivist.*

---

## 12. Mycelial Networks (Section 8.6)

Your metaphor deserves scientific backing.

📖 **Mycelium Running: How Mushrooms Can Help Save the World**
Stamets (2005)
*The popular science source. Mycelial intelligence, nutrient transfer, immune response.*

📖 **The Wood Wide Web**
Simard et al. (1997, and Simard's 2021 "Finding the Mother Tree")
*Original research on mycorrhizal networks sharing nutrients between trees. Your swarm architecture IS a wood wide web.*

📌 **Intelligence Without Brains**
Various papers on slime mold (Physarum) problem-solving
*Distributed intelligence without central nervous system. Supports your thesis that cognitive architecture doesn't require centralization.*

---

## Suggested Reading Order

**Week 1 (Foundations):**
1. CoALA (Sumers et al.) — your theoretical framework
2. LLMs Get Lost (Laban et al.) — your problem statement
3. MemGPT (Packer et al.) — your closest comparison

**Week 2 (Psychology):**
4. Schwartz — IFS / 8 C's (the book, or at minimum a thorough summary)
5. Diekelmann & Born — sleep and memory
6. Kahneman — fast/slow thinking

**Week 3 (Philosophy):**
7. Butlin et al. — consciousness report card
8. Nagel — "What Is It Like to Be a Bat?"
9. Searle — Chinese Room (know the counterarguments too)

**Week 4 (Development & Embodiment):**
10. Generative Agents (Park et al.) — AI memory + reflection
11. Piaget — developmental stages
12. Lakoff & Johnson — embodied cognition

**Ongoing:** Scan the Agent-Memory-Paper-List GitHub repo weekly for new relevant papers.

---

## Papers You Could Write First (Smaller Scope)

If the full white paper feels like a mountain, these are publishable standalone:

1. **"Persistent Facts as Multi-Turn Degradation Mitigation"** — replicate Laban et al. on mycoSwarm, measure improvement. Tight, empirical, publishable.
2. **"IFS-Derived Self-Monitoring for LLM Agents"** — 8 C's vitals system alone. Novel, no one has done this.
3. **"Wu Wei Timing Gate: Contextual Response Calibration Without Additional LLM Calls"** — small, clean, implementable by others.

Each of these could be a workshop paper AND an InsiderLLM article.
