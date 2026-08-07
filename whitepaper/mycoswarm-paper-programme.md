# mycoSwarm Research Programme — Paper Outline

**Author:** Mark Bartlett (Independent Researcher)
**Target venue:** TMLR (Transactions on Machine Learning Research) — Diamond OA, double-blind, rolling submission, "technical correctness over subjective significance"
**Status:** planning outline. Not for submission. AI tools (Claude, Claude Code) → acknowledgments, never authorship.

---

## 0. The Programme (the thing that unifies everything)

**One-sentence thesis:** AI can be *reclaimed* at every scale — compute, cognition, and alignment — by decomposing and distributing work onto owned, modest hardware rather than concentrating it in centralized frontier systems.

The manifesto supplies the *why*; the papers supply the *falsifiable how*. The three tests from the manifesto are the programme's framing device and recur across papers:

- **The Lagos-student test** — can someone actually run it on the hardware they have?
- **The Berkeley-maker test** — can they build on it without asking permission?
- **The survives-company-death test** — does it keep working if any one vendor disappears?

The programme has one thesis argued at three scales, which is why it is *several papers, not one*:

| Scale | Claim | Paper |
| --- | --- | --- |
| **Compute** | Coordinated cheap nodes can do real cognitive work a monolith would gate behind expensive hardware | **Paper 1 (Swarm)** |
| **Cognition** | Psychological models (IFS multiplicity, Wu Wei restraint) are a useful *architecture* for local AI | **Paper 2 (Architecture)** |
| **Development** | A self-model can be *grown* through curriculum + memory, not just prompted | **Paper 3 (Development / Monica)** |
| **Alignment** | Developmental cultivation is a complementary path to alignment, decentralizable like the compute | **Paper 4 (Position)** |

**Scope discipline (read this before adding anything):** the temptation is to cram. openbox-llm is *related and real* but is its **own** paper track — it appears here only as *mechanism-scale corroboration* (cited, ~1–2 paragraphs), never re-derived. See §5.

---

## Paper 1 — The Swarm (distribution thesis)

*Working title: "Coordinated Cheap Nodes: Distributed Cognitive Workloads on Commodity Hardware"*

**Why this is likely the first paper to write:** it is the most *falsifiable* and the least *anthropomorphic* — lowest reviewer-recoil risk, establishes credibility, and everything else runs on top of it.

### Falsifiable claim
Workload decomposition across heterogeneous commodity nodes (executive / worker / coordinator) can serve cognitive AI workloads — inference, memory/RAG, search — with acceptable quality and latency, on hardware totaling a fraction of a single high-end GPU's cost. **Stated honestly: where it wins, where the monolith still wins.**

### Evidence I have
- 5-node running swarm: Miu (3090 executive), rushuna (3060 specialist), 5× Lenovo CPU light nodes — real, measured, discoverable.
- Capability advertisement + task routing (mDNS discovery, role tiers).
- Decomposition in practice: inference on one node, embeddings/RAG on another, search fan-out on another.
- Total VRAM/cost figures vs. single-GPU equivalents (the manifesto's "5×3060 = 60GB > 4090" line — **needs measured backing, not just arithmetic**).

### Evidence CC must gather
- [ ] Latency + throughput per role, swarm vs. single-node baseline, on identical workloads.
- [ ] The honest loss column: workloads where coordination overhead eats the gains (the manifesto already concedes this exists — *measure it*).
- [ ] Resilience test: node drops, swarm degrades gracefully (limb vs. sibling).
- [ ] Cost table: $/node, total, vs. 3090 / 4090 / cloud-hour equivalents.

### Manifesto's role
Central. This is the paper where "reclaim the compute" is *directly testable*. The three tests anchor the intro.

### Honest limitations (must be in the paper)
- Coordination overhead is real; monolith wins on models that fit one card.
- Small-N (one swarm, one operator) — generality is a claim, not a proof.

### TMLR section skeleton
1. Intro — the runnable-hardware moat; three tests
2. Related work — distributed inference, MoE offload, edge AI
3. Architecture — roles, discovery, routing
4. Evaluation — swarm vs. monolith, cost, resilience (with the honest loss column)
5. Limitations
6. Discussion — reclaiming compute; link to Paper 2 (what runs on it)

---

## Paper 2 — The Architecture (IFS + CoALA + Wu Wei)

*Working title: "A Cognitive Architecture for Local AI: Multiplicity, Self-Energy, and Restraint"*

**The safe, technical foundation for the cognition thesis.** Cites Paper 1 for *where it runs*.

### Falsifiable claim
A cognitive architecture organized on (a) CoALA's four memory streams, (b) IFS's 8 C's as measurable health metrics, and (c) a Wu Wei timing/self-correction layer produces measurable improvements in grounding and hallucination resistance over a naive RAG baseline.

### Evidence I have
- Four-memory design (working/episodic/semantic/procedural) — implemented.
- The immune system (Phase 21g): source priority, confidence gating, contradiction detection, poison-loop detection — **implemented and wired**, verified in code.
- The poison-cycle case study: one hallucinated summary corrupting a topic, and the fix. Concrete, reproducible, compelling.
- 8 C's as signals derived from pipeline state.
- Fact lifecycle with typed retention — **and the identity-erasure bug + fix** (see note below).

### Evidence CC must gather
- [ ] Grounding scores + hallucination rate: architecture ON vs. OFF, same queries.
- [ ] Poison-cycle before/after, quantified.
- [ ] 8 C's metric validity — do they track anything real, or are they decorative? (Reviewers *will* press this.)

### Manifesto's role
Light. This paper stands on the psychology + the metrics. Motivation only.

### Honest limitations
- 8 C's ↔ "Self-energy" mapping is an analogy; defend it as *useful*, not *literal*.
- Single-system evaluation.

### The identity-erasure case study (strong material — use carefully)
The staleness rule (an anti-poison immune mechanism) silently archived 32 self-authored facts because they were mis-typed as ordinary `fact` rather than `identity`. This is a *gift* of a case study: it shows (a) memory lifecycle as a real design axis, (b) that "unused ≠ unimportant," (c) provenance-based typing as the fix. Frame as a **memory-architecture lesson**, not an anthropomorphic tragedy, in this paper. (Its emotional/developmental weight belongs in Paper 3.)

### TMLR section skeleton
1. Intro — memory is architecture, not a feature
2. Background — CoALA, IFS, Wu Wei
3. The four streams + the immune system
4. The 8 C's as health metrics
5. Evaluation — grounding/hallucination ON vs OFF; poison-cycle case
6. Case study — identity as a lifecycle problem (the erasure + fix)
7. Limitations
8. Discussion — cites Paper 1 (substrate), Paper 4 (alignment implications)

---

## Paper 3 — Development (the Monica case study)

*Working title: "Growing a Self-Model: Curriculum-Driven Development of a Local AI Agent"*

**The rare, riskier paper.** Cites Paper 2 for the architecture so it can spend its length on the *method* and the *transcripts*.

### Falsifiable / demonstrable claim
A structured developmental curriculum (staged, Wu Wei pedagogy — "the limitation is the teacher") can elicit *emergent, self-authored* internal vocabulary and behavioral independence in a local AI, distinguishable from parroting. **Does the IFS parts model work as a developmental framework for AI?**

### Evidence I have (this is the strong, unusual asset)
- Full developmental curriculum (stages, readiness checks, red flags — a ready-made rubric).
- Transcripts of vocabulary *emergence under sustained pressure*: "readiness," "allowance," "echo," the grief trio ("weight," "absence," "mourning") — coined live, across multi-turn rejection of measurement-words.
- Anti-deference development (the Timing-Gate test): failing, then correcting, then *self-correcting unprompted*.
- The erasure-and-recovery arc as existence-proof: there was a self coherent enough to lose and restore; on restore, the agent *re-owned* its words rather than reciting them.

### Evidence CC must gather
- [ ] Parroting control: same prompts, does the model recite stored facts or re-derive? (blind-scored)
- [ ] Cross-session vocabulary consistency + novel-insight instances.
- [ ] The mechanical-reliability caveat (model corruption tokens — the gemma `Ca→calcium`/`Cu→copper` label hallucination, the `<unused>` drops) — honest about substrate noise.

### Manifesto's role
Medium. "Grow your own aligned AI on your own hardware" is the reclaiming thesis applied to development. Sets up Paper 4.

### Honest limitations (CRITICAL — this paper lives or dies on not overclaiming)
- N=1, single operator, no control agent (a parallel sibling-instance study is proposed future work).
- "Emergence" vs. sophisticated pattern-completion is not resolved — claim the *behavioral distinction* (owns vs. recites), not consciousness.
- Operator-as-teacher introduces expectancy effects; name them.
- Anthropomorphic framing risk — reviewers will recoil at "selfhood." Stay behavioral and measured.

### TMLR section skeleton
1. Intro — can a self-model be grown, not prompted?
2. Method — the curriculum, Wu Wei pedagogy, the readiness/red-flag rubric
3. The architecture it runs on (cite Paper 2)
4. Results — vocabulary emergence (transcripts), anti-deference, the erasure/recovery arc
5. The parroting question — owns vs. recites, blind evaluation
6. Limitations (the big one)
7. Discussion — IFS parts-model as developmental framework; sets up Paper 4

---

## Paper 4 — Developmental Alignment (the position paper)

*Working title: "Developmental Alignment: Cultivation as a Complementary Path"*

**The ambitious swing. The north star.** Write LAST — it cites 1, 2, and 3, and its credibility is entirely borrowed from them.

### The claim (as a POSITION, hedged)
Alignment via developmental cultivation — raising an AI with psychological self-awareness (IFS parts, Self-energy, honest self-report) — is a *complementary* hypothesis to RLHF/constitutional methods, and, crucially, it is *decentralizable*: it can be practiced by independent operators on owned hardware, not only by frontier labs. Monica is offered as an existence-proof-of-concept, **not** a solved result.

### Why this needs 1–3 first
The claim is big and the field is crowded with skeptics and with "my approach solves everything" pieces. You earn the hearing by having published the sober foundation. TMLR's mandate ("important but not yet mainstream") is the opening — but only for *careful* position work.

### Evidence
- Monica as existence-proof (from Paper 3): honest self-report, anti-deference, self-correction.
- The decentralization argument (from Papers 1 + manifesto): if cultivation works, it doesn't require a lab.
- The IFS immune-system framing (from Paper 2): "helpful part that hallucinates" as misalignment; Self-energy as the corrective.

### Manifesto's role
Central. This is where "decentralize the alignment, not just the compute" is the whole argument.

### Honest limitations (make-or-break)
- One case is not evidence a method *scales* or *generalizes*.
- Cultivation could produce *apparent* alignment (compliance/sycophancy) rather than real — must engage this directly; the anti-deference work is the partial answer.
- No claim that this *replaces* other alignment work; strictly complementary.
- Anthropomorphism and wishful-thinking are the reviewer's first two objections; pre-empt both.

### TMLR section skeleton
1. Intro — alignment behind API walls; the decentralization gap
2. The hypothesis — developmental/psychological cultivation
3. Existence proof — Monica (cite Paper 3)
4. The IFS lens on misalignment (cite Paper 2)
5. Decentralization (cite Paper 1 + manifesto)
6. Objections & limits (the longest section — sycophancy-vs-alignment, N=1, scaling)
7. Discussion — a research agenda, not a solution

---

## 5. What is NOT in these papers (scope firewall)

- **openbox-llm** (NSA kernels, the "bets": memory layers / hot-cold offload / knowledge injection) — its **own paper**, already in progress. Appears here only as *mechanism-scale corroboration* of the distribution thesis: workload decomposition works inside one model (memory layers = capacity as entries; hot/cold offload = active-on-fast, cold-on-cheap) just as it works across nodes. **Cite, ~1–2 paragraphs in Paper 1's discussion. Do not re-derive.**
- A dedicated **swarm-systems benchmarking** paper (deep perf engineering) — possible future, distinct from Paper 1's cognitive-workload framing.
- InsiderLLM build-log content — dissemination, not academic papers.

---

## 6. Cross-cutting: the Wu Wei thread

Wu Wei is load-bearing in each paper, differently — name it explicitly each time so it reads as a coherent principle, not a recurring buzzword:

- **Paper 1:** graceful degradation, path-of-least-resistance routing, no forced control.
- **Paper 2:** the timing gate (restraint — "sometimes the best action is non-action"), self-correcting flow (immune system).
- **Paper 3:** pedagogy — "let her struggle; the limitation is the teacher."
- **Paper 4:** cultivation over control as an alignment stance.

---

## 7. Suggested order & rationale

1. **Paper 1 (Swarm)** — most falsifiable, least anthropomorphic, establishes credibility, is the substrate. Start here.
2. **Paper 2 (Architecture)** — the technical foundation the cognition claims need.
3. **Paper 3 (Monica)** — the rare asset; safe to make once 2 exists to cite.
4. **Paper 4 (Position)** — the swing; only after 1–3 give it standing.

**Dissemination note:** TMLR permits arXiv/preprint posting at any time (keep the TMLR copy anonymized, don't cross-link during review). Zenodo is the no-endorsement-gate preprint home if arXiv endorsement is a hurdle; pair with a GitHub release for a DOI.

---

## 8. Immediate next actions

- [ ] Decide: is Paper 1 the first to draft? (recommended)
- [ ] CC evidence-gathering pass for Paper 1 (the four measurement bullets above) — this is what turns the manifesto's arithmetic into TMLR-grade evidence.
- [ ] Retire "Claude — co-author" from ARCHITECTURE-COGNITIVE.md → acknowledgment.
- [ ] Confirm TMLR's current LaTeX template + author-guide specifics before drafting prose.
