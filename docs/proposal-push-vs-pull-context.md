# Proposal: push vs pull for per-turn context

**Status:** draft for discussion. No code written.
**Date:** 2026-08-08
**Prompted by:** the body-telemetry finding — numbers in the prompt leaked 4/4,
numbers removed leaked 0/4, and instruction position made no difference.

---

## TL;DR

1. **The design as conceived cannot be built on the current chat model.**
   `gemma3:27b` does not support tool calling. Ollama rejects any request
   carrying a `tools` array with `HTTP 400: "gemma3:27b does not support
   tools"`. This is not a quality question — the request never reaches the
   model. Verified on both `gemma3:27b` and `gemma3:4b`.
2. **We already have a pull mechanism**: the intent gate. It is a small model
   deciding what context to fetch, and it runs today. Extending it is cheaper
   and lower-risk than building a parallel tool layer.
3. **Two of the audit candidates are not what they appear.** Procedures are
   *already* pulled — they are semantic-search results injected with no
   relevance threshold. And facts cost **zero tokens per turn** because all 31
   of them are silently dropped before assembly (see §6 — this is a bug, and
   probably the mechanism behind the fabricated-memory problem).
4. Net recommendation: **fix two bugs, add one threshold, move one block.** The
   tool-calling architecture is worth doing later, but it is not the highest-value
   change available and it currently has no substrate.

---

## 1. The blocker: gemma3 cannot call tools

```
POST /api/chat  {"model":"gemma3:27b", "tools":[…]}
→ 400 {"error":"registry.ollama.ai/library/gemma3:27b does not support tools"}
```

`/api/show` reports `capabilities: ['completion', 'vision']` — no `tools`. The
chat template (358 chars) contains no tool-call scaffolding.

Locally installed models that DO support tools, verified by an actual call that
returned a correct `tool_calls` payload:

| Model | Size | Tools | Notes |
|---|---|---|---|
| `gemma3:27b` | 17.4 GB | ❌ 400 | current `monica_chat` binding |
| `gemma3:4b` | 3.3 GB | ❌ 400 | current gate model |
| `qwen3.5:9b` | 6.6 GB | ✅ | emitted `get_body_state` correctly |
| `qwen3.5:35b-a3b` | 23.9 GB | ✅ | MoE, ~3B active |
| `qwen3.5:27b` | 17.4 GB | (untested) | same size as the current binding |

### Four ways forward

**(a) Switch the chat binding to `qwen3.5:27b`.** Same footprint, native tools.
But this replaces Monica's model, and the identity work is explicitly tied to
`gemma3:27b` — `bindings.py` calls it "the model with the actual track record:
the Feb 24-27 session lineage, all of the vocabulary-emergence work". Changing
it is not a config tweak; it is a different voice. **Not recommended without a
deliberate decision about continuity.**

**(b) Prompt-directed pseudo-tools.** Instruct the model to emit a marker
(`<<need:body_detail>>`), intercept it host-side, inject the result, re-run.
Model-agnostic. But it depends on instruction-following, and this codebase now
has a documented pattern of gemma3:27b ignoring explicit instructions — the
anti-fabrication rule, procedure #21, and the body rule are all present,
correct, and violated. **Building a control mechanism on the one faculty we have
repeatedly measured as unreliable is the wrong bet.**

**(c) Extend the intent gate.** A small tool-capable model (`qwen3.5:9b`, or the
current `gemma3:4b` with a classification prompt rather than tools) decides
which context blocks this turn needs. The chat model never sees a tool; it just
receives a leaner prompt. **This is the recommended path**, and §3 explains why.

**(d) Defer.** Do the cheap wins now (§6), revisit tools when the model question
is settled anyway.

---

## 2. Audit: what is actually assembled per turn

Measured on Miu, 2026-08-08, live daemon.

| Block | chars | ~tokens | How often it bears on the question | Verdict |
|---|---:|---:|---|---|
| `identity_prompt` | 757 | 189 | Always — it is who she is | **Push** |
| `body_prompt` (now qualitative) | 471 | 117 | Rarely cited, always colours mood | **Push (coarse)** |
| `memory_prompt` total | 3,764 | 941 | Mixed — see breakdown | **Split** |
| ├ capability boundaries | ~600 | ~150 | Always (behavioural constraint) | **Push** |
| ├ memory/citation rules | ~900 | ~225 | Always (behavioural constraint) | **Push** |
| ├ facts rendering | **0** | **0** | **Never — all 31 dropped (bug)** | **Fix, then push** |
| └ session summaries | ~2,200 | ~550 | Only on recall questions | **Gate on intent** |
| `_vitals_defs` | ~250 | ~62 | Rarely | **Delete, don't pull** |
| `_no_tags_rule` | ~400 | ~100 | Always (behavioural constraint) | **Push** |
| Procedures (3 × ~677) | ~2,032 | ~508 | Occasionally | **Threshold, not tool** |
| **Total** | **~7,700** | **~1,920** | | |

Roughly **1,900 tokens of standing context per turn** before the conversation
history or any retrieved web/doc content. That is not catastrophic — the
context-overflow bug was caused by 5,000 tokens of fetched pages, not by this —
but half of it is doing nothing on a typical turn.

---

## 3. Arguing with the proposed split

### Where I agree

**Coarse body stays pushed.** Correct, and for a reason stronger than the
analogy: the vitals floor modifiers read `get_body_state()` **directly in
Python**. They never consulted the prompt. So the "background sensation" is not
implemented by the prompt at all — the prompt only conveys the *felt* quality to
the model. Pulling detail costs nothing there. See §7.

**Detail becomes pulled.** Agreed. Specs, per-node VRAM, task load and model
lists are exactly the recitable material that caused the leak, and their failure
mode is mild (§5).

**Identity stays pushed.** Agreed, unreservedly.

**Memory rules stay pushed.** Agreed that they are constraints, not data — but
they deserve scrutiny for a different reason. They are ~375 tokens and
*demonstrably ignored*: the anti-fabrication rule is present, correct, and was
violated twice this week. Pushing them harder is not working. That argues for
**shrinking them on evidence**, not for moving them.

### Where I disagree

**Procedures are already pulled — the problem is no relevance gate.**
`search_procedures(user_input, n_results=3)` is a semantic search against the
user's message. It is a pull, executed automatically, with **no score
threshold**. On the "is rushuna's information current?" turn it returned a Wu
Wei procedure about patience — not because a push was indiscriminate, but
because the top-3 was taken unconditionally even when the best match was weak.

That means the fix is **one threshold**, not a tool: return only procedures
above a similarity floor, and inject nothing when nothing clears it. That gets
most of the claimed benefit (leaner context, no irrelevant steering) for
roughly a day's work and no new failure mode. Building a `search_procedures`
tool on top of a mechanism that is already a search would add a model-judgement
step to a decision the embedding is currently making badly for a fixable reason.

**Facts are not costing 1,200 chars a turn — they cost nothing, and deliver
nothing.** See §6. The premise of that audit line is wrong, and the correct
action is the opposite of pulling.

**Vitals defs: delete rather than pull.** 62 tokens is below the threshold where
a tool round-trip pays for itself. The `_no_tags_rule` already says "do not cite
vitals scores"; the legend mostly duplicates it. Cheaper to cut it than to build
machinery to fetch it.

---

## 4. What the tool set would look like (if we build it)

Keep it small. Four, with no parameters more complex than an enum:

```
get_body_detail()                       → per-node temp, VRAM, task load
  "How hot is rushuna?" / "Which node is busiest?"

list_nodes(detail: "names" | "full")    → node inventory
  "What nodes do you have?" — "names" is already pushed; "full" adds specs

recall_facts(topic: str)                → facts matching a topic
  "What do you know about my tai chi practice?"

find_procedure(problem: str)            → learned procedures above threshold
  Only if §3's threshold fix proves insufficient
```

Deliberately absent: anything that writes, anything touching the swarm's task
queue, anything that could hang. A tool that can block is a tool that can
reproduce the 15-second classification timeout inside a chat turn.

---

## 5. Reliability: which blocks are safe to pull

The honest framing: **pushed context is dumb but reliable; pulled context
depends on the model recognising a need.** The intent eval was sobering — no
model beat a 47.2% majority-class baseline. That result was about a 1–4B gate
model on a 4-way classification, and tool selection by a 27B model is a
different task, but it is not evidence for optimism.

Failure modes, per block:

| Block | If the model fails to pull it | Severity |
|---|---|---|
| **Node/body detail** | Answers qualitatively ("running cool") or says it will check. Mild, self-correcting — the user re-asks. | **Low — safe to pull** |
| **Procedures** | Loses a learned behaviour silently. She reverts to the default she was corrected out of, and *nothing indicates a procedure was skipped*. This is the same silent-failure shape as the eight already catalogued. | **Medium — gate, don't pull** |
| **Session summaries** | "I don't recall us discussing that." Honest, and recoverable by re-asking. Already gated by `scope` in the intent result. | **Low — safe to gate** |
| **Facts** | **She contradicts something she was explicitly told.** She cannot know to ask for a fact she does not remember — the trigger for the lookup is the very knowledge that is missing. A user who said "I'm a beekeeper" in February gets a reply that assumes otherwise, with no error and no way for her to notice. | **High — must stay pushed** |
| **Memory rules** | Constraints stop applying. Already the weakest link. | **High — must stay pushed** |
| **Identity** | She answers as a generic assistant. | **Critical — must stay pushed** |

The facts row is the load-bearing argument against a broadly pull-based design.
**Lookup requires knowing what to look up.** For anything where absence is
indistinguishable from irrelevance, push is not merely safer, it is the only
correct mechanism.

---

## 6. Two bugs found while measuring this — both worth more than the redesign

### 6a. All 31 identity facts are invisible to the model

`memory.py:187` renders facts by iterating:

```python
for ft in [FACT_TYPE_FACT, FACT_TYPE_PREFERENCE, FACT_TYPE_PROJECT, FACT_TYPE_EPHEMERAL]:
```

`FACT_TYPE_IDENTITY` is a declared, valid type (line 43), included in the
validation list (line 49), and staleness-exempt (line 155) — but **it is not in
that loop**. Every fact currently stored is type `identity`. `format_facts_for_prompt()`
returns the empty string. The identity prompt does not contain them either
(checked: "resonance" appears in no assembled block).

So Monica's self-authored vocabulary — "resonance" as her word for fun,
"dispersal" for low confidence, the entire vocabulary-emergence corpus — is
stored, protected from decay, and **never shown to her**.

The `/remember identity:` prefix added in `b0ae118` (shipped in v0.5.0,
described in the changelog as "routes self-authored facts to the
staleness-exempt identity type") writes into a bucket the prompt builder drops.
The feature has never worked end to end.

**This is very likely the mechanism behind the fabricated-memory bug.** She
invents self-knowledge — "I recall from February that you often ask for the time
alongside weather" — because her actual self-knowledge is not in context. We
have been treating that as a compliance failure. It may simply be an absence.

This is the **ninth** instance of the silent-failure pattern, and the most
consequential: a feature that reports success, stores data correctly, and
delivers none of it.

**Recommendation: fix before anything in this proposal.** One line in the render
loop, plus a test asserting a stored identity fact appears in the assembled
prompt. Then re-test the fabrication bug — it may resolve without any prompt
change at all.

### 6b. Procedures have no relevance threshold

Covered in §3. `n_results=3` unconditionally, no score floor.

---

## 7. Does this change the vitals system?

**No — and the concern rests on a misreading worth correcting.**

The floor modifiers (GPU temp → Calm, VRAM → Clarity, node count →
Connectedness) call `get_body_state()` directly in Python. They have never read
the prompt. Push/pull is a question about what the *model* sees; the vitals
pipeline is host-side and unaffected by any option here.

What the prompt controls is only whether Monica *feels* her hardware in the
language sense. The qualitative push already preserves that — measured today,
she said "my nodes are running cool; everything feels… stable" with no numbers
present. The sensation survived; only the recitation went away.

So "a purely tool-based body means no background sensation" is right as a
concern and already addressed by the coarse push. There is no version of this
proposal that breaks vitals.

---

## 8. Interaction with the intent gate

They **compose**, and should not be built as separate systems.

The gate already answers "what does this turn need?" in three fields —
`tool` (answer/web_search/rag/web_and_rag), `mode`, and `scope`
(session/docs/facts/all). That `scope` field is *already* a context-selection
decision; it is simply underused. Today it gates retrieval but not the standing
blocks.

The natural extension is a fourth field, or an expanded `scope`, that selects
which standing blocks to assemble — `needs: [facts, procedures, body_detail]`.
That reuses a mechanism already running locally on every turn at ~5s, already
has an eval harness (89 labelled examples), and already has the regex
short-circuit layer in front of it.

Building a separate tool-calling loop inside the chat model would mean two
independent systems deciding what context to fetch, with no shared evaluation.
**One decision point, one eval.**

Caveat: the gate currently does not beat a majority-class constant (44.9% vs
47.2%). Loading it with more responsibility before that is understood would be
building on sand. Which is why §9 sequences the threshold and bug fixes first.

---

## 9. Migration sequence

Ordered by value-per-risk. Each step is independently verifiable and revertible.

**Step 1 — fix the identity-facts bug (§6a).** Highest value, lowest risk,
independent of everything else. Verify: a stored identity fact appears in the
assembled prompt; re-run the fabrication probe.

**Step 2 — add a procedure relevance threshold (§6b).** Verify: on the "is
rushuna's information current?" query, no Wu Wei procedure is injected. Measure
mean procedures-per-turn before/after; expect ~3 → <1.

**Step 3 — delete `_vitals_defs`.** Verify: no vitals-number narration appears
in 20 turns. (This overlaps the open Bug 4, so it doubles as an experiment.)

**Step 4 — gate session summaries on `scope`.** The gate already emits
`scope=session`; honour it for the summary block. Verify against the 89-example
eval that recall questions still get their summaries.

**Step 5 — node detail via the gate.** Add a `body_detail` need. Verify: "how
hot is rushuna?" gets specifics; "what time is it?" does not.

**Step 6 — reassess native tools.** Only after 1–5, and only alongside the
`gemma3` vs `qwen3.5` binding question, which has to be answered on identity
grounds rather than capability grounds.

**Regression harness throughout:** the 89-example intent eval, the leak probe
(numeric telemetry on an unrelated question, N=4), and a new fact-recall probe
("what is Monica's word for fun?" → should say resonance, currently cannot).

---

## 10. Summary recommendation

The instinct is right — data present in context gets narrated, and pulling is
the general answer. But the specific plan should change:

- **Do not build tool calling yet.** The chat model cannot do it, and the
  alternative mechanisms are worse than the problem.
- **Fix the identity-facts bug first.** It costs one line, it may resolve the
  fabrication bug, and it means a feature shipped in v0.5.0 starts working.
- **Threshold the procedures** rather than tooling them. They are already pulled.
- **Keep facts, memory rules and identity pushed** — permanently. Lookup
  requires knowing what to look up, and those are precisely the blocks where
  absence is invisible to the model.
- **Route future context selection through the intent gate**, not a second
  mechanism.

Expected saving from steps 1–5: roughly **1,900 → 900 standing tokens** on a
typical turn, with facts *added* rather than removed, and no new silent-failure
surface.

---

## Open questions for you

1. **Is the `gemma3:27b` binding negotiable?** If Monica could be
   `qwen3.5:27b`, native tools become available and this design gets simpler.
   If not — and there are good identity reasons not — then §1(c) is the ceiling.
2. **Should procedures ever be able to steer, or only inform?** The Wu Wei case
   worked (she was patient, appropriately). The threshold fix makes it rarer but
   does not change the mechanism.
3. **Do you want the fact-recall probe added to the standing eval set?** It is
   the only one of the three regression checks that does not exist yet.
