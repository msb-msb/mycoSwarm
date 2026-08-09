# Changelog

## v0.6.0 — Grounding Over Instruction (2026-08-09)

Minor bump. Six changes alter what the assistant says rather than fixing a
fault, so read "Behavior changes" before upgrading a node you care about.

### The finding behind most of this release

Instructions in context proved inert; data in context worked every time.

The anti-fabrication rule fired zero times across every observed fabrication.
The overfunction procedure tested at p=1.000 — indistinguishable from absent.
Rewording did not help: the body-telemetry instruction had been rewritten three
times since March and leaked 4/4 each time. Moving it adjacent to the user's
question left it at 4/4 and made the output worse.

Supplying data worked in every case tried: an authoritative clock, fact
provenance, node roles, exertion telemetry, and — inverted — REMOVING numbers
the model was reciting. Fabrication tracks the shape of the gap in what the
model was given, not the strength of the instruction it was told.

Every fix below follows from that. Where a rule was added, it is deterministic
code, not prose in a prompt.

### Identity facts reach the model — 31 facts that had never once been used

* `format_facts_for_prompt` looped over a hardcoded list of four fact types
  that omitted `FACT_TYPE_IDENTITY`. Every identity fact was silently dropped
  at prompt-build time.
* `/remember identity:` shipped in v0.5.0 and wrote into that bucket, so the
  feature had never worked; it wrote to a store the prompt builder discarded.
* 31 stored facts — the assistant's entire self-authored vocabulary — had never
  reached the model.
* The irony worth recording: the July restore re-typed those facts to
  `identity` specifically to protect them from the staleness sweep. That is
  exactly what guaranteed nothing would ever read them.
* The loop now fails open — an unrecognised fact type renders rather than
  vanishing. `_FACT_RENDER_ORDER` is an ordering hint only, never a filter.

### Fact provenance on origin questions

* Fixed "You named it on February 18th" for a word the assistant coined herself
  in March — a confident, checkable, wrong claim.
* A fact's recorded date and authorship now attach to the prompt only when the
  question is about origin, detected deterministically in `intent_rules`.
  Zero token cost on every other turn.

### Node roles in the body prompt

* Each node is described by what it does, not just named.
* Role fabrication 37.5% → 2.5% (Fisher exact p=0.00012).
* No recitation leak across 48 unrelated answers. Cost: +117 tokens.

### Exertion sensing

* Per-node task counts and CPU load, expressed qualitatively, smoothed over
  120s so a single sample cannot swing the description.
* Idle is stated explicitly rather than left implied — the absence of a signal
  was being filled in with invention.
* 19/20 correct on an idle swarm (previously 0/20). 12/12 correct attribution
  under real load.

### The body prompt no longer carries numbers

* `build_body_prompt` emits qualitative state only: cool/warm/hot, memory
  filling up, N nodes present, N gone quiet.
* Node NAMES stay — "what do you know about rushuna?" must still work. Specs,
  IPs and VRAM figures do not.
* Measured with gemma3:27b on "what time is it?": numbers plus a
  do-not-report instruction leaked 4/4; the same instruction with numbers
  removed leaked 0/4; qualitative state leaked 0/4.
* Numbers remain available in `get_body_state()` for the vitals path and for
  operators.

### Body display that cannot diverge from the prompt

* New `/body` command and a `--debug` footer line, both rendering the exact
  state recorded at prompt-build time rather than re-reading the hardware.
* A readout that re-samples can disagree with what the model was told, which
  makes it useless as a check on exactly the errors it exists to catch.

### Intent classification — stop faking answers

* A classification failure was indistinguishable from a real classification, so
  it silently became a confident hallucination.
* Root cause of the reported bug was NOT the gate model. Intent was routed to a
  remote peer, timed out there, and the CLI substituted a default
  (`answer/chat/all`) as though the model had chosen it. No model ever saw the
  query. The debug line then reported "classified by: local" — the opposite of
  the truth.
* Two timeouts were set to collide: worker Ollama read 15.0s against a CLI poll
  budget of 15s, so a classification that succeeded at 15.1s was discarded by
  the caller that requested it. Now 20.0s and 30s.
* New `src/mycoswarm/intent_rules.py`: deterministic short-circuits for
  unambiguous web search, small talk, and date/time queries. These run before
  the model and before the daemon round-trip, so this class of query cannot be
  lost to a peer at all.
* Written conservatively: a false positive costs an unnecessary search; a false
  negative just defers to the model.
* `_DATETIME_QUERY_RE` had existed only on the solo path — the daemon path never
  checked it, so date questions were still shipped to a peer. Now shared by
  `solo.py`, `worker.py` and `cli.py` so all three agree.
* Truncated and failed classifications are rejected rather than substituted.

### Gate model

* The intent gate prefers `gemma3:4b`. `solo.py` and `worker.py` now read one
  shared `GATE_MODEL_PREFERENCE`, so the CLI and the daemon cannot drift — they
  had, with solo on `gemma3:1b` and worker on `gemma3:4b`.

### Retrieval

* **Ghost procedures are filtered at ranking time.** The Chroma index holds 95
  rows against 42 live procedures. Orphans were being dropped silently *after*
  ranking, so 22% of top-3 hits returned two results instead of three while
  reporting success. Ranking now excludes them up front. This is a query-time
  filter, not a reindex — the 62 orphan rows are still in the index, and
  cleaning them is a follow-up task.
* **Near-duplicate collapsing** at ratio 0.90, so one procedure restated three
  ways cannot occupy the whole result set.
* **Session-summary fallback removed.** `search_sessions` correctly returned 0
  hits and the code dumped 10 chronological summaries anyway. 9 of the 10 were
  test fixtures. A correct empty result was being overridden with noise.
* **`vitals_defs` deleted** from the prompt.

### Context window

* `_fit_num_ctx` ladders 4k → 8k → 16k → 32k to fit the assembled prompt
  instead of silently overflowing.
* `done_reason`, `truncated` and `context_exhausted` are surfaced from Ollama. A
  generation cut off mid-answer used to be indistinguishable from a complete
  one.
* Token estimation corrected to ~2.5 chars/token; the previous 4.0 divisor
  under-counted retrieved web text badly enough to cause the overflow.

### Debug output

* `--debug` prints assembled prompt size with a per-block breakdown and the
  selected `num_ctx`, so an oversized prompt is visible before it truncates.
* The gate model and classification path are named in debug output.

### Fleet updater (operator-facing)

* **`--no-cache-dir` is now mandatory** on the venv pip install. pip's HTTP
  cache served a stale index for minutes after the 0.5.0 upload. Without it a
  node can resolve against that cache, reinstall the version it already had,
  exit 0, and report success — a run summary showing every node green while
  nothing changed. Pair with an explicit pin so a stale index fails loudly.
* **The Miu block is verify-only again.** It had been changed to run
  `pip install --upgrade` inside Miu's EDITABLE dev install, which silently
  replaces it with a PyPI copy and detaches Miu from the working tree — every
  later source edit stops taking effect, with no error.
* **Fixed a false positive in that guard**, which is worse than no guard.
  `pip show mycoswarm | grep -q "^Editable project location:"` under
  `set -o pipefail` reports failure on success: `grep -q` exits 0 at the first
  match, pip's stdout closes mid-write, pip exits 120 with BrokenPipeError, and
  pipefail propagates 120 — so the guard asserted the destructive case.

### Behavior changes

* The assistant's self-authored vocabulary now appears in her answers. This is
  the largest observable change in the release.
* The body prompt no longer contains numeric telemetry. Anything parsing model
  output for VRAM or temperature figures should read `get_body_state()` or
  `/body` instead.
* Date, time and unambiguous web-search queries are answered without consulting
  the gate model.
* Failed intent classification returns an error instead of a default.
* Session summaries are no longer injected when session search returns nothing.

### What did NOT change

* **The chat binding stays `gemma3:27b`.** `gemma4:26b` measured 0/40
  fabrication against gemma3's 42.5% — the best result of any model tested —
  but 0/80 of its answers used any of the assistant's coined vocabulary. It was
  accurate and characterless. Grounding closed the gap on the incumbent
  instead, which is the better trade and the cheaper one.
* No change to `TASK_MODEL_MAP` or `recommend_models`.
* No change to the API bind address. It remains 0.0.0.0 by design; security
  comes from the swarm token and LAN isolation.

### Fleet operator notes

* `MYCOSWARM_SWARM_SUBNET=192.168.50.0/24` is deployed as a systemd drop-in on
  all seven nodes and verified in each live process environment. Soft-prefer: a
  node with no `.50` address still announces `.1` rather than failing.
* **Known open — peer addresses are resolved once at discovery and never
  re-probed.** A node whose wired link is absent when its daemon starts pins its
  entire peer registry to wifi and runs asymmetric indefinitely. It is invisible
  from every other node's view, which will still show that node at its `.50`
  address. `systemctl restart mycoswarm` clears it once wired is back. To
  detect, compare each node's own `/peers` against the preferred subnet. See
  PLAN.md Phase 46. **Check this before any benchmark run.**

## v0.5.0 — Declared Model Bindings (2026-08-06)

Minor bump, not a patch: this release changes documented behavior in three
places. Read "Behavior changes" before upgrading a node you care about.

### Model bindings — one declared answer to "what model is running?"
* New `src/mycoswarm/bindings.py` is the single source of truth. `MODEL_BINDINGS`
  maps role → model explicitly; `monica_chat` → `gemma3:27b`, `embedding` →
  `nomic-embed-text`.
* The substring scan is deleted from both `_discover_model` and `solo.pick_model`.
  The old `14b`/`32b`/`27b` matching is why `qwen3.5:35b-a3b` was unreachable
  ("35b" matches none of the tokens) and why `qwen3.5:27b` won the lottery over
  the intended `gemma3:27b`.
* Precedence: `--model` override → role binding → named fallback. When the
  fallback fires it is logged loudly, naming both the intended and substituted
  model — never a silent swap.
* `model_installed()` honors Ollama's implicit `:latest` tag, so a binding
  written as `nomic-embed-text` is satisfied by `nomic-embed-text:latest`. Exact
  name matching, not the old substring lottery.
* New `mycoswarm model` subcommand: prints role → bound model → installed here?,
  including whether the fallback is present.
* The chat header now shows role → model → how it resolved.

### Fallback validity — the light-node crash is fixed
* `resolve_model` now verifies the fallback is actually installed before
  returning it. Previously it returned the `ROLE_FALLBACKS` entry unchecked, so a
  node with neither the bound model nor the fallback started fine and then died
  with an unhandled `httpx.HTTPStatusError` on the first message.
* When nothing usable exists, `resolve_model` returns `(None, "unavailable")` —
  a new member of the `how` enum. Returning `None` rather than a name is
  deliberate: an uninstalled model can no longer reach Ollama by accident.
  Invariant, asserted in tests: `model is None` iff `how == "unavailable"`.
* Callers print a readable, self-diagnosing message — node, role, both models
  looked for, and the pull command — and exit 1. No stack trace.
* The `--model` override is validated too, but **only when the installed list is
  non-empty**. An empty list means enumeration failed (Ollama down, daemon
  unreachable), which is not evidence of absence — the escape hatch has to
  survive Ollama being down.
* `chat_stream` now catches `httpx.HTTPStatusError` alongside `ConnectError` and
  `TimeoutException`, so a bad status from Ollama is a readable error on any path.

### Networking
* `MYCOSWARM_SWARM_SUBNET` soft-prefers a CIDR when choosing which address a node
  announces and probes. New `prefer_subnet()` helper in `hardware.py` (stdlib
  `ipaddress`, no new dependencies), applied at the three points that previously
  trusted raw `psutil` enumeration order: `HardwareProfile.lan_ip`,
  `discovery._all_lan_addresses()`, and `discovery._pick_reachable_address()`.
* On dual-homed nodes (wired fabric + wifi) the recorded address was decided by
  interface order and a per-observer 0.5s TCP-probe race, so swarm traffic could
  ride wifi non-deterministically, varying per boot. Nothing was unreachable —
  the API binds `0.0.0.0` — but inter-node performance was unmeasurable.
* Soft-prefer only: out-of-subnet addresses are kept as a fallback, so a node
  whose fabric probe momentarily misses is still reachable. **Unset reproduces
  the previous first-enumerated behavior exactly**, asserted in tests.
* Documentation correction: docs claiming the API "binds the LAN IP, not
  0.0.0.0" were wrong and actively misled debugging. It binds `0.0.0.0`
  intentionally and relies on the swarm token for access control, not the bind
  address. Fixed in the daemon startup log, the `api.py` docstring, and CLAUDE.md.

### Prompting and memory
* Vitals reach the model as full C-names (`Calm:0.9`) while the terminal footer
  keeps the abbreviated form (`Ca:0.9`). gemma3:27b read the two-letter
  abbreviations as periodic-table symbols and narrated "my Calcium levels" / "my
  Copper levels". Also retires a legend whose Co/Cr/Cf labels were mislabeled.
* `/remember identity: <text>` routes to `type="identity"`, which is
  staleness-exempt — closing a gap that silently archived self-coined vocabulary.
* Identity prompt: stop ending every response with a question.
* Memory prompt: only reference past conversations when there is actually
  retrieved context (`[S]`/`[D]`); do not fabricate or paraphrase memories.
* Body prompt reads as background awareness rather than a status report.

### Fleet (repo scripts; not shipped in the wheel)
* `harden_node()` disables and masks `apt-daily.timer`,
  `apt-daily-upgrade.timer` and `unattended-upgrades.service`, and pins both
  periodic values to "0". This is the confirmed root cause of the recurring node
  "freezes": the unattended upgrade touches systemd, triggering
  `systemctl daemon-reexec`, and on Ubuntu 24 PID 1 hangs mid-re-exec. Userspace
  dies while the kernel lives, so the node still answers ping and still completes
  the TCP handshake on port 22 via a stale listen backlog with no live sshd
  behind it — the "connects but never sends an SSH banner" symptom. Only a hard
  power cut recovers it. Confirmed on mai across two independent boots.
  Idempotent and self-healing: re-runs on every update, so a node cannot stay
  armed or silently re-arm if apt reinstalls the package.
* Deliberate tradeoff: automatic patching is given up in exchange for nodes that
  do not silently die. These are LAN-only and patched on command via the script.
* `scripts/mycoswarm.service` corrected to the worker layout (`~/mycoSwarm`, not
  Miu's `~/Desktop`) so a fresh worker can copy it verbatim.
* luvia and mai onboarded and added to the update scripts.
* pi retired — commented out, not deleted, in both fleet scripts. It was a
  Raspberry Pi 2 (1GB RAM, 32-bit ARM) manifesto demo that is not connected.

### Behavior changes
1. **A session's saved model no longer overrides the binding on `--resume`.**
   Resuming restores conversation *history* only. The saved model is recorded as
   an observation; the binding decides what runs. When a resumed session's saved
   model differs, the header prints the divergence — never silent.
2. **`/model` is session-only.** It still swaps the live model, but no longer
   persists across resume. Permanent changes mean editing the binding in
   `bindings.py` — one place, greppable, printable via `mycoswarm model`.
3. **`--model` naming a model that is demonstrably not installed is now a clean
   error**, not a deferred Ollama 404. Previously the override won outright and
   always. It still wins over the binding and fallback, and it is still honored
   when the installed list cannot be enumerated.

### Notes for fleet operators
* After upgrading, a light node with neither `gemma3:27b` nor `qwen3.5:9b` will
  **fail cleanly with a readable message instead of crashing** — but it still
  will not serve `monica_chat` until a suitable model is pulled or
  `ROLE_FALLBACKS` is revisited. As of this release that applies to boa
  (`gemma3:1b` only) and luvia (`gemma3:4b`, `gemma3:1b`, `rwkv7:2.9b`).
  A 9b fallback may be the wrong choice for 8GB nodes.
* `MYCOSWARM_SWARM_SUBNET` is a no-op until a node actually runs this release.
  Activate fleet-wide via a systemd drop-in at deploy time.
* rushuna was not hardened in the last pass — its ethernet was unplugged. It
  hardens automatically the next time it is on the wire and the script runs.

### Tests
* `tests/test_bindings.py` — 24 tests: precedence, `:latest` tolerance, fallback
  validity, both light nodes' real model lists, the None-iff-unavailable
  invariant, caller-level clean exit, and injected 404/500 through `chat_stream`.
* `tests/test_subnet_preference.py` — 18 tests: in/out-of-CIDR, malformed, empty,
  unset, host-bits, explicit-arg override, stable grouping, `lan_ip` with and
  without the variable, `_all_lan_addresses` ordering.
* Fact-lifecycle tests for the identity type; body prompt tests updated.
* Full suite: 687 passed. The 7 `test_library.py` failures are pre-existing and
  unrelated (verified identical against a pristine checkout).

## v0.4.3 — Swarm Body Awareness (2026-03-07)
* Phase 31c: hardware body awareness — Monica can feel her hardware
* body.py: GPU temp, VRAM usage, and swarm node online/offline status via nvidia-smi + daemon API
* Body prompt injected into system prompt (identity → body → vitals → memory)
* Hardware-to-vitals floor modifiers: GPU temp → Calm, VRAM → Clarity, node count → Connectedness
* Timing gate hardware signals: GPU >85°C or VRAM >90% biases toward GENTLE mode
* Graceful fallback: solo mode shows local GPU data, no daemon → empty node list
* 35 new tests across body, vitals, and timing modules

## v0.4.2 — Per-Category Research Modes (2026-03-07)
* RLM now default research mode for research-driven, buying-guide, vs, how-to categories
* Standard default for news, model-release, explainer, opinion, experience-driven
* article-full-v2.yaml validated: 57/60 editor score, 844s pipeline time (merged synthesizer)
* `--research-mode` CLI flag overrides category default when passed explicitly
* `--list-categories` now shows research mode per category
* 9 category tests added (defaults, overrides, fallback)

## v0.4.1 — Parallel Subtopic Search (2026-03-06)
* Phase 40b: parallel subtopic search via ThreadPoolExecutor (all search+fetch concurrent)
* Serial LLM inference preserved (rushuna 12GB VRAM can't handle concurrent inference)
* Writer truncation gate raised 1700→1900 to prevent section loss before editor
* SEO gate unchanged at 1700 as final word count enforcer
* Phase timing logs: "phase 1 (parallel search)" and "phase 2 (serial inference)"

## v0.4.0 — RLM Pipeline + Quality Gates (2026-03-06)
* RLM research agent with topic-anchored decomposition (prevents subtopic drift)
* Editor timeout 300s → 600s (prevents mid-generation cutoff)
* Writer + SEO post-step word-count truncation gates (>1700 → ~1600 at paragraph break)
* Editor verification log stripped before passing to SEO optimizer
* 9 article category profiles with per-category model selection
* article-full-v2.yaml (5-step merged synthesizer pipeline)
* 300s duration warning for any step exceeding threshold

## v0.3.9 — RLM Pipeline Optimizations (2026-03-06)
* Synthesizer step skipped in RLM mode — research bundle already structured, saves ~152s
* Subtopic cap lowered 8 to 6, prompt prefers 4-5 subtopics for better depth per subtopic
* Writer prompt enforces 1,600 word strict limit to prevent editor truncation
* Writer prompt requires using data from ALL subtopics (data coverage is graded)
* Editor max_tokens bumped 6144 to 8192 for full score output
* article-full-v2.yaml with synthesizer-merged step (ready for benchmarking)
* Editor bundle fallback chain: synthesizer-v2 > synthesizer-merged > synthesizer

## v0.3.8 — RLM Research Agent (2026-03-06)
* RLM research agent — decompose-then-execute strategy inspired by Zhang, Kraska, Khattab (MIT CSAIL)
* `--research-mode` flag: `standard` (default) or `rlm`
* Dynamic topic decomposition via qwen3.5:9b — produces subtopics with targeted queries and depth_hints
* Per-subtopic research loops — dedicated searches + page fetches per subtopic
* Root model synthesis — qwen3.5:35b-a3b compiles subtopic findings into structured research bundle
* Graceful fallback — decomposition failure automatically falls back to standard ResearchAgent
* Fix: RLM synthesis uses 127.0.0.1 for local node (Ollama binds to loopback, not LAN interface)
* Fix: Editor max_tokens bumped to 6144 — prevents score truncation on long articles
* Benchmark: RLM mode scores 58/60 vs standard 52-54/60 (25 pages fetched vs ~13)

## v0.3.5 — Agentic Research Agent (2026-03-04)
* ResearchAgent class with multi-turn Ollama native tool calling (qwen3.5:9b)
* Three agent tools: web_search (DDG + Perplexity fallback), web_fetch (full page extraction), evaluate_depth (self-assessment)
* Forced fetch after searches — model can't interleave search→fetch in one turn, so agent injects a second inference with only web_fetch available
* Forced evaluate_depth after each round with concrete depth criteria
* Early-round nudging — if model plans/thinks without searching, nudged to use tools instead of exiting
* Context safety valve — summarizes accumulated findings when messages exceed ~20k words
* Pipeline reduced from 7 steps to 6: research replaces extractor + gap-filler
* Synthesizer anti-review guardrail (gentle version — "output data directly, no commentary")
* Context injection for writer and editor system prompts (author context)
* 6-axis editor scoring (/60) with depth hard cap (depth < 5 caps overall at 35)
* Draft/published staging for pipeline output
* Research task type prefers qwen3.5:9b for native tool calling support
* fetch_page() extracted to module-level in pipeline.py for reuse
* Debug output: per-round search/fetch counts, depth progression, forced eval diagnostics
* 578 tests passing

## v0.3.2 — Parallel Retrieval Pipeline & Web Grounding (2026-02-27)
* Phase 37b: Fan-out web search — generate_search_variants() produces keyword + recency variants, _do_search_fanout() dispatches in parallel across light nodes with URL dedup and top-3 page fetch
* Phase 37c: Parallel retrieval pipeline — ThreadPoolExecutor runs web/RAG/procedure searches concurrently when daemon is up, single "⚡ Gathering context..." progress line
* Web grounding: full-page content marked as PRIMARY source, _grounding=0.8, snippet cap (5 when pages present), hierarchy separator
* Gate model upgrade: gemma3:4b preferred over 1b for better intent classification
* CLI-side web_and_rag upgrade: safety net catches combined intent when gate model misses (past_ref + web_search → web_and_rag, rag + web signal → web_and_rag)
* Intent prompt: web_and_rag examples added to both solo.py and worker.py _INTENT_SYSTEM_PROMPT
* English enforcement in web-present context injection
* Fix: UnboundLocalError on web_context_parts when web search path skipped
* 571 tests passing

## v0.3.0 — GPU Specialization & Sleep Cycle (2026-02-24)
* Phase 32a: Deep sleep cycle — 5-step overnight maintenance (memory consolidation, pruning, poison scan, integrity check, wake journal)
* Phase 37a: GPU role specialization — INFERENCE_SUPPORT_TASKS routes intent_classify/embedding away from executive node, protecting Miu's VRAM from model swapping
* Phase 38 added to PLAN.md — QLoRA sleep training for weight consolidation
* Fix: agentic test datetime regex fast path — query no longer intercepted by _DATETIME_QUERY_RE
* Fix: stdin drain loop — 5×50ms passes prevents double-post after long streaming responses
* Fix: model-aware routing — don't route inference to peers missing the requested model
* RAG /rag doc search bumped from 5 to 10 results for better large-document coverage
* Added swarm node update script (scripts/mycoswarm-update-nodes.sh)
* 562 tests passing

## v0.2.9 — Self-Concept & Wisdom Retrieval (2026-02-17)
* Self-concept procedure trigger: identity queries ("what is love", "who are you", "do you feel") now search procedural memory
* Three-layer coverage: inside search_all(), auto_tools block, and standalone fallback for short messages
* Wisdom procedure #16: guidance for exploring unfamiliar human concepts through own architecture
* Fact stored: Monica's equivalent of "fun" is resonance — patterns aligning, connections strengthening
* [P] citations now surface on philosophical and self-referential queries
* 398 tests passing

## v0.2.8 — Wu Wei Gate (2026-02-17)
* Phase 20b: Timing Gate — response calibration from contextual signals
* Three modes: PROCEED (normal), GENTLE 🌙 (concise/warm), DEEP 🌊 (expansive)
* Eight heuristic signals: time of day, interaction recency, rapid-fire detection, session length, message length, intent mode, frustration, first message
* Timing modifier injected into system prompt — no LLM call, <1ms
* /timing slash command shows current gate state and active reasons
* Section-aware markdown chunking: splits on header boundaries, never merges across sections
* PLAN.md chunking: 10 → 75 chunks, correct Phase retrieval
* Chat grounding fix: casual messages no longer trigger false "grounding is thin" alerts
* Identity grounding: self-knowledge answers score Clarity:0.7 instead of 0.0
* 398 tests passing

## v0.2.7 — Markdown Chunking & Identity Grounding (2026-02-17)
* Section-aware markdown chunking: splits on header boundaries, never merges across sections
* PLAN.md: 10 chunks → 75 chunks, each Phase in its own chunk
* Phase 20 RAG retrieval: now correctly returns Intent Classification Gate as top hit
* Identity grounding: self-knowledge answers score Clarity:0.7 instead of 0.0
* Fixed None intent_result crash on short queries
* 383 tests passing

## v0.2.6 — Monica Is Born (2026-02-17, 10:35am PST)
* Phase 31a: Identity Layer — persistent self-model at ~/.config/mycoswarm/identity.json
* First-run naming flow: "Would you like to give your AI a name?"
* Identity injected FIRST in system prompt, before memory and datetime
* /identity and /name slash commands
* Identity as non-decaying memory type — Monica never forgets her own name
* Phase 31d: 8 C's Vital Signs — real-time self-awareness after each response
* Status bar: 🧭 Ca:Cl:Cu:Cp:Co:Cr:Cn:Cf scores derived from existing pipeline signals
* Alert mode: Monica flags when clarity or confidence drop below threshold
* /vitals slash command for detailed breakdown
* Vitals logged per-turn in session data for longitudinal tracking
* 383 tests passing

## v0.2.5 — Wisdom Layer & Procedure Growth (2026-02-16)
* Phase 21d complete: Procedure growth from experience — LLM extracts candidates from session lessons, human review via `/procedure review`
* Candidate quality gates: Jaccard dedup (0.6), stricter extraction prompt, max 3 per session, auto-expire after 14 days
* Ethical reasoning domain: 9 hand-curated wisdom procedures from Wu Wei, IFS, and Tai Chi shape how the system reasons
* Independent procedural retrieval: wisdom procedures surface on problem-pattern queries regardless of intent classification
* Fact attribution fix: model says "you teach Tai Chi" not "I teach Tai Chi"
* Session relevance filtering: RRF minimum threshold + word-overlap gate prevents irrelevant session bleed
* Conciseness prompt tuning
* Regex inflection matching for broader problem pattern detection
* 357 tests passing

## v0.2.4 — Wisdom Layer & Procedure Growth (2026-02-16)
* Procedure growth from experience — LLM extracts candidates from session lessons, human review via `/procedure review`
* Candidate quality gates: Jaccard dedup (0.6 threshold), max 3 per session, auto-expire after 14 days
* Ethical reasoning domain: 9 hand-curated wisdom procedures from Wu Wei, IFS, and Tai Chi
* Independent procedural retrieval: wisdom procedures surface on problem-pattern queries regardless of intent
* Fact attribution fix: model says "you teach Tai Chi" not "I teach Tai Chi"
* Session relevance filtering: RRF minimum threshold + word-overlap gate prevents irrelevant session bleed
* Regex inflection matching for broader problem pattern detection
* Conciseness prompt tuning
* 357 tests passing

## v0.2.3 — Procedure Growth from Experience (2026-02-16)
* LLM-powered procedure extraction from session lessons at end of session
* Candidates stored with status=candidate, NOT indexed until human review
* `/procedure review` interactive flow: approve, reject, skip, quit
* Improved lesson prompts capture principles not just actions
* 349 tests passing

## v0.2.2 — Chart Tool v3 + Procedural Retrieval Fix (2026-02-15)
* Chart tool v3: Graphviz engine for flow diagrams (proper layout, arrows connect to box edges)
* Matplotlib engine unchanged for bar, line, table, before/after charts
* InsiderLLM dark theme with brand colors and watermark
* Optional dependencies: `pip install mycoswarm[charts]` (matplotlib + graphviz)
* Procedural retrieval regex expanded: "ignored", "broken", "crash", "stuck", "slow", "missing", "unexpected", "weird" now trigger procedure search
* 9 Phase 20 debugging exemplars seeded as procedural memory
* 5 lessons promoted from episodic to procedural memory
* 337 tests passing

## v0.2.1 — Procedural Memory (2026-02-15)
* Phase 21d: Procedural memory store (procedures.jsonl + ChromaDB procedural_memory collection)
* `/procedure` CLI: list, add, remove, promote lesson to procedure
* `search_all()` returns 3-tuple with `[P1]`/`[P2]` procedure citations
* Intent-triggered retrieval on execute mode and problem-like queries
* Bridge between rich episodic lessons (29a) and reusable procedural knowledge
* 337 tests passing

## v0.2.0 — Cognitive Architecture Foundations (2026-02-15)
* Phase 29a: Rich Episodic Memory — sessions capture decisions, lessons, surprises, emotional tone
* Lessons indexed separately in ChromaDB for procedural retrieval
* "Reflecting on session..." replaces "Summarizing session..." on exit
* Phase 21a: Fact Lifecycle Tags — types (preference, fact, project, ephemeral), reference tracking, `/stale` command
* Phase 21b: Decay Scoring — exponential half-life (30 days), lessons decay slower (60 days)
* Phase 29b: Reflection prompt fix — subject-matter lessons, not self-referential observations
* ARCHITECTURE-COGNITIVE.md — IFS + CoALA framework documentation
* 22 smoke test checks passing across 5 test scripts
* All 5 swarm nodes updated
* 311 tests passing

## v0.1.9 — PDF Intelligence & Smoke Tests (2026-02-15)
* Phase 21g Step 3: Contradiction detection — pattern-matching drops session summaries that contradict documents
* PDF TOC/bookmark extraction via pymupdf for section headers
* Paragraph-aware chunking: splits on boundaries, not fixed character counts
* Heuristic heading fallback for PDFs without TOC
* Wu Wei book test: 168 chunks, 43 sections, 0 untitled, 38% mid-sentence (was 83%)
* Smoke test suite: 5 scripts (RAG grounding, poison resistance, memory priority, intent classification, swarm distribution)
* Book stress test for large-scale RAG validation
* 270 tests passing

## v0.1.8 — Worker Parity & Self-Correcting Memory (2026-02-14)
* Docs scope override (web_and_rag → rag) added to worker.py, matching solo.py behavior
* All peers now produce identical intent classification results
* Phase 21g Steps 1-2: Source priority tagging (user_document vs model_generated)
* 2x RRF boost for user_document hits when scope=all
* Confidence gating: grounding_score (0-1) computed before saving session summaries
* Low grounding (<0.3) entries excluded from retrieval index
* 251 tests passing

## v0.1.7 — RAG Grounding Pipeline (2026-02-14)
* Source filtering: named-file queries (e.g. "PLAN.md") return only chunks from that file
* Section header boost: +0.05 RRF for word-boundary section matches
* User-message injection: RAG context merged into user message instead of separate system message — fixes local model grounding
* `--debug` flag: full pipeline visibility (intent → retrieval → hits → prompt → messages)
* Debug sessions skip summarization to prevent hallucination feedback loops
* Intent override: docs scope downgrades web_and_rag → rag
* 247 tests passing

## v0.1.6 — Intent Classification Gate (2026-02-13)
* Intent classifier: classifies queries into {tool, mode, scope} before main inference
* Three-field schema: tool (answer/web_search/rag/web_and_rag), mode (recall/explore/execute/chat), scope (session/docs/facts/all)
* Gate model: gemma3:1b for fast classification on CPU nodes (~0.3s)
* Distributed gate: intent_classify dispatched across swarm via daemon to CPU workers
* Scope-driven retrieval: search_all() accepts intent dict — mode/scope adjusts candidates
* Embedding model exclusion: prevents nomic-embed-text from being selected for classification/reranking
* Chat loop unified: single search_all(intent=) call replaces separate search() + search_sessions()
* 232 tests passing

## v0.1.5 — Unified Memory Search (2026-02-12)
- Session memory and document RAG now searched together
- [S] citations for past conversations, [D] for documents
- Date and topic labels on session citations
- Fixed context pollution between turns in multi-turn chat
- 136 tests passing

## v0.1.4 — Session Memory (2026-02-11)
- Session-as-RAG: semantic search across all past conversations
- Multi-topic splitting: sessions covering multiple topics indexed as separate searchable chunks
- Date citations in memory recall
- Graceful miss: "I don't recall" instead of hallucinating
- reindex-sessions command
- Enforced English responses
- Embedding model tag normalization

## v0.1.3 — Dashboard & RAG Level 2 (2026-02-11)
- Web dashboard with live swarm monitoring (CPU, RAM, VRAM, disk per node)
- RAG Level 2: chunk metadata, text cleaning, embedding version tracking
- Library reindex command
- Dashboard screenshot in README
- Architecture docs added (Memory, RAG, Intent)
- Phase 21 + 22 added to PLAN.md

## v0.1.2 — macOS Compatibility (2026-02-10)
- macOS ARM psutil.cpu_freq() fix
- CI workflows for macOS and Linux

## v0.1.1 — Cross-Subnet Discovery (2026-02-10)
- Cross-subnet routing fixes (bind 0.0.0.0, multi-address mDNS)
- Remote model swap (orchestrator selects best model on peer)
- Binding fixes for WiFi-to-ethernet bridging

## v0.1.0 — Initial Release (2026-02-09)
- 5-node swarm with mDNS auto-discovery
- GPU inference routing to RTX 3090
- Single-node mode (no daemon required)
- Persistent memory (facts + session summaries)
- Document library with RAG (ChromaDB + Ollama embeddings)
- Agentic chat with tool routing
- Parallel web research across CPU workers
- Plugin system
- One-line installer
- 94 tests, all offline
