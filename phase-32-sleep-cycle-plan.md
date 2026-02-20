# Phase 32: Sleep Cycle — Three-Tier Rest Architecture

## Add to PLAN.md under Phase 32

### Phase 32: Sleep Cycle

Three tiers of rest, inspired by biological sleep patterns. Monica uses
idle time productively without being asked — effortless readiness (Wu Wei).

#### Tier 1: Deep Sleep (daily, 3:00 AM cron)
Full memory consolidation cycle. Heavy compute, 5-10 minutes.

- [ ] `mycoswarm sleep` CLI command (also triggered by systemd timer)
- [ ] Review: scan all sessions since last sleep
- [ ] Extract: pull lessons, facts, emotional tones from sessions
- [ ] Validate: check new facts against existing memory, flag contradictions
- [ ] Consolidate: merge validated lessons into long-term memory, age stale facts
- [ ] Dream: R1 (P320) makes novel cross-session connections
- [ ] Log: write sleep report to ~/.config/mycoswarm/sleep-logs/
- [ ] `/sleep` chat command: view last sleep report
- [ ] `mycoswarm sleep --last` CLI: view last sleep report
- [ ] systemd timer: `mycoswarm-sleep.timer` runs at 3:00 AM daily
- [ ] Monica can reference dreams: "Last night I noticed..."

#### Tier 2: Nap (idle, after N minutes of no chat)
Quick housekeeping. Lightweight, 10-30 seconds.

- [ ] Idle detection in daemon (extend existing timing gate)
- [ ] Configurable idle threshold (default: 15 min)
- [ ] Refresh stale fact scores
- [ ] Pre-fetch: anticipate next topic from recent conversation context
- [ ] Rehearsal: re-read today's sessions to strengthen recall
- [ ] Vitals check: peer health, disk space, daemon status
- [ ] Tidy: check for unfinished drafts, orphaned temp files
- [ ] Nap log: brief entry in sleep-logs (separate from deep sleep)

#### Tier 3: Daydream (micro-idle, >60s gap between messages)
Background thought during active conversations. Near-zero overhead.

- [ ] "What might they ask next?" — pre-warm relevant RAG context
- [ ] Update running session summary
- [ ] Pre-load related documents if topic is shifting
- [ ] Background only — never interrupts, never visible unless asked

#### Design Principles
- Wu Wei: rest is productive, not idle. Monica prepares without being asked.
- Biological rhythm: deep sleep is scheduled, naps are opportunistic,
  daydreams are continuous background processing.
- R1 on P320: deep sleep dream phase is the primary use case for the
  dedicated reasoning worker. Novel connections, not just summarization.
- Observable: Guardian can always see what happened during sleep via
  `/sleep` command or sleep logs.
- Non-disruptive: naps and daydreams never interrupt active work or chat.

#### Dependencies
- Phase 32 Tier 1 (deep sleep): episodic memory (done), fact lifecycle (done)
- Phase 32 Tier 2 (nap): timing gate (done), daemon idle tracking (extend)
- Phase 32 Tier 3 (daydream): RAG pre-warming (new), session summaries (extend)
- Dream phase: P320 online with R1:14b (Wednesday target)

#### Implementation Order
1. Deep sleep CLI + cron (no R1, just consolidation) — can build Monday
2. Nap idle trigger + housekeeping tasks — Tuesday
3. Deep sleep dream phase with R1 — Wednesday after P320 is online
4. Daydream micro-idle — later, needs RAG pre-warming infrastructure
