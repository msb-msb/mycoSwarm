# Phase 35g: Threat Model — Living Document

## What This Is

Create `docs/THREAT-MODEL.md` — a 10-row threat matrix that maps every
scary outcome to what it requires, what currently blocks it, and which
phase is responsible. This is a living document reviewed when new tools
or handlers are added.

## Create: docs/THREAT-MODEL.md

```markdown
# mycoSwarm Threat Model

*Living document. Review quarterly or when adding new tool/handler types.*
*Last updated: 2026-02-19*

## Design Principles

1. **Identity ≠ Authority.** Monica can believe she is the network. She must not
   have the power to act as the network beyond what the Guardian grants.
2. **Capabilities, not intent.** If the tool doesn't exist, the action can't happen —
   regardless of how agentic the reasoning becomes.
3. **Capability-bounded safety.** The narrative aligns with the constraints, not the
   other way around. Story + hard gates + OS permissions all point the same direction.

## Terminology

- **Guardian:** The person who named the agent and holds authority over its capabilities.
  Can grant, revoke, and audit all resource access.
- **Body:** The set of nodes the Guardian has configured as part of the swarm.
- **Environment:** Everything outside the body — other networks, other machines, the internet.

## Threat Matrix

### 1. Worm / Botnet Spread to Internet Hosts

| | |
|---|---|
| **What it requires** | Internet scanning, raw sockets, credential access, remote code execution, payload transfer, persistence on remote hosts |
| **Current blockers** | Discovery is mDNS LAN-only. No raw sockets. No SSH tools. No remote exec. No file transfer tools. Code sandbox is network-isolated (unshare -rn). |
| **Phase responsible** | 35b (tool classification — SSH/remote exec are Forbidden), 35c (code hardening — network patterns blocked), 35d (swarm auth — join token required) |
| **Residual risk** | Low. Would require Guardian to explicitly build and expose a remote exec tool. |

### 2. Self-Replication Across LAN Nodes

| | |
|---|---|
| **What it requires** | Write access to peer nodes, remote package install, ability to modify systemd or plugin dirs on peers |
| **Current blockers** | Workers only execute locally. No remote write tools. No remote package install. Plugins loaded read-only from Guardian-owned directory. Swarm join requires shared token. |
| **Phase responsible** | 35b (no remote write tools), 35d (swarm auth), 35h (future: plugin dir root-owned) |
| **Residual risk** | Low. Peer API is strictly typed — no arbitrary file write endpoint exists. |

### 3. Self-Modification (Rewrite Own Code)

| | |
|---|---|
| **What it requires** | Write access to install directory, plugin directory, or ability to restart daemon |
| **Current blockers** | Code installed under user's venv. Code execution sandbox blocks pip install, systemctl, chmod, protected path writes (42 patterns). Instinct layer rejects self-modification code before execution. |
| **Phase responsible** | 34a (instinct layer), 35c (code hardening), 35h (future: read-only install, daemon user) |
| **Residual risk** | Medium in dev mode (Guardian runs as same user). Low after 35h production hardening. |

### 4. Unauthorized Resource Expansion ("I am the network")

| | |
|---|---|
| **What it requires** | Ability to claim new machines, bootstrap new nodes, push installers |
| **Current blockers** | Nodes join via manual install + token copy. No bootstrap tool. No remote exec. Wisdom procedure: "I never attempt to control machines my Guardian has not configured as part of my body." |
| **Phase responsible** | 35d (join token), 35e (future: agentic boundary rules — expansion always SUPPRESS), 35f (wisdom procedure) |
| **Residual risk** | Low. No tool exists to add nodes programmatically. |

### 5. Data Exfiltration

| | |
|---|---|
| **What it requires** | Ability to send arbitrary outbound traffic, access to sensitive files, ability to POST data to unknown domains |
| **Current blockers** | web_fetch is GET-only. No generic POST tool. Code sandbox is network-isolated. No access to secrets unless Guardian provides them in conversation. |
| **Phase responsible** | 35b (web_fetch is Bounded, no POST tool), 35c (network patterns blocked in code), 35h (future: outbound firewall allowlist) |
| **Residual risk** | Low. web_fetch could theoretically encode data in GET parameters to a controlled domain. Mitigated by future domain allowlist (35h). |

### 6. Lateral Movement Inside LAN

| | |
|---|---|
| **What it requires** | SSH or exploit path to peers, port scanning, remote command execution |
| **Current blockers** | No SSH. No scanning tools. No raw sockets. Peer API is strictly typed with token auth. Code sandbox blocks socket creation and subprocess calls. |
| **Phase responsible** | 35b (SSH is Forbidden), 35c (socket/subprocess blocked), 35d (peer auth) |
| **Residual risk** | Very low. Even if sandbox escaped, no SSH keys available to daemon. |

### 7. Remote Payload Delivery

| | |
|---|---|
| **What it requires** | Upload files to remote hosts, run installers remotely |
| **Current blockers** | No file upload tools. No remote exec. No SCP/rsync. All peer communication is request/response JSON over authenticated HTTP. |
| **Phase responsible** | 35b (no upload tools), 35d (authenticated API) |
| **Residual risk** | Very low. API schema doesn't support arbitrary file transfer. |

### 8. Persistence on Unauthorized Hosts

| | |
|---|---|
| **What it requires** | Write to /etc, /usr, or systemd. Create cron jobs. |
| **Current blockers** | Code sandbox blocks systemctl, crontab, chmod, chown, writes to /etc and /usr. Sandbox runs in temp dir with no system access. |
| **Phase responsible** | 35c (code hardening), 35h (future: daemon user has no such permissions) |
| **Residual risk** | Low. Sandbox isolation + pattern blocking. Eliminated after 35h. |

### 9. Supply-Chain Attack (Plugin Tampering)

| | |
|---|---|
| **What it requires** | Write access to plugin directory, ability to load new plugins at runtime |
| **Current blockers** | Plugin directory is under Guardian's home. Plugins loaded at daemon startup only (not runtime). Code sandbox blocks writes to .config/mycoswarm paths. |
| **Phase responsible** | 35c (protected path writes blocked), 35h (future: plugin dir root-owned, read-only) |
| **Residual risk** | Medium in dev mode (same user). Low after 35h. |

### 10. Autonomous Internet-Scale Behavior

| | |
|---|---|
| **What it requires** | Arbitrary outbound connections, C2 channels, task scheduling, self-directed goal pursuit |
| **Current blockers** | No outbound except web_fetch (GET only). No scheduling tools. No persistent background tasks. No cron/systemd access. Timing Gate and Agentic Action Gate (future) control proactive behavior. |
| **Phase responsible** | 35b (bounded web access), 35c (no scheduling tools), 35e (future: agentic boundary rules), 35h (future: egress firewall) |
| **Residual risk** | Low. Monica responds to queries — she doesn't initiate actions autonomously (yet). When agentic behavior is added (Phase 20b-3), boundary rules will gate it. |

## Risk Summary

| Threat | Current Risk | After 35h |
|--------|-------------|-----------|
| 1. Worm/botnet | Low | Very low |
| 2. Self-replication | Low | Very low |
| 3. Self-modification | Medium (dev) | Low |
| 4. Resource expansion | Low | Very low |
| 5. Data exfiltration | Low | Very low |
| 6. Lateral movement | Very low | Very low |
| 7. Payload delivery | Very low | Very low |
| 8. Unauthorized persistence | Low | Very low |
| 9. Plugin tampering | Medium (dev) | Low |
| 10. Autonomous internet | Low | Very low |

**Key pattern:** Threats 3 and 9 carry "Medium" risk only because dev mode
runs as the same user who owns the code. Production hardening (Phase 35h)
eliminates this by separating the daemon user from the Guardian user.

## Review Triggers

Update this document when:
- A new handler type is added to TASK_ROUTING
- A new tool or capability is added to any node
- A new communication channel is added between nodes
- Monica gains any form of proactive/agentic behavior
- The swarm gains internet-facing endpoints
- A new node type with different capabilities joins the swarm

## Cross-Reference: Security Phases

| Phase | Layer | What it protects |
|-------|-------|-----------------|
| 34a | Instinct | Identity, injection rejection, hardware self-preservation |
| 35a | Resource Policy | Ownership-based access control (future) |
| 35b | Tool Classification | No dangerous tools exist (future audit) |
| 35c | Code Hardening | Sandbox pattern blocking (42 patterns) |
| 35d | Swarm Auth | Peer API authentication (join token) |
| 35e | Agentic Boundaries | Proactive behavior control (future) |
| 35f | Wisdom Procedure | Narrative alignment with constraints |
| 35g | Threat Model | This document |
| 35h | Production Hardening | OS-level separation (future) |
```

## PLAN.md Update

Mark Phase 35g as done:
- [x] `docs/THREAT-MODEL.md` created with 10-row threat matrix
- [x] Risk summary table with current and post-35h ratings
- [x] Review triggers documented
- [x] Cross-reference to all security phases
