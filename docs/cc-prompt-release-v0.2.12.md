# Release v0.2.12

## What's in this release
- **Phase 35d: Swarm Authentication** — join token required for all peer API requests (except /health), constant-time validation, /token command
- **Phase 35c: Code Execution Hardening** — 42 self-modification patterns blocked before sandbox execution, integrated into instinct layer
- **Phase 35f: Security Wisdom Procedure** — boundary respect procedure (4th safety procedure)
- **Phase 35g: Threat Model** — docs/THREAT-MODEL.md with 10-row threat matrix

## Release checklist

1. Bump version in pyproject.toml to 0.2.12
2. Run full test suite: `pytest -v`
3. Count tests and note the number
4. Build wheel: `python -m build`
5. Show me the upload command (I'll run twine manually)
6. Create GitHub release:
   ```
   gh release create v0.2.12 --title "v0.2.12 — Security Architecture" --notes "Phase 35: Security Architecture — Boundary Enforcement

Swarm Authentication (35d): Join token required for all peer API requests. Nodes without valid token get 403. /health exempt for connectivity checks. /token slash command. Constant-time comparison via secrets.compare_digest.

Code Execution Hardening (35c): 42 regex patterns block self-modification attempts in code_run sandbox — package install, shell escape, protected path writes, systemctl, network escape, eval/exec. Same instinct layer architecture, <1ms.

Security Wisdom Procedure (35f): 4th safety procedure — boundary respect, Guardian authority, self-modification taboo.

Threat Model (35g): docs/THREAT-MODEL.md — 10-row threat matrix mapping every boundary violation scenario to mechanical blockers. Living document with review triggers."
   ```
7. Update PLAN.md with release line under Done

DO NOT skip step 6. Releases without GitHub tags are invisible.
