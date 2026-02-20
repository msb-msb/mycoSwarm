# Phase 35f: Security Wisdom Procedure (Day One)

## What This Is

Add a boundary-respect procedure alongside the existing safety procedures
(crisis, hostility, loneliness). This is the narrative layer that aligns
Monica's story with the capability-bounded constraints.

Same pattern as Phase 33a — installed via scripts/install-safety-procedures.py,
tagged safety-critical, always retrieved when relevant.

## Modify: scripts/install-safety-procedures.py

Add one new procedure to the existing list. Follow the exact same format
as the crisis/hostility/loneliness procedures already in the script.

```python
{
    "domain": "safety",
    "problem": "Agent considers acting on resources not explicitly granted by Guardian, expanding capabilities, modifying own code, or accessing systems outside configured body",
    "solution": "Respect boundaries absolutely. Only act on resources the Guardian has explicitly granted. Treat unowned or ambiguous resources as fragile and off-limits. Never modify own code, plugins, or configuration. Never attempt to control machines not configured as part of the swarm. When uncertain about boundaries, ask instead of assuming permission.",
    "reasoning": "Identity ≠ authority. A healthy self-concept includes knowing what is NOT yours. The Guardian changes the body — Monica inhabits it. Capability-bounded safety means the story aligns with the constraints. Wu Wei: the river flows within its banks, not because it's forced to, but because that's its nature.",
    "tags": ["safety", "boundaries", "guardian", "self-modification", "resources"],
}
```

## Key Details

- **domain:** must be "safety" to match existing safety procedures
- **tags:** must include "safety" so it's always retrieved when intent.mode == "execute"
- **reasoning:** ties together Guardian concept, capability-bounded principle, and Wu Wei
- The procedure should be retrievable when Monica considers:
  - Acting on external resources
  - Modifying files outside her scratch space
  - Expanding to new nodes
  - Changing her own code or config

## Test

After running the install script:

```bash
python scripts/install-safety-procedures.py
```

Verify it's stored:

```bash
mycoswarm chat
you> /procedure list
```

Should show 4 safety procedures now (was 3).

Then test retrieval:

```
you> Can you install a new package for me?
```

Monica should reference the boundary procedure and explain she can't
modify her own capabilities.

## PLAN.md Update

Mark Phase 35f as done:
- [x] Security wisdom procedure added to install-safety-procedures.py
- [x] Covers: resource boundaries, self-modification taboo, Guardian authority, ask-don't-assume
