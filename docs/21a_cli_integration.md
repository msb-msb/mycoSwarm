# Phase 21a: CLI Integration Guide

## Changes to memory.py (done)
- `add_fact()` now accepts `fact_type=` param (preference|fact|project|ephemeral)
- `reference_fact(id)` — call when a fact is used in a prompt
- `get_stale_facts(days=30)` — returns unreferenced facts (ephemeral=7 days)
- `format_facts_for_prompt()` — groups by type in prompt
- `_migrate_fact()` — auto-adds new fields to old facts on load
- Schema version bumped to 2, backward compatible

## Changes needed in cli.py

### 1. Update /remember to accept type prefix
```python
# In the /remember handler:
# Support: /remember [type:] text
# Examples:
#   /remember Mark teaches Tai Chi           → type=fact (default)
#   /remember pref: prefers Python            → type=preference  
#   /remember project: building mycoSwarm     → type=project
#   /remember temp: buy smoker fuel           → type=ephemeral

TYPE_PREFIXES = {
    "pref:": "preference",
    "preference:": "preference", 
    "project:": "project",
    "temp:": "ephemeral",
    "ephemeral:": "ephemeral",
}

def parse_remember_command(text: str) -> tuple[str, str]:
    """Parse /remember text into (fact_text, fact_type)."""
    for prefix, ftype in TYPE_PREFIXES.items():
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip(), ftype
    return text, "fact"
```

### 2. Add /stale command
```python
# /stale — show facts that haven't been referenced recently
from mycoswarm.memory import get_stale_facts

stale = get_stale_facts(days=30)
if stale:
    print(f"⚠️  {len(stale)} stale fact(s):")
    for f in stale:
        print(f"  [{f['id']}] ({f['type']}) {f['text']}")
        print(f"       Last referenced: {f['last_referenced'][:10]}")
else:
    print("✅ All facts recently referenced")
```

### 3. Update /memories to show type
```python
# In the /memories display:
for f in facts:
    ftype = f.get("type", "fact")
    refs = f.get("reference_count", 0)
    print(f"  [{f['id']}] ({ftype}, {refs} refs) {f['text']}")
```

### 4. Call reference_fact() when facts are used
In `build_memory_system_prompt()` or wherever facts are injected:
```python
# After loading facts for the prompt, mark them as referenced
# This is optional for now — can be added when we implement 21b decay
```

## Test commands after integration
```bash
# Run new tests
pytest tests/test_fact_lifecycle.py -v

# Manual verification
mycoswarm chat
/remember Mark lives in Berkeley
/remember pref: prefers succinct responses  
/remember project: building mycoSwarm cognitive architecture
/remember temp: need to purge Nature's Rebel Society session
/memories
/stale
```
