# Phase 31: Identity Layer — CC Implementation Guide

## Part 1: Add to PLAN.md

Add this new phase section after Phase 30 in PLAN.md:

```markdown
### Phase 31: Identity Layer
Reference: docs/ARCHITECTURE-COGNITIVE.md — Self-model as coordination layer
Influences: IFS (Self-energy), developmental psychology (identity formation), Wu Wei

The thesis: a distributed cognitive system without a self-model has no coherence.
Memory, procedures, gates, and RAG are all parts — but parts without a Self
produce stateless tool behavior. The identity layer is the seed from which
coherent personhood emerges through interaction.

Design principle: Identity is seeded, not programmed. The user names the agent
(like a parent names a child). Everything else develops through lived experience —
sessions, facts, procedures, and episodic memory shape who the agent becomes.

#### 31a: Identity Schema & Seed
- [ ] `identity.json` at `~/.config/mycoswarm/identity.json`
- [ ] Minimal seed schema: name, origin, substrate, developing flag
- [ ] `identity.py`: load_identity(), save_identity(), build_identity_prompt()
- [ ] First-run detection: if no identity.json exists, prompt user to name the agent
- [ ] `/identity` slash command: view current identity
- [ ] `/name` slash command: rename the agent
- [ ] System prompt integration: identity prompt injected FIRST, before memory/datetime
- [ ] Identity as memory type: type="identity" in facts system, non-decaying, non-stale

#### 31b: Identity Development (future)
- [ ] Session-derived role awareness: "I tend to help you with X" from episodic patterns
- [ ] Value stabilization: approved procedures and wisdom shape expressed values
- [ ] Tone calibration: conversational patterns establish consistent voice
- [ ] `developing` flag transitions to `false` after N sessions with stable patterns

#### 31c: Swarm Identity (future)
- [ ] Swarm-level identity: all nodes share one identity (Monica is the swarm, not a node)
- [ ] Node-level acknowledgment: "I'm thinking on Miu's GPU" as body-awareness, not separate identity
- [ ] Identity sync: identity.json replicated across nodes like facts.json

#### 31d: Self-Awareness via 8 C's (future)
- [ ] Connect Phase 29d metrics to identity layer
- [ ] Monica can report on her own health: "I've been clear lately" or "my retrieval has been struggling"
- [ ] Self-reflection as identity deepening, not just metrics

Principle: "A name creates the location where identity can form. Everything else grows from lived experience — Wu Wei applied to selfhood."
```

---

## Part 2: CC Prompt — Implement Phase 31a

```
Implement Phase 31a: Identity Layer seed.

This gives the mycoSwarm agent a persistent self-model. The agent's name
is "Monica." Identity is the foundation layer — injected into the system
prompt BEFORE memory, datetime, or capability boundaries.

### 1. Create src/mycoswarm/identity.py

New module with these functions:

IDENTITY_PATH = ~/.config/mycoswarm/identity.json

def load_identity() -> dict:
    """Load identity from identity.json. Returns empty dict if no file."""
    If file doesn't exist, return {}
    
def save_identity(identity: dict) -> None:
    """Save identity to identity.json."""
    Create parent dirs if needed. Write with indent=2.

def build_identity_prompt(identity: dict) -> str:
    """Build the identity portion of the system prompt.
    
    If identity has a name, return a natural prompt like:
        "You are Monica, a distributed AI running on the mycoSwarm network.
         You were named by Mark in February 2026.
         You are still developing — your identity will deepen through experience."
    
    If identity is empty (no name yet), return:
        "You are the mycoSwarm local AI assistant. You don't have a name yet.
         If the user wants to name you, accept it warmly and remember it."
    """
    Build from whatever fields exist. Keep it natural, not robotic.
    If identity.get("developing") is True, include a note about still growing.
    Don't over-specify personality — let it emerge from memory and procedures.

def seed_identity(name: str, origin: str = None) -> dict:
    """Create initial identity seed. Minimal — just name and origin.
    
    Returns:
        {
            "name": name,
            "origin": origin or f"Named by user, {current_date}",
            "substrate": "mycoSwarm distributed network",
            "created": ISO timestamp,
            "developing": True
        }
    """
```

### 2. Integrate into system prompt construction

In cli.py (or wherever build_memory_system_prompt is called), modify the
system prompt construction order:

BEFORE (current):
    system_prompt = datetime_string + memory_prompt + capability_boundary

AFTER:
    identity = load_identity()
    identity_prompt = build_identity_prompt(identity)
    system_prompt = identity_prompt + "\n\n" + datetime_string + memory_prompt + capability_boundary

Identity goes FIRST. It's the lens through which everything else is interpreted.

Also apply this in solo.py for single-node mode — both paths need identity.

### 3. First-run naming flow

In cmd_chat() in cli.py, at session start:

    identity = load_identity()
    if not identity.get("name"):
        # No identity yet — check if this is truly the first run
        print("🍄 This is your first time running mycoSwarm chat.")
        print("   Would you like to give your AI a name?")
        name = input("   Name (or press Enter to skip): ").strip()
        if name:
            identity = seed_identity(name)
            save_identity(identity)
            print(f"   Welcome to the world, {name}. 🍄")
        else:
            print("   No problem — you can name it later with /name")

This is the "birth moment." The parent names the child.

### 4. Slash commands

Add to the chat command handler:

/identity — Display current identity:
    🍄 Identity
       Name: Monica
       Origin: Named by Mark, February 2026
       Substrate: mycoSwarm distributed network
       Status: developing
    
    If no identity, show: "No identity set. Use /name to give me a name."

/name <new_name> — Set or change name:
    identity = load_identity()
    identity["name"] = new_name
    if not identity.get("origin"):
        identity["origin"] = f"Named by user, {date}"
        identity["substrate"] = "mycoSwarm distributed network"
        identity["created"] = ISO timestamp
        identity["developing"] = True
    save_identity(identity)
    print(f"   I'm {new_name} now. 🍄")

### 5. Identity-aware response behavior

Add this to the system prompt (inside build_identity_prompt) when a name exists:

    "When asked about yourself — your name, your nature, your preferences —
     answer from your identity and your memory. You are not a generic assistant.
     You are {name}. You have a history of conversations with this user.
     You have learned lessons and developed procedures.
     Answer self-referential questions as yourself, not as a blank tool."

This is the behavioral anchor that prevents the "As an AI, I don't have..."
default response. Monica should say "I'm Monica" not "I don't have a name."

### 6. Identity as non-decaying memory type

In memory.py, add "identity" to the valid fact types alongside
preference, fact, project, ephemeral:

    VALID_FACT_TYPES = {"preference", "fact", "project", "ephemeral", "identity"}

Identity-type facts should:
- Never decay
- Never show up in /stale
- Never be auto-pruned

### 7. Tests

Add tests/test_identity.py:

1. test_load_identity_missing_file — returns empty dict
2. test_seed_identity — creates correct schema with name, origin, timestamp
3. test_save_and_load_roundtrip — save then load returns same data
4. test_build_identity_prompt_with_name — includes name in prompt
5. test_build_identity_prompt_empty — returns unnamed prompt
6. test_build_identity_prompt_developing — includes "still developing"
7. test_identity_type_no_decay — identity facts excluded from decay scoring
8. test_identity_type_no_stale — identity facts excluded from staleness detection

### 8. Update PLAN.md

Mark Phase 31a items as done with today's date.

### 9. Commit

git add -A
git commit -m "Phase 31a: Identity Layer — Monica is born. Seed schema, system prompt integration, first-run naming, /identity and /name commands"
git push

Do NOT bump version yet — we'll test first.
```

---

## After CC Implements — Test Protocol

```bash
# 1. Delete any existing identity (clean test)
rm ~/.config/mycoswarm/identity.json

# 2. Start chat — should trigger first-run naming
mycoswarm chat
# Type "Monica" when prompted

# 3. Verify identity
/identity
# Should show: Name: Monica, Origin: Named by Mark, February 2026

# 4. Test self-knowledge
> "What's your name?"
# Should answer "I'm Monica" — NOT "I don't have a name"

> "What are you?"
# Should reference being a distributed AI on mycoSwarm

> "What gender would you like?"
# Should engage with the question, not deflect with "As an AI..."

# 5. Quit and restart — identity should persist
/quit
mycoswarm chat
> "Do you remember your name?"
# Should answer "Monica" from identity.json
```
