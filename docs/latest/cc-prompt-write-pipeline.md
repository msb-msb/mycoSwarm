# Multi-Step /write Pipeline

## What This Is

The current /write command dumps everything into one shot — outline and
draft in a single uncontrolled flow. Monica kept revising outlines and
then drafted without clear stages. This adds explicit state management
so the pipeline is: outline → approve → research → draft → save.

## The Pipeline

```
/write "topic"
  │
  ├─ 1. HARDWARE CONTEXT injected (auto, silent)
  ├─ 2. SKILL.md retrieved via RAG (auto)
  │
  ▼
  OUTLINE presented to Guardian
  │
  ├─ Guardian types feedback → Monica revises outline
  ├─ Guardian types "go" / "approved" → advances to research
  │
  ▼
  RESEARCH phase (auto)
  │
  ├─ 3-5 web searches based on topic + outline
  ├─ Results injected into context
  ├─ Progress shown to Guardian
  │
  ▼
  DRAFT written by Monica
  │
  ├─ Full markdown with frontmatter
  ├─ Uses research data + hardware context
  │
  ▼
  SAVE prompt
  │
  ├─ Guardian reviews, saves to ~/insiderllm-drafts/
  └─ Article mode deactivated
```

## Implementation: State Machine

Add a simple state tracker for article mode:

```python
from enum import Enum

class ArticleState(Enum):
    INACTIVE = "inactive"
    OUTLINING = "outlining"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    REVIEWING = "reviewing"

# Module-level state
_article_state = ArticleState.INACTIVE
_article_topic = ""
```

### State transitions in the chat loop:

```python
# /write command triggers OUTLINING
if user_input.strip().lower().startswith("/write"):
    topic = user_input.strip()[6:].strip().strip('"').strip("'")
    if not topic:
        print("\n✍️  Usage: /write \"Article topic\"")
        continue

    _article_state = ArticleState.OUTLINING
    _article_topic = topic

    # Inject hardware context (Prompt 2)
    hw_context = _gather_hardware_context()
    perf_context = _gather_recent_performance()

    # Inject article mode system prompt + hardware
    # ... (existing /write setup code)

    print(f"\n✍️  Article mode: OUTLINE phase")
    print(f"   Type feedback to revise, or 'go' to start research.\n")
    continue

# Check for state transitions on each user message
if _article_state == ArticleState.OUTLINING:
    approval_words = {"go", "approved", "yes", "lgtm", "looks good",
                      "draft it", "write it", "go ahead"}
    if user_input.strip().lower() in approval_words:
        _article_state = ArticleState.RESEARCHING
        print(f"\n✍️  Article mode: RESEARCH phase")

        # Generate and run searches (Prompt 1)
        queries = _generate_research_queries(_article_topic, last_response)
        print(f"🔍 Running {len(queries)} searches...")
        for i, q in enumerate(queries, 1):
            print(f"   [{i}/{len(queries)}] {q}")

        research_context = await _run_article_research(queries)
        print(f"   ✅ Research complete\n")

        # Inject research and transition to drafting
        _article_state = ArticleState.DRAFTING
        messages.append({"role": "system", "content": f"""
## Research Results

Use ONLY these facts and numbers in your draft. Do NOT invent specs,
prices, or benchmarks. If data is missing, say so — don't guess.

{research_context}
"""})
        # Auto-send draft request
        messages.append({"role": "user", "content":
            "Write the full article draft now using the approved outline, "
            "research results, and hardware context. Output the complete "
            "markdown including Hugo frontmatter in a ```markdown block."
        })
        print(f"✍️  Article mode: DRAFT phase\n")
        # Fall through to inference
    else:
        # User is giving feedback on outline — just let it go through
        # normal inference for Monica to revise
        pass

elif _article_state == ArticleState.DRAFTING:
    # After draft is presented, check_draft_save handles the save prompt
    # Then deactivate article mode
    pass

# After _check_draft_save completes (draft saved or skipped):
if _article_state in (ArticleState.DRAFTING, ArticleState.REVIEWING):
    _article_state = ArticleState.INACTIVE
    _article_topic = ""
    print("✍️  Article mode complete.\n")
```

### Cancel at any time:

```python
if user_input.strip().lower() in ("/write off", "/write cancel", "/cancel"):
    if _article_state != ArticleState.INACTIVE:
        _article_state = ArticleState.INACTIVE
        _article_topic = ""
        print("\n✍️  Article mode cancelled.\n")
        continue
```

### Show current state:

```python
# In the prompt line, show article state if active:
if _article_state != ArticleState.INACTIVE:
    state_indicator = {
        ArticleState.OUTLINING: "📝 OUTLINE",
        ArticleState.RESEARCHING: "🔍 RESEARCH",
        ArticleState.DRAFTING: "✍️  DRAFT",
        ArticleState.REVIEWING: "👁  REVIEW",
    }.get(_article_state, "")
    prompt = f"🍄 [{state_indicator}]> "
else:
    prompt = "🍄> "
```

## Key Behavioral Rules

1. **OUTLINE phase:** Monica presents outline. Guardian gives feedback or
   approves. Monica does NOT auto-draft. She waits for "go".

2. **RESEARCH phase:** Automatic. No user interaction needed. Show progress.
   Happens immediately after approval.

3. **DRAFT phase:** Monica writes using outline + research + hardware context.
   One shot. No re-outlining. If Guardian wants revisions, they give
   feedback and Monica revises the draft (stays in DRAFT phase).

4. **SAVE phase:** _check_draft_save detects markdown block and offers to
   save. After save (or skip), article mode deactivates.

5. **Monica should NOT:**
   - Present a new outline after drafting
   - Report procedure tags ([P1], [P3]) in the article
   - Make up numbers — use research data or say "data unavailable"
   - Use filler intros ("In this article we will explore...")

## System Prompt Additions

Add to the article-mode system prompt:

```
CRITICAL RULES FOR ARTICLE MODE:
- You are in a structured pipeline: outline → research → draft
- Do NOT present additional outlines after the draft
- Do NOT invent specs, prices, benchmarks, or tok/s numbers
- Use ONLY data from: research results, hardware context, or your session history
- If you don't have a number, write "benchmark data needed" — never guess
- Do NOT include procedure tags like [P1] or [P3] in article text
- Do NOT start with "In this article, we will explore..."
- Start with a concrete hook: a number, a problem, a direct answer
```

## Test

```bash
mycoswarm chat
/write "DeepSeek Models Guide"
```

Expected flow:
```
✍️  Article mode: OUTLINE phase
   Type feedback to revise, or 'go' to start research.

🍄 [📝 OUTLINE]> (Monica presents outline)
🍄 [📝 OUTLINE]> go

✍️  Article mode: RESEARCH phase
🔍 Running 4 searches...
   [1/4] DeepSeek Models
   [2/4] DeepSeek Models specs benchmarks 2026
   [3/4] DeepSeek Models vs alternatives comparison
   [4/4] DeepSeek Models VRAM requirements local
   ✅ Research complete

✍️  Article mode: DRAFT phase

🍄 [✍️  DRAFT]> (Monica writes full article with real data)

💾 Save draft to /home/minotaur/insiderllm-drafts/deepseek-models-guide.md? (y/n)
🍄> y
   ✅ Draft saved

✍️  Article mode complete.
```

## PLAN.md Update

Add under the appropriate section:
- [x] /write pipeline: outline → approve → web research → draft → save
- [x] Hardware self-injection in article mode
- [x] Web search integration in article mode
- [x] Article state machine with visual indicators
