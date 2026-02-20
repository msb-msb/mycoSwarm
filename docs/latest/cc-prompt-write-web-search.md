# Wire Web Search into /write Article Mode

## What This Is

The /write command currently dumps Monica straight into drafting with no
research step. She hallucinates specs, prices, and benchmarks because she
has no access to current data. The research pipeline (web_search, web_fetch)
already exists in the codebase but isn't called from article mode.

This adds an automatic research phase between outline approval and drafting.

## How It Should Work

Current flow:
```
/write "topic" → outline → (user approves) → draft → save
```

New flow:
```
/write "topic" → outline → (user approves) → AUTO-RESEARCH → draft with research context → save
```

## Implementation

### Step 1: After outline approval, trigger research

When the user approves the outline (types "go", "approved", "yes", etc.),
before drafting, the system should:

1. Extract 3-5 search queries from the topic and outline
2. Run web_search for each query
3. Optionally web_fetch the top 1-2 results for more detail
4. Collect all research into a structured context block
5. Inject that context into the messages before the draft request

### Step 2: Generate search queries from outline

Add a function that takes the topic and outline and generates search queries:

```python
def _generate_research_queries(topic: str, outline: str) -> list[str]:
    """Generate 3-5 search queries from the article topic and outline.

    Uses simple keyword extraction, not an LLM call.
    """
    queries = []

    # Always search the main topic
    queries.append(topic)

    # Add "specs" or "benchmarks" variant
    queries.append(f"{topic} specs benchmarks 2026")

    # Add "vs" comparison variant
    queries.append(f"{topic} vs alternatives comparison")

    # Add VRAM/hardware variant for hardware-related topics
    hardware_keywords = ["gpu", "vram", "model", "llm", "rtx", "deepseek",
                         "llama", "mistral", "qwen", "stable diffusion", "flux"]
    if any(kw in topic.lower() for kw in hardware_keywords):
        queries.append(f"{topic} VRAM requirements local")

    # Add pricing variant for buying guides
    buying_keywords = ["buying", "guide", "budget", "vs", "compare", "best"]
    if any(kw in topic.lower() for kw in buying_keywords):
        queries.append(f"{topic} price 2026")

    return queries[:5]  # Cap at 5
```

### Step 3: Run the searches

Use the existing web search infrastructure. Check how the `/research` or
`/auto` commands call web_search — reuse that pattern.

```python
import httpx

async def _run_article_research(queries: list[str], swarm_token: str | None = None) -> str:
    """Run web searches and compile results into a research context block."""
    results = []

    for query in queries:
        try:
            # Use the daemon's search endpoint if available,
            # or call the search function directly
            # Adapt this to match your actual web_search implementation
            search_results = await web_search(query, max_results=3)
            for r in search_results:
                results.append({
                    "query": query,
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url": r.get("url", ""),
                })
        except Exception as e:
            logger.warning("Research search failed for %r: %s", query, e)

    # Format as context block
    if not results:
        return "No research results found. Draft based on existing knowledge."

    context = "## Research Results\n\n"
    for r in results:
        context += f"**{r['title']}** ({r['url']})\n"
        context += f"  {r['snippet']}\n\n"

    return context
```

**NOTE TO CC:** The above is illustrative. You need to find the actual
web_search implementation in the codebase (check how `/research` or `/auto`
commands work) and reuse it. The key is:
- Run 3-5 searches
- Collect titles, snippets, and URLs
- Format as a text block that gets injected into messages

### Step 4: Inject research into article context

After research completes, append it to the messages before asking Monica
to draft:

```python
# After user approves outline and research runs:
research_context = await _run_article_research(queries)

research_msg = f"""Here is current research data for your article. Use these
facts and numbers in your draft. Do NOT make up specs, prices, or benchmarks
— use only what's provided here or what you know from direct experience on
the mycoSwarm network.

{research_context}

Now write the full article draft based on the approved outline and this
research. Include all data in appropriate tables and comparisons."""

messages.append({"role": "system", "content": research_msg})
messages.append({"role": "user", "content": "Draft the full article now using the research above."})
```

### Step 5: Show research status to user

Print progress so the Guardian knows what's happening:

```python
print(f"\n🔍 Researching: running {len(queries)} searches...")
for i, q in enumerate(queries, 1):
    print(f"   [{i}/{len(queries)}] {q}")

research_context = await _run_article_research(queries)

print(f"   ✅ Found {len(results)} results")
print(f"\n✍️  Drafting article with research context...\n")
```

## Detection of Outline Approval

The /write flow needs to know when the user has approved the outline so it
can trigger research. Add a simple state flag:

```python
_article_mode = False
_article_topic = ""
_article_outline_approved = False

# In /write handler:
_article_mode = True
_article_topic = topic
_article_outline_approved = False

# After each user message in article mode, check if it's an approval:
if _article_mode and not _article_outline_approved:
    approval_words = {"go", "approved", "yes", "lgtm", "looks good", "draft it", "write it"}
    if user_input.strip().lower() in approval_words:
        _article_outline_approved = True
        # Trigger research phase here
        queries = _generate_research_queries(_article_topic, last_response)
        # ... run research and inject context
```

## Exiting Article Mode

Add a way to exit article mode:

```python
if user_input.strip().lower() in ("/write off", "/write cancel"):
    _article_mode = False
    _article_outline_approved = False
    print("\n✍️  Article mode deactivated.\n")
    continue
```

## Test

```bash
mycoswarm chat
/write "DeepSeek Models Guide"
```

1. Monica presents outline
2. Type "go"
3. Should see: "🔍 Researching: running 4 searches..."
4. Research results printed
5. Monica drafts with real data
6. Draft should contain actual specs and benchmarks from search results

## Key Principle

Monica should NEVER make up numbers in article mode. The research injection
gives her real data. The system prompt reinforcement tells her to only use
provided facts. If she can't find a number, she should say "benchmark data
unavailable" rather than guess.
