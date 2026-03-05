"""Agentic research agent — multi-turn tool-calling loop via Ollama.

Replaces the old extractor + gap-filler pipeline steps with a single
agent that plans searches, evaluates depth, and iterates until the
research is thorough enough (depth >= 7) or 5 rounds are exhausted.

Uses Ollama's native tool calling with qwen3.5:9b (or fallback model).
"""

import json
import os
import re
import time

import httpx

from mycoswarm.solo import web_search_solo
from mycoswarm.pipeline import _perplexity_search, fetch_page


# --- Ollama tool schemas ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. Returns a list of results "
                "with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch and extract text content from a URL. Returns the "
                "page text (up to 2000 words)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_depth",
            "description": (
                "Self-evaluate the depth and completeness of your research so far. "
                "Call this after each search round to decide whether to continue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Depth score 1-10. 1=surface level, 10=comprehensive expert analysis.",
                    },
                    "strong_claims": {
                        "type": "string",
                        "description": "Claims well-supported by sources found so far.",
                    },
                    "weak_areas": {
                        "type": "string",
                        "description": "Topics or claims still lacking evidence or depth.",
                    },
                    "stop": {
                        "type": "boolean",
                        "description": "True if research is sufficient, false to continue searching.",
                    },
                },
                "required": ["depth", "strong_claims", "weak_areas", "stop"],
            },
        },
    },
]

# --- Config ---

OLLAMA_CONFIG = {
    "num_ctx": 32768,
    "temperature": 0.7,
    "num_predict": 4096,
    "stream": False,
}

MAX_ROUNDS = 5
MIN_DEPTH = 7
CONTEXT_WORD_LIMIT = 20000

# evaluate_depth tool only — used to force evaluation when model skips it
EVALUATE_DEPTH_TOOL = [TOOLS[2]]


def _word_count(text: str) -> int:
    return len(text.split())


def _summarize_findings(messages: list[dict]) -> str:
    """Extract accumulated findings from message history for context compression."""
    findings = []
    for msg in messages:
        if msg["role"] == "tool" and msg.get("content"):
            content = msg["content"]
            # Keep search results and page fetches but truncate
            if len(content) > 500:
                content = content[:500] + "..."
            findings.append(content)
    combined = "\n\n".join(findings)
    words = combined.split()
    if len(words) > 3000:
        combined = " ".join(words[:3000]) + "\n[findings condensed]"
    return combined


class ResearchAgent:
    """Multi-turn research agent using Ollama tool calling."""

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.debug_log: list[str] = []

    def _log(self, msg: str):
        self.debug_log.append(msg)
        print(f"   🔬 {msg}")

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Single Ollama /api/chat call. Returns the response dict."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": OLLAMA_CONFIG,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=300) as client:
            resp = client.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def _execute_tool(self, name: str, args: dict, debug: bool = False) -> str:
        """Execute a tool call and return the result as a string."""
        if name == "web_search":
            query = args.get("query", "")
            self._log(f"searching: {query}")
            results = web_search_solo(query, max_results=8)
            if not results:
                # Perplexity fallback
                results = _perplexity_search(query, debug=debug)
                if results:
                    self._log(f"  → perplexity fallback: {len(results)} results")
            if not results:
                return "No results found."
            lines = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                lines.append(f"- {title}\n  URL: {url}\n  {snippet}")
            return "\n".join(lines)

        elif name == "web_fetch":
            url = args.get("url", "")
            self._log(f"fetching: {url[:80]}")
            result = fetch_page(url)
            if not result:
                return f"Failed to fetch {url}"
            return result

        elif name == "evaluate_depth":
            depth = args.get("depth", 0)
            strong = args.get("strong_claims", "")
            weak = args.get("weak_areas", "")
            stop = args.get("stop", False)
            self._log(f"depth={depth}/10 stop={stop}")
            self.debug_log.append(f"  strong: {strong[:200]}")
            self.debug_log.append(f"  weak: {weak[:200]}")
            return json.dumps({
                "depth": depth,
                "strong_claims": strong,
                "weak_areas": weak,
                "stop": stop,
                "message": f"Depth score recorded: {depth}/10",
            })

        return f"Unknown tool: {name}"

    def _total_words(self, messages: list[dict]) -> int:
        """Count total words across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += _word_count(content)
        return total

    def run(
        self,
        topic: str,
        reference_data: str = "",
        context: str = "",
        workspace_dir: str = ".",
        step_name: str = "research",
        debug: bool = False,
    ) -> str:
        """Run the research loop. Returns the final research bundle text."""
        start = time.time()

        system_prompt = self._build_system_prompt(topic)

        # Build initial user message
        user_parts = [f"Research topic: {topic}"]
        if reference_data:
            user_parts.append(f"\n{reference_data}")
        if context:
            user_parts.append(f"\n## Author Context\n{context}")
        user_parts.append(
            "\nBegin your research. Use web_search to find data, web_fetch to read "
            "promising pages, and evaluate_depth after each search round. "
            "Aim for depth >= 7 before stopping."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        last_depth = 0
        rounds = 0

        for round_num in range(1, MAX_ROUNDS + 1):
            rounds = round_num

            # Context safety valve
            if self._total_words(messages) > CONTEXT_WORD_LIMIT:
                self._log("context limit hit — condensing findings")
                summary = _summarize_findings(messages)
                # Keep system + first user + condensed summary
                messages = [
                    messages[0],  # system
                    messages[1],  # initial user
                    {"role": "user", "content": f"## Condensed findings so far\n{summary}\n\nContinue researching. Focus on weak areas."},
                ]

            # Call model with tools
            try:
                response = self._chat(messages, tools=TOOLS)
            except Exception as e:
                self._log(f"inference error: {e}")
                break

            msg = response.get("message", {})
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Append assistant message
            assistant_msg = {"role": role}
            if content:
                assistant_msg["content"] = content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if debug and content:
                # Strip think tags for preview
                preview = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if preview:
                    print(f"   🐛 assistant: {preview[:150]}...")

            # Execute tool calls
            if not tool_calls:
                self._log("no tool calls — ending loop")
                break

            should_stop = False
            depth_called = False
            search_count = 0
            fetch_count = 0

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                # Arguments may be a dict or a JSON string
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                result = self._execute_tool(name, args, debug=debug)

                # Append tool result
                messages.append({
                    "role": "tool",
                    "content": result,
                })

                # Track per-round stats
                if name == "web_search":
                    search_count += 1
                elif name == "web_fetch":
                    fetch_count += 1

                # Check if evaluate_depth says stop
                if name == "evaluate_depth":
                    depth_called = True
                    depth = args.get("depth", 0)
                    last_depth = depth
                    stop = args.get("stop", False)
                    if depth >= MIN_DEPTH or stop:
                        self._log(f"stopping: depth={depth}, stop={stop}")
                        should_stop = True

            # Force evaluate_depth if the model didn't call it
            if not depth_called and not should_stop:
                messages.append({
                    "role": "user",
                    "content": (
                        "Stop and assess your research progress. Call evaluate_depth now. "
                        "Rate depth 1-10 where:\n"
                        "1-3: Surface level, could find on any blog\n"
                        "4-6: Good facts but missing why/analysis\n"
                        "7-8: Deep, has specifics others don't cover\n"
                        "9-10: Expert level, original synthesis\n\n"
                        "Set stop=true if depth >= 7 or you believe further "
                        "searching won't improve quality."
                    ),
                })
                try:
                    eval_response = self._chat(messages, tools=EVALUATE_DEPTH_TOOL)
                    eval_msg = eval_response.get("message", {})
                    eval_content = eval_msg.get("content", "")
                    eval_tool_calls = eval_msg.get("tool_calls", [])

                    # Append assistant message
                    eval_assistant = {"role": eval_msg.get("role", "assistant")}
                    if eval_content:
                        eval_assistant["content"] = eval_content
                    if eval_tool_calls:
                        eval_assistant["tool_calls"] = eval_tool_calls
                    messages.append(eval_assistant)

                    if eval_tool_calls:
                        for etc in eval_tool_calls:
                            efunc = etc.get("function", {})
                            ename = efunc.get("name", "")
                            eargs = efunc.get("arguments", {})
                            if isinstance(eargs, str):
                                try:
                                    eargs = json.loads(eargs)
                                except json.JSONDecodeError:
                                    eargs = {}
                            if ename == "evaluate_depth":
                                result = self._execute_tool(ename, eargs, debug=debug)
                                messages.append({"role": "tool", "content": result})
                                depth = eargs.get("depth", 0)
                                last_depth = depth
                                stop = eargs.get("stop", False)
                                if depth >= MIN_DEPTH or stop:
                                    self._log(f"stopping: depth={depth}, stop={stop}")
                                    should_stop = True
                    else:
                        # Model refused to call tool — assume surface level
                        self._log("⚠️ forced eval produced no tool call — defaulting depth=3")
                        last_depth = 3
                except Exception as e:
                    self._log(f"⚠️ forced eval failed: {e} — defaulting depth=3")
                    last_depth = 3

            self._log(
                f"round {round_num}/{MAX_ROUNDS} → "
                f"searched {search_count}, fetched {fetch_count} → "
                f"depth: {last_depth}/10"
            )

            if should_stop:
                break

        # --- Final compilation ---
        self._log("compiling final research bundle")
        messages.append({
            "role": "user",
            "content": (
                "Research phase complete. Now compile ALL your findings into a "
                "structured research bundle with these sections:\n\n"
                "## Key Facts & Data Points\n"
                "(every factual claim with source URL)\n\n"
                "## Price Data\n"
                "(all prices found, new vs used, with sources)\n\n"
                "## Benchmark Data\n"
                "(performance numbers, tok/s, comparisons, with sources)\n\n"
                "## Key Specs\n"
                "(VRAM, TDP, architecture for each product mentioned)\n\n"
                "## Expert Opinions & Analysis\n"
                "(what experts/reviewers are saying, with sources)\n\n"
                "## Gaps\n"
                "(what data is still missing)\n\n"
                "## Suggested Angle\n"
                "(how InsiderLLM should cover this differently)\n\n"
                "Include source URLs for every claim. Be comprehensive."
            ),
        })

        try:
            final_response = self._chat(messages, tools=None)
            output = final_response.get("message", {}).get("content", "")
            # Strip think tags
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        except Exception as e:
            self._log(f"final compilation error: {e}")
            output = "Research agent failed during final compilation."

        duration = time.time() - start
        words = _word_count(output)
        self._log(f"done: {words} words, {rounds} rounds, depth={last_depth}, {duration:.0f}s")

        # --- Write debug file ---
        if debug:
            debug_path = os.path.join(workspace_dir, f"{step_name}-debug.md")
            with open(debug_path, "w") as f:
                f.write(f"# Research Agent Debug Log\n\n")
                f.write(f"- Topic: {topic}\n")
                f.write(f"- Model: {self.model}\n")
                f.write(f"- Ollama URL: {self.ollama_url}\n")
                f.write(f"- Rounds: {rounds}/{MAX_ROUNDS}\n")
                f.write(f"- Final depth: {last_depth}/10\n")
                f.write(f"- Duration: {duration:.0f}s\n")
                f.write(f"- Output words: {words}\n\n")
                f.write("## Tool Call Log\n\n")
                for line in self.debug_log:
                    f.write(f"- {line}\n")
            print(f"   🐛 debug log: {debug_path}")

        return output

    def _build_system_prompt(self, topic: str) -> str:
        """Build the system prompt for the research agent."""
        return f"""You are a research analyst for InsiderLLM.com, a site focused on local AI hardware and inference.

Your task is to research a topic thoroughly using web search and page fetching tools. You work in rounds:

1. **Plan**: Identify what data you need (prices, benchmarks, specs, expert opinions)
2. **Search**: Use web_search to find relevant sources
3. **Fetch**: Use web_fetch on the most promising URLs to get full content
4. **Evaluate**: Use evaluate_depth to assess your research completeness

## Tools Available
- `web_search(query)` — search the web, returns titles + URLs + snippets
- `web_fetch(url)` — fetch full page text from a URL
- `evaluate_depth(depth, strong_claims, weak_areas, stop)` — self-evaluate and decide whether to continue

## Depth Rubric
- 1-3: Surface level, just headlines and basic specs
- 4-5: Decent coverage but missing prices, benchmarks, or expert opinions
- 6-7: Good coverage with specific numbers, multiple sources, some analysis
- 8-9: Comprehensive with cross-referenced data, expert opinions, market context
- 10: Expert-level with unique insights, historical context, forward-looking analysis

## Rules
- Use /think before planning your search strategy
- Call evaluate_depth after each search round
- Aim for depth >= 7 before stopping
- Always note source URLs for every fact
- Search for CURRENT data (2025-2026 prices, latest benchmarks)
- Do NOT make up data — only report what you find in search results
- If a search returns no results, try different query terms
- Fetch pages that look like they have detailed specs, benchmarks, or pricing

## Mandatory Coverage
For any GPU/hardware topic, you MUST search for:
- Current pricing (new and used/eBay)
- Performance benchmarks (tok/s for LLM inference if applicable)
- VRAM and memory bandwidth specs
- Power consumption (TDP)
- Expert recommendations and comparisons

Topic: {topic}"""
