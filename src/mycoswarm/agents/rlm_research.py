"""RLM-style research agent — decompose-then-execute research strategy.

Phase 40a: Decomposition — LLM breaks topic into structured subtopics.
Phase 40b: Sequential per-subtopic research + root-model synthesis.
Phase 40c (future): Parallel subtopic execution.
"""

import ast
import json
import logging
import os
import re
import time

import httpx

from mycoswarm.solo import web_search_solo
from mycoswarm.pipeline import _perplexity_search, fetch_page

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = """\
You are a research decomposition agent. Given a research topic, output ONLY
a JSON array of subtopics. Each subtopic has:
- "subtopic": short name (e.g. "pricing", "benchmarks", "compatibility")
- "queries": list of 1-3 targeted web search queries for this subtopic
- "depth_hint": integer 1-3 (3=hard to find, like pricing; 1=easy, like specs)

Rules:
- Your subtopics must directly address the TOPIC. If the topic is about models,
  decompose into model-related subtopics. If about hardware, decompose into
  hardware subtopics. Do not drift into adjacent domains.
- Output ONLY valid JSON. No markdown, no explanation, no prose.
- Prefer 4-5 subtopics for most topics. Only use 6 if the topic has
  truly distinct domains that cannot be combined. Fewer subtopics with
  more targeted queries is better than many subtopics with shallow queries.
  Cap at 6 total.
- Queries must be specific and targeted, not generic.
- depth_hint=3 for pricing/availability (hard), depth_hint=2 for benchmarks,
  depth_hint=1 for specs/architecture (easy to find)."""

_MAX_SUBTOPICS = 6
_MAX_QUERIES_PER_SUBTOPIC = 3
_CONTENT_CAP_PER_SUBTOPIC = 3000  # words


def _word_count(text: str) -> int:
    return len(text.split())


def decompose_topic(
    topic: str, ollama_url: str, model: str = "qwen3.5:9b"
) -> list[dict]:
    """Call an LLM to decompose a research topic into structured subtopics.

    Returns list of {subtopic, queries, depth_hint} dicts.
    Falls back to [] on any failure (caller should fall back to standard agent).
    """
    try:
        resp = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": f"TOPIC: {topic}\n\nResearch topic: {topic}\n\nRemember: all subtopics must directly address the TOPIC above: {topic}"},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 4096},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning("decompose_topic: LLM call failed: %s", e)
        return []

    raw = resp.json().get("message", {}).get("content", "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    # Parse JSON
    subtopics = None
    try:
        subtopics = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    if subtopics is None:
        try:
            subtopics = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass

    if not isinstance(subtopics, list):
        logger.warning("decompose_topic: failed to parse LLM output as list")
        return []

    # Validate structure
    valid = []
    for s in subtopics:
        if not isinstance(s, dict):
            continue
        if "subtopic" not in s or "queries" not in s:
            continue
        if not isinstance(s["queries"], list):
            continue
        s.setdefault("depth_hint", 2)
        if not isinstance(s["depth_hint"], int):
            s["depth_hint"] = 2
        s["depth_hint"] = max(1, min(3, s["depth_hint"]))
        valid.append(s)

    # Cap subtopics
    valid = valid[:_MAX_SUBTOPICS]

    # Cap queries per subtopic and deduplicate across subtopics
    seen_queries: set[str] = set()
    for s in valid:
        deduped = []
        for q in s["queries"][:_MAX_QUERIES_PER_SUBTOPIC]:
            if isinstance(q, str) and q not in seen_queries:
                seen_queries.add(q)
                deduped.append(q)
        s["queries"] = deduped

    # Drop subtopics with no queries after dedup
    valid = [s for s in valid if s["queries"]]

    return valid


def _execute_search(query: str, debug: bool = False) -> list[dict]:
    """Run a web search query. Returns list of {title, url, snippet} dicts."""
    results = web_search_solo(query, max_results=8)
    if not results:
        results = _perplexity_search(query, debug=debug)
    return results or []


def _extract_urls(results: list[dict], max_urls: int = 2) -> list[str]:
    """Extract top URLs from search results."""
    urls = []
    for r in results:
        url = r.get("url", "")
        if url.startswith("http") and url not in urls:
            urls.append(url)
            if len(urls) >= max_urls:
                break
    return urls


def _execute_fetch(url: str) -> str:
    """Fetch and extract text from a URL. Returns page text or empty string."""
    result = fetch_page(url)
    return result or ""


class RLMResearchAgent:
    """RLM-style research agent with decompose -> execute -> synthesize loop."""

    def __init__(
        self,
        ollama_url: str,
        model: str,
        root_model: str | None = None,
        decompose_url: str | None = None,
        decompose_model: str = "qwen3.5:9b",
        synthesis_url: str | None = None,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.root_model = root_model
        self.decompose_url = decompose_url or ollama_url
        self.decompose_model = decompose_model
        self.synthesis_url = synthesis_url or ollama_url
        self.debug_log: list[str] = []

    def _log(self, msg: str):
        self.debug_log.append(msg)
        print(f"   🧬 {msg}")

    def decompose(self, topic: str) -> list[dict]:
        """Wrapper around decompose_topic(), logs result."""
        subtopics = decompose_topic(
            topic, self.decompose_url, self.decompose_model
        )
        if subtopics:
            total_queries = sum(len(s["queries"]) for s in subtopics)
            self._log(
                f"decomposed: {len(subtopics)} subtopics, {total_queries} queries"
            )
            for s in subtopics:
                self._log(
                    f"  {s['subtopic']} (depth_hint={s['depth_hint']}): {s['queries']}"
                )
        else:
            self._log("decomposition failed — will fall back to standard agent")
        return subtopics

    def _research_subtopic(
        self, subtopic: dict, debug: bool = False
    ) -> dict:
        """Research a single subtopic by running its queries and fetching pages.

        Returns dict with keys: subtopic, depth_hint, queries_run,
        pages_fetched, sources, raw_content.
        """
        name = subtopic["subtopic"]
        queries = subtopic["queries"]
        depth_hint = subtopic.get("depth_hint", 2)

        all_content: list[str] = []
        sources: list[str] = []
        search_count = 0
        fetch_count = 0

        for query in queries:
            results = _execute_search(query, debug=debug)
            search_count += 1

            urls = _extract_urls(results, max_urls=2)

            # Also keep snippets as fallback content
            for r in results:
                snippet = r.get("snippet", "")
                if snippet:
                    url = r.get("url", "")
                    all_content.append(f"[snippet from {url}] {snippet}")

            for url in urls:
                page_text = _execute_fetch(url)
                if page_text:
                    all_content.append(page_text)
                    sources.append(url)
                    fetch_count += 1

        # Cap raw content per subtopic
        combined = "\n\n".join(all_content)
        words = combined.split()
        if len(words) > _CONTENT_CAP_PER_SUBTOPIC:
            combined = " ".join(words[:_CONTENT_CAP_PER_SUBTOPIC]) + "\n[truncated]"

        return {
            "subtopic": name,
            "depth_hint": depth_hint,
            "queries_run": queries,
            "searches": search_count,
            "pages_fetched": fetch_count,
            "sources": sources,
            "raw_content": combined,
        }

    def _synthesize(
        self, topic: str, findings: list[dict], reference_data: str = "",
        context: str = "",
    ) -> str:
        """Call the root model to compile subtopic findings into a research bundle."""
        synth_model = self.root_model or self.model
        synth_url = self.synthesis_url

        # Build context from all subtopic findings
        findings_text = []
        for f in findings:
            section = f"### {f['subtopic']} (depth_hint={f['depth_hint']})\n"
            section += f"Queries: {', '.join(f['queries_run'])}\n"
            section += f"Sources: {', '.join(f['sources']) if f['sources'] else 'none'}\n\n"
            section += f['raw_content']
            findings_text.append(section)

        all_findings = "\n\n---\n\n".join(findings_text)

        # Cap total context to avoid blowing the model window
        finding_words = all_findings.split()
        if len(finding_words) > 12000:
            all_findings = " ".join(finding_words[:12000]) + "\n[findings truncated]"

        system_prompt = (
            "You are a research analyst for InsiderLLM.com. Your task is to "
            "compile raw research findings from multiple subtopics into a "
            "single structured research bundle.\n\n"
            "Output MUST include these sections:\n"
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
            "Rules:\n"
            "- Include source URLs for every claim\n"
            "- Be comprehensive — include ALL data from ALL subtopics\n"
            "- Do NOT make up data — only use what's in the findings below\n"
            "- Deduplicate overlapping facts across subtopics\n"
            "- Flag conflicting data from different sources"
        )

        user_parts = [f"Research topic: {topic}"]
        if reference_data:
            user_parts.append(f"\n## Reference Data\n{reference_data}")
        if context:
            user_parts.append(f"\n## Author Context\n{context}")
        user_parts.append(f"\n## Raw Research Findings\n\n{all_findings}")
        user_parts.append(
            "\nCompile these findings into the structured research bundle format."
        )

        try:
            resp = httpx.post(
                f"{synth_url}/api/chat",
                json={
                    "model": synth_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "\n".join(user_parts)},
                    ],
                    "stream": False,
                    "options": {
                        "num_ctx": 16384,
                        "temperature": 0.5,
                        "num_predict": 4096,
                    },
                    "think": False,
                },
                timeout=300.0,
            )
            resp.raise_for_status()
            output = resp.json().get("message", {}).get("content", "").strip()
            # Strip think tags
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
            return output
        except Exception as e:
            self._log(f"synthesis failed: {e}")
            return ""

    def run(
        self,
        topic: str,
        reference_data: str = "",
        context: str = "",
        workspace_dir: str = ".",
        step_name: str = "research",
        debug: bool = False,
        **kwargs,
    ) -> str:
        """Run the full RLM research loop: decompose -> research -> synthesize."""
        start = time.time()

        # Phase 1: Decompose
        subtopics = self.decompose(topic)
        if not subtopics:
            return ""  # caller falls back to ResearchAgent

        # Phase 2: Research each subtopic sequentially
        findings: list[dict] = []
        total = len(subtopics)
        total_searches = 0
        total_fetches = 0

        for idx, st in enumerate(subtopics, 1):
            self._log(
                f"subtopic {idx}/{total}: {st['subtopic']} — "
                f"{len(st['queries'])} queries, depth_hint={st['depth_hint']}"
            )
            result = self._research_subtopic(st, debug=debug)
            findings.append(result)
            total_searches += result["searches"]
            total_fetches += result["pages_fetched"]
            self._log(
                f"  done — {result['searches']} searches, "
                f"{result['pages_fetched']} pages fetched, "
                f"{len(result['sources'])} sources"
            )

        self._log(
            f"all subtopics complete: {total_searches} searches, "
            f"{total_fetches} pages fetched"
        )

        # Phase 3: Synthesize with root model
        synth_model = self.root_model or self.model
        self._log(f"compiling final bundle ({synth_model} synthesis)")
        output = self._synthesize(
            topic, findings,
            reference_data=reference_data,
            context=context,
        )

        if not output:
            self._log("synthesis returned empty — falling back")
            return ""

        duration = time.time() - start
        words = _word_count(output)
        self._log(f"done: {words} words, {total} subtopics, {duration:.0f}s")

        # Write debug file
        if debug:
            os.makedirs(workspace_dir, exist_ok=True)
            debug_path = os.path.join(workspace_dir, f"{step_name}-rlm-debug.md")
            with open(debug_path, "w") as f:
                f.write(f"# RLM Research Debug Log\n\n")
                f.write(f"- Topic: {topic}\n")
                f.write(f"- Model: {self.model}\n")
                f.write(f"- Root model: {self.root_model}\n")
                f.write(f"- Decompose: {self.decompose_model} @ {self.decompose_url}\n")
                f.write(f"- Synthesis: {synth_model} @ {self.synthesis_url}\n")
                f.write(f"- Subtopics: {total}\n")
                f.write(f"- Total searches: {total_searches}\n")
                f.write(f"- Total pages fetched: {total_fetches}\n")
                f.write(f"- Duration: {duration:.0f}s\n")
                f.write(f"- Output words: {words}\n\n")
                f.write("## Subtopic Findings\n\n")
                for finding in findings:
                    f.write(f"### {finding['subtopic']}\n")
                    f.write(f"- Queries: {finding['queries_run']}\n")
                    f.write(f"- Searches: {finding['searches']}\n")
                    f.write(f"- Pages fetched: {finding['pages_fetched']}\n")
                    f.write(f"- Sources: {finding['sources']}\n")
                    f.write(f"- Content words: {_word_count(finding['raw_content'])}\n\n")
                f.write("## Agent Log\n\n")
                for line in self.debug_log:
                    f.write(f"- {line}\n")
            self._log(f"debug log: {debug_path}")

        return output
