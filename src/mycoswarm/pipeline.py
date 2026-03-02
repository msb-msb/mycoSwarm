"""mycoSwarm pipeline runner — sequential multi-agent workflows.

Each pipeline is a YAML file defining ordered steps. Each step's output
feeds the next as input context. Steps can optionally use web_search
and rag_search tools to gather context before inference.
"""

import json
import os
import re
import sys
import time
import uuid

import httpx
import yaml

from mycoswarm.hardware import detect_all


def load_pipeline(yaml_path: str) -> dict:
    """Parse and validate a pipeline YAML file.

    Returns the pipeline dict with keys: name, description, steps.
    Each step has: name, description, model, node_affinity, system_prompt, tools.
    """
    with open(yaml_path) as f:
        pipeline = yaml.safe_load(f)

    if not pipeline or not isinstance(pipeline, dict):
        print(f"❌ Invalid pipeline file: {yaml_path}")
        sys.exit(1)

    name = pipeline.get("name")
    if not name:
        print(f"❌ Pipeline missing 'name' field: {yaml_path}")
        sys.exit(1)

    steps = pipeline.get("steps")
    if not steps or not isinstance(steps, list):
        print(f"❌ Pipeline missing 'steps' list: {yaml_path}")
        sys.exit(1)

    for i, step in enumerate(steps):
        if not step.get("name"):
            print(f"❌ Step {i + 1} missing 'name'")
            sys.exit(1)
        if not step.get("system_prompt"):
            print(f"❌ Step '{step['name']}' missing 'system_prompt'")
            sys.exit(1)
        step.setdefault("model", None)
        step.setdefault("task_type", None)
        step.setdefault("node_affinity", "any")
        step.setdefault("tools", [])
        step.setdefault("depends_on", [])
        step.setdefault("description", "")

    return pipeline


def _get_daemon_url(port: int = 7890) -> str | None:
    """Return daemon URL if reachable, else None."""
    from mycoswarm.solo import check_daemon

    if check_daemon(port):
        profile = detect_all()
        ip = profile.lan_ip or "localhost"
        return f"http://{ip}:{port}"
    return None


def _swarm_headers() -> dict:
    """Load swarm auth headers."""
    try:
        from mycoswarm.auth import load_token, get_auth_header
        token = load_token()
        return get_auth_header(token) if token else {}
    except Exception:
        return {}


def _resolve_node_name(daemon_url: str, node_id: str) -> str:
    """Map a node_id to a hostname via /status and /peers."""
    try:
        with httpx.Client(headers=_swarm_headers(), timeout=3) as client:
            status = client.get(f"{daemon_url}/status").json()
            if status.get("node_id") == node_id:
                return status.get("hostname", node_id)
            for p in client.get(f"{daemon_url}/peers").json():
                if p.get("node_id") == node_id:
                    return p.get("hostname", node_id)
    except Exception:
        pass
    return node_id


def _submit_and_poll(url: str, task_payload: dict, timeout: int = 300) -> dict | None:
    """Submit a task to the daemon and poll until completion.

    The POST may block while the daemon routes to a remote node that
    needs to cold-start a model, so we use a generous read timeout
    derived from the caller's ``timeout`` while keeping a short
    connect timeout.  The poll loop uses a fixed 5 s timeout since
    each GET is a lightweight status check.
    """
    task_id = task_payload["task_id"]
    submit_timeout = httpx.Timeout(timeout, connect=10.0)
    try:
        with httpx.Client(headers=_swarm_headers(), timeout=submit_timeout) as client:
            resp = client.post(f"{url}/task", json=task_payload)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
        return None

    start = time.time()
    with httpx.Client(headers=_swarm_headers(), timeout=5) as client:
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                result_resp = client.get(f"{url}/task/{task_id}")
                data = result_resp.json()
                if data.get("status") in ("completed", "failed"):
                    return data
            except Exception:
                pass
    return None


def _stream_response(url: str, task_id: str, timeout: int = 300) -> tuple[str, dict]:
    """Consume SSE stream from daemon. Returns (full_text, metrics)."""
    tokens: list[str] = []
    metrics: dict = {}

    try:
        with httpx.Client(headers=_swarm_headers(), timeout=httpx.Timeout(5.0, read=timeout)) as client:
            with client.stream("GET", f"{url}/task/{task_id}/stream") as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if event.get("error"):
                        print(f"\n❌ {event['error']}")
                        return "".join(tokens), metrics

                    token = event.get("token", "")
                    if token and not event.get("done"):
                        print(token, end="", flush=True)
                        tokens.append(token)

                    if event.get("done"):
                        metrics = {
                            k: event[k]
                            for k in ("model", "tokens_per_second", "duration_seconds", "node_id")
                            if k in event
                        }
    except (httpx.ConnectError, httpx.ReadTimeout):
        pass

    return "".join(tokens), metrics


def _extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    try:
        host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


# Priority scoring for page fetching — higher = fetch first
_FETCH_PRIORITY = {
    "techpowerup.com": 3, "tomshardware.com": 3, "anandtech.com": 3,
    "pugetsystems.com": 3, "servethehome.com": 3,
    "insiderllm.com": 3, "rtings.com": 2,
    "reddit.com": 2, "overclock.net": 2,
}
_SKIP_FETCH = {"pinterest.com", "quora.com", "youtube.com", "facebook.com", "twitter.com", "x.com"}


def _do_web_search(
    topic: str, daemon_url: str | None, debug: bool = False,
    queries: list[str] | None = None,
) -> tuple[str, dict]:
    """Run wide web search and return (context_string, stats).

    Generates 20+ diverse queries (via LLM or templates) and dispatches
    them across the swarm in parallel. Deduplicates results by URL,
    fetches top 8 pages prioritized by source quality.

    If queries is provided, uses those directly instead of generating.
    """
    results: list[dict] = []
    pages: list[str] = []
    query_counts: list[tuple[str, int]] = []
    raw_count = 0
    stats: dict = {
        "num_queries": 0, "raw_count": 0, "result_count": 0,
        "pages_fetched": 0, "top_sources": [], "query_counts": [],
    }

    if daemon_url:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        variants = queries if queries is not None else _generate_search_queries(topic, n=20, debug=debug, daemon_url=daemon_url)
        stats["num_queries"] = len(variants)

        def _search_variant(query: str) -> list[dict]:
            task_id = f"ws-pipe-{uuid.uuid4().hex[:8]}"
            payload = {
                "task_id": task_id,
                "task_type": "web_search",
                "payload": {"query": query, "max_results": 10},
                "source_node": "pipeline",
                "priority": 7,
                "timeout_seconds": 60,
            }
            data = _submit_and_poll(daemon_url, payload, timeout=60)
            if data and data.get("status") == "completed":
                return data.get("result", {}).get("results", [])
            return []

        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            futures = {pool.submit(_search_variant, v): v for v in variants}
            for future in as_completed(futures, timeout=60):
                try:
                    variant_results = future.result()
                    query_counts.append((futures[future], len(variant_results)))
                    results.extend(variant_results)
                except Exception:
                    pass

        raw_count = len(results)

        # Dedup by URL
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                deduped.append(r)
        results = deduped

        # Fetch top 8 pages, prioritized by source quality
        scored: list[tuple[str, int]] = []
        for r in results:
            url_str = r.get("url", "")
            domain = _extract_domain(url_str)
            if any(s in domain for s in _SKIP_FETCH):
                continue
            score = 1
            for d, s in _FETCH_PRIORITY.items():
                if d in domain:
                    score = s
                    break
            scored.append((url_str, score))
        scored.sort(key=lambda x: -x[1])
        candidates = [url for url, _ in scored[:8]]

        def _fetch(page_url: str) -> str | None:
            try:
                with httpx.Client(timeout=10, follow_redirects=True) as client:
                    resp = client.get(page_url)
                    if resp.status_code != 200:
                        return None
                    if "text/html" not in resp.headers.get("content-type", ""):
                        return None
                    text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    words = text.split()
                    if len(words) > 2000:
                        text = " ".join(words[:2000]) + " [truncated]"
                    return f"[FULL PAGE: {page_url}]\n{text}"
            except Exception:
                return None

        if candidates:
            with ThreadPoolExecutor(max_workers=min(len(candidates), 6)) as pool:
                for future in as_completed(
                    [pool.submit(_fetch, u) for u in candidates], timeout=20
                ):
                    try:
                        text = future.result()
                        if text and len(text) > 200:
                            pages.append(text)
                    except Exception:
                        pass
    else:
        from mycoswarm.solo import web_search_solo
        variants = queries if queries is not None else _generate_search_queries(topic, n=10, debug=debug, daemon_url=None)
        stats["num_queries"] = len(variants)
        for v in variants:
            vr = web_search_solo(v, max_results=10)
            query_counts.append((v, len(vr)))
            results.extend(vr)

        raw_count = len(results)

        # Dedup solo results by URL
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                deduped.append(r)
        results = deduped

    # Extract top source domains
    domain_counts: dict[str, int] = {}
    for r in results:
        d = _extract_domain(r.get("url", ""))
        if d:
            domain_counts[d] = domain_counts.get(d, 0) + 1
    top_sources = sorted(domain_counts, key=lambda d: -domain_counts[d])[:5]

    # Build stats
    stats["raw_count"] = raw_count
    stats["result_count"] = len(results)
    stats["pages_fetched"] = len(pages)
    stats["top_sources"] = top_sources
    stats["query_counts"] = query_counts

    if not results and not pages:
        return "", stats

    parts: list[str] = []
    if results:
        snippet_lines = []
        for i, r in enumerate(results[:20], 1):
            snippet_lines.append(
                f"[{i}] {r.get('title', '')}\n    {r.get('url', '')}\n    {r.get('snippet', '')}"
            )
        parts.append("## WEB SEARCH RESULTS\n" + "\n\n".join(snippet_lines))

    if pages:
        parts.append("## FETCHED PAGE CONTENT\n" + "\n\n---\n\n".join(pages))

    return "\n\n".join(parts), stats


def _do_rag_search(
    topic: str, debug: bool = False,
) -> tuple[str, dict]:
    """Run RAG search and return (context_string, stats).

    Stats: {"doc_hits": int, "session_hits": int, "procedure_hits": int,
            "top_sources": list[str]}
    """
    stats: dict = {"doc_hits": 0, "session_hits": 0, "procedure_hits": 0, "top_sources": []}

    try:
        from mycoswarm.library import search_all
    except ImportError:
        if debug:
            print("   ⚠️  RAG not available (chromadb not installed)")
        return "", stats

    try:
        doc_hits, session_hits, procedure_hits = search_all(
            topic, n_results=5,
        )
    except Exception as e:
        if debug:
            print(f"   ⚠️  RAG search failed: {e}")
        return "", stats

    stats["doc_hits"] = len(doc_hits)
    stats["session_hits"] = len(session_hits)
    stats["procedure_hits"] = len(procedure_hits)
    stats["top_sources"] = list(dict.fromkeys(
        h.get("source", "unknown") for h in doc_hits[:5]
    ))

    parts: list[str] = []

    if doc_hits:
        doc_lines = []
        for i, h in enumerate(doc_hits, 1):
            source = h.get("source", "unknown")
            text = h.get("text", h.get("document", ""))
            doc_lines.append(f"[D{i}] ({source})\n{text}")
        parts.append("## DOCUMENT CONTEXT\n" + "\n\n".join(doc_lines))

    if session_hits:
        sess_lines = []
        for i, h in enumerate(session_hits, 1):
            text = h.get("text", h.get("document", ""))
            sess_lines.append(f"[S{i}]\n{text}")
        parts.append("## SESSION CONTEXT\n" + "\n\n".join(sess_lines))

    if procedure_hits:
        proc_lines = []
        for i, h in enumerate(procedure_hits, 1):
            text = h.get("text", h.get("document", ""))
            proc_lines.append(f"[P{i}]\n{text}")
        parts.append("## PROCEDURAL CONTEXT\n" + "\n\n".join(proc_lines))

    return "\n\n".join(parts), stats


def _run_inference(
    system_prompt: str,
    user_content: str,
    model: str,
    daemon_url: str | None,
    timeout: int = 300,
) -> tuple[str, dict]:
    """Run a single inference call. Returns (output_text, metrics).

    Uses daemon streaming if available, falls back to solo chat_stream.
    Metrics include: model, tokens_per_second, duration_seconds, node_id.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if daemon_url:
        task_id = f"pipe-{uuid.uuid4().hex[:8]}"
        payload = {
            "task_id": task_id,
            "task_type": "inference",
            "payload": {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "num_ctx": 16384,
                "max_tokens": 4096,
            },
            "source_node": "pipeline",
            "priority": 5,
            "timeout_seconds": timeout,
        }
        try:
            with httpx.Client(headers=_swarm_headers(), timeout=5) as client:
                resp = client.post(f"{daemon_url}/task", json=payload)
                resp.raise_for_status()
                submit_data = resp.json()
                # Capture routing info (e.g. "Routed to Miu")
                routed_msg = submit_data.get("message", "")
                if "Routed to" in routed_msg:
                    node_hint = routed_msg.replace("Routed to ", "")
                else:
                    node_hint = ""
        except httpx.ConnectError:
            daemon_url = None
            node_hint = ""

        if daemon_url:
            # Stream from target node directly when routed remotely.
            # The local daemon doesn't create a stream queue for remote
            # tasks — the stream lives on the target node.
            target_ip = submit_data.get("target_ip")
            target_port = submit_data.get("target_port")
            if target_ip and target_port:
                stream_url = f"http://{target_ip}:{target_port}"
            else:
                stream_url = daemon_url

            text, metrics = _stream_response(stream_url, task_id, timeout=timeout)

            # If streaming returned nothing (e.g. stream finished before
            # we connected), poll the daemon for the stored result.
            if not text:
                poll_start = time.time()
                with httpx.Client(headers=_swarm_headers(), timeout=5) as client:
                    while time.time() - poll_start < timeout:
                        time.sleep(1.0)
                        try:
                            r = client.get(f"{daemon_url}/task/{task_id}")
                            data = r.json()
                            if data.get("status") == "completed":
                                text = data.get("result", {}).get("response", "")
                                metrics = {
                                    "model": data.get("result", {}).get("model", model),
                                    "tokens_per_second": data.get("result", {}).get("tokens_per_second", 0),
                                    "duration_seconds": data.get("duration_seconds", 0),
                                    "node_id": data.get("node_id", ""),
                                }
                                break
                            elif data.get("status") == "failed":
                                break
                        except Exception:
                            pass

            # Resolve node name
            if not metrics.get("node_name"):
                nid = metrics.get("node_id", "")
                if node_hint:
                    metrics["node_name"] = node_hint
                elif nid and daemon_url:
                    metrics["node_name"] = _resolve_node_name(daemon_url, nid)
                else:
                    metrics["node_name"] = "local"
            return text, metrics

    # Solo fallback
    from mycoswarm.solo import chat_stream
    text, metrics = chat_stream(messages, model)
    metrics["node_name"] = "local"
    return text, metrics


def _build_solo_routing(steps: list[dict]) -> list:
    """Build routing for solo mode (no daemon) using local Ollama models."""
    from mycoswarm.capabilities import TASK_MODEL_MAP
    from mycoswarm.router import RouteResult
    from mycoswarm.solo import check_ollama

    _, solo_models = check_ollama()
    solo_set = set(solo_models)

    results: list[RouteResult] = []
    for step in steps:
        if step.get("model"):
            m = step["model"]
        else:
            task_type = step.get("task_type", "general")
            task_config = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["general"])
            m = None
            for preferred in task_config["prefer_models"]:
                if preferred in solo_set:
                    m = preferred
                    break
            if m is None:
                for sm in solo_set:
                    if any(s in sm for s in ("27b", "32b", "14b")):
                        m = sm
                        break
            if m is None and solo_set:
                m = next(iter(solo_set))
            if m is None:
                print("❌ No models available.")
                sys.exit(1)

        results.append(RouteResult(
            model=m,
            node_hostname="local",
            node_ip="127.0.0.1",
            node_port=0,
            is_local=True,
            reason="Solo mode",
            task_type=step.get("task_type", "general"),
        ))
    return results


_GARBAGE_RE = re.compile(r'<unused\d+>|<\|[a-z_]+\|>|<0x[0-9A-Fa-f]+>')


def _word_count(text: str) -> int:
    """Rough word count."""
    return len(text.split())


def _clean_output(text: str) -> str:
    """Strip garbage tokens from model output."""
    return _GARBAGE_RE.sub('', text).strip()


_URL_RE = re.compile(r'https?://\S+')


def _filter_unsourced_bullets(text: str, debug: bool = False) -> str:
    """Remove bullet points that don't contain a URL."""
    lines = text.split("\n")
    filtered = []
    stripped = 0
    for line in lines:
        # Keep non-bullet lines (headers, blank lines)
        if not line.strip().startswith("-"):
            filtered.append(line)
            continue
        # Bullet must contain a URL
        if _URL_RE.search(line):
            filtered.append(line)
        else:
            stripped += 1
    if debug and stripped:
        print(f"   🐛 stripped {stripped} unsourced bullet(s)")
    return "\n".join(filtered)


def _truncate_to_word_limit(text: str, max_words: int, debug: bool = False) -> str:
    """Truncate text at the last complete bullet point before the word limit."""
    lines = text.split("\n")
    truncated = []
    count = 0
    for line in lines:
        line_words = len(line.split())
        if count + line_words > max_words:
            break
        truncated.append(line)
        count += line_words
    result = "\n".join(truncated)
    if debug:
        print(f"   🐛 truncated to {count} words (max: {max_words})")
    return result


_THINK_RE = re.compile(r'<think>.*?</think>', flags=re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from deepseek-r1 output."""
    return _THINK_RE.sub('', text).strip()


_OLLAMA_BASE = "http://localhost:11434"

_QUERY_GEN_PROMPT = (
    "Generate {n} diverse web search queries to thoroughly research "
    "this topic: {topic}\n\n"
    "Cover ALL of these angles:\n"
    "- General overview queries (2-3)\n"
    "- Specific product comparisons (3-4): \"X vs Y\"\n"
    "- Price/value queries (2-3): used market, deals, price history\n"
    "- Technical benchmark queries (3-4): specific metrics, specs\n"
    "- Community/forum queries (2-3): reddit, forum discussions\n"
    "- Contrarian/alternative queries (2-3): underrated options, alternatives\n"
    "- Software ecosystem queries (2): compatibility, driver support\n"
    "- Recent news queries (2): latest releases, price drops\n\n"
    "Rules:\n"
    "- Each query must be 3-8 words\n"
    "- No two queries should return the same search results\n"
    "- Include specific product names where relevant\n"
    "- Output ONLY the queries, one per line, no numbering, no explanation"
)

_QUERY_GEN_MODELS = ("gemma3:4b", "gemma3:12b", "llama3.2:3b", "gemma3:1b")

_MUST_HAVE_QUERIES = [
    "RTX 3090 used price 2026",
    "GPU tokens per second LLM benchmark",
    "r/LocalLLaMA best GPU recommendation",
    "RTX 4060 Ti vs RTX 3060 AI inference",
    "cheapest 24GB VRAM GPU",
    "best GPU for ollama local LLM",
    "tom's hardware GPU benchmark AI",
    "RTX 5060 Ti review local AI",
    "used GPU for AI eBay prices",
    "GPU power consumption AI inference watts",
    # RTX 3090 specific
    "RTX 3090 tokens per second ollama benchmark",
    "RTX 3090 vs RTX 5060 Ti AI inference",
    "RTX 3090 24GB used price local AI",
    "RTX 3090 power consumption TDP watts",
    # RTX 3060 specific
    "RTX 3060 12GB local LLM performance",
    "RTX 3060 used price 2026 budget AI",
    "RTX 3060 tokens per second 7B 13B model",
]


def _has_word_overlap(query: str, existing: list[str], threshold: float = 0.6) -> bool:
    """Check if query has >threshold word overlap with any existing query."""
    q_words = set(query.lower().split())
    if not q_words:
        return False
    for e in existing:
        e_words = set(e.lower().split())
        overlap = len(q_words & e_words)
        if overlap / len(q_words) > threshold:
            return True
    return False


def _append_must_haves(queries: list[str]) -> tuple[int, int]:
    """Append must-have queries, skipping fuzzy duplicates.

    Returns (added_count, deduped_count).
    """
    added = 0
    deduped = 0
    for mq in _MUST_HAVE_QUERIES:
        if _has_word_overlap(mq, queries):
            deduped += 1
        else:
            queries.append(mq)
            added += 1
    return added, deduped


def _parse_query_lines(raw: str) -> list[str]:
    """Parse LLM output into individual search queries."""
    queries: list[str] = []
    seen: set[str] = set()
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip('"\'').strip()
        lower = line.lower()
        if 3 <= len(line.split()) <= 12 and lower not in seen:
            queries.append(line)
            seen.add(lower)
    return queries


def _query_gen_via_swarm(
    prompt: str,
    daemon_url: str,
    temperature: float = 0.8,
    max_tokens: int = 600,
) -> tuple[str, str | None, str]:
    """Run query gen prompt through the swarm via Router.

    Returns (raw_text, model_used, node_hostname).
    Routes as inference task — the scoring flip naturally
    prefers specialist nodes for small models.
    """
    from mycoswarm.capabilities import TASK_MODEL_MAP
    from mycoswarm.router import Router

    try:
        from mycoswarm.auth import load_token
        token = load_token()
    except Exception:
        token = None

    try:
        router = Router.from_daemon(daemon_url, swarm_token=token)
    except Exception:
        return "", None, ""

    # Collect all models across swarm
    all_models: set[str] = set(router.identity.available_models)
    for p in router._peers:
        all_models.update(p.available_models)

    # Pick best model from query_gen preferences (substring match)
    prefer = TASK_MODEL_MAP.get("query_gen", {}).get(
        "prefer_models", list(_QUERY_GEN_MODELS),
    )
    model = None
    for pattern in prefer:
        for avail in all_models:
            if pattern in avail:
                model = avail
                break
        if model:
            break

    if not model:
        return "", None, ""

    # Resolve which node will handle it (for debug output)
    result = router.resolve_sync("inference", model=model)
    node_hostname = result.node_hostname if result else "unknown"

    # Submit inference task via daemon
    task_id = f"querygen-{uuid.uuid4().hex[:8]}"
    payload = {
        "task_id": task_id,
        "task_type": "inference",
        "payload": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_ctx": 4096,
        },
        "source_node": "pipeline",
        "priority": 7,
        "timeout_seconds": 60,
    }

    data = _submit_and_poll(daemon_url, payload, timeout=60)
    if data and data.get("status") == "completed":
        text = data.get("result", {}).get("response", "")
        return text, model, node_hostname
    return "", model, node_hostname


def _generate_search_queries(
    topic: str, n: int = 20, debug: bool = False,
    daemon_url: str | None = None,
) -> list[str]:
    """Generate diverse search queries using LLM + must-have queries + templates.

    1. LLM (gemma3:4b preferred) generates n diverse queries
    2. Must-have queries appended with fuzzy dedup (>60% word overlap = skip)
    3. Template fallback pads to n if LLM unavailable or returned too few
    Final count is typically 25-30 queries.
    """
    queries, model_used, node_used = _llm_generate_queries(topic, n, daemon_url=daemon_url)
    llm_count = len(queries)

    if llm_count < n:
        queries = _pad_with_templates(topic, queries, n)

    if debug:
        if model_used:
            label = f"{model_used} ({node_used})" if node_used else model_used
            print(f"   🐛 query gen model: {label}")
            print(f"   🐛 LLM generated {llm_count} queries")
        else:
            print(f"   🐛 LLM unavailable, using template queries")

    # Always append must-have queries (fuzzy dedup against existing)
    added, deduped = _append_must_haves(queries)
    if debug:
        print(f"   🐛 added {added} must-have queries ({deduped} deduped)")
        print(f"   🐛 search queries: {len(queries)}")

    return queries


def _llm_generate_queries(
    topic: str, n: int, daemon_url: str | None = None,
) -> tuple[list[str], str | None, str]:
    """Generate queries via LLM. Routes through swarm when daemon available.

    Returns (queries, model_name, node_hostname).
    model_name is None if unavailable. node_hostname may be empty.
    """
    prompt = _QUERY_GEN_PROMPT.format(n=n, topic=topic)

    # Swarm path: route through daemon (graceful fallback on any failure)
    if daemon_url:
        try:
            raw, model, node = _query_gen_via_swarm(
                prompt, daemon_url, temperature=0.8, max_tokens=600,
            )
            if raw:
                return _parse_query_lines(raw), model, node
        except Exception:
            pass  # fall through to solo

    # Solo fallback: call localhost Ollama directly
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{_OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return [], None, ""

    model = None
    for pattern in _QUERY_GEN_MODELS:
        for m in available:
            if pattern in m:
                model = m
                break
        if model:
            break

    if not model:
        return [], None, ""

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=30.0)) as client:
            resp = client.post(
                f"{_OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.8, "num_predict": 600},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
    except Exception:
        return [], None, ""

    return _parse_query_lines(raw), model, "local"


_FILLER_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "for", "to", "and",
    "or", "how", "does", "do", "can", "will", "are", "was", "were",
    "been", "being", "have", "has", "had", "about", "with", "from",
    "this", "that", "these", "those", "it", "its", "my", "your",
    "our", "their", "best", "most", "top",
}


def _pad_with_templates(
    topic: str, existing: list[str], n: int,
) -> list[str]:
    """Pad query list with template-based fallbacks to guarantee n queries."""
    from datetime import datetime
    year = str(datetime.now().year)

    keywords = [
        w for w in topic.split()
        if w.lower().strip("?.,!") not in _FILLER_WORDS
    ]
    kw = " ".join(keywords) if keywords else topic

    templates = [
        topic,
        f"best used {kw} {year}",
        f"{kw} vs alternatives comparison",
        f"{kw} price used market {year}",
        f"reddit {kw} {year}",
        f"reddit best {kw} recommendation",
        f"{kw} benchmark tokens per second",
        f"{kw} VRAM requirements performance",
        f"{kw} power consumption wattage",
        f"underrated {kw} alternatives {year}",
        f"{kw} driver support compatibility",
        f"latest {kw} news {year}",
        f"{kw} price drop deals {year}",
        f"{kw} review hands on {year}",
        f"forum discussion {kw} experience",
        f"{kw} real world performance test",
        f"budget {kw} build guide {year}",
        f"{kw} long term reliability issues",
        f"{kw} buying guide {year}",
        f"{kw} comparison chart specs",
    ]

    existing_lower = {q.lower() for q in existing}
    for t in templates:
        if t.lower() not in existing_lower and len(existing) < n:
            existing.append(t)
            existing_lower.add(t.lower())

    return existing[:n]


def _extract_gaps(synth_text: str) -> str:
    """Extract the ## Gaps section from synthesizer output."""
    match = re.search(r'## Gaps\s*\n(.*?)(?=\n## |\Z)', synth_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


_GAP_QUERY_PROMPT = (
    "Convert each gap into 2-3 specific web search queries:\n\n"
    "Gaps:\n{gaps_text}\n\n"
    "Output ONLY search queries, one per line. Be specific — include "
    "GPU model names, metric names (tok/s, TDP, watts), and site "
    "names (reddit, techpowerup, tomshardware)."
)


def _generate_gap_queries(
    gaps_text: str, daemon_url: str | None = None, debug: bool = False,
) -> list[str]:
    """Generate targeted search queries from synthesizer gaps using LLM."""
    prompt = _GAP_QUERY_PROMPT.format(gaps_text=gaps_text)

    # Swarm path: route through daemon (graceful fallback on any failure)
    if daemon_url:
        try:
            raw, model, node = _query_gen_via_swarm(
                prompt, daemon_url, temperature=0.7, max_tokens=400,
            )
            if raw:
                if debug and model:
                    print(f"   🐛 gap query gen model: {model} ({node})")
                queries = _parse_query_lines(raw)
                if debug:
                    print(f"   🐛 gap queries: {len(queries)}")
                return queries
        except Exception:
            if debug:
                print("   🐛 gap query gen swarm failed, falling back to local")

    # Solo fallback: call localhost Ollama directly
    model = None
    try:
        with httpx.Client(timeout=3) as client:
            resp = client.get(f"{_OLLAMA_BASE}/api/tags")
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []

    for pattern in _QUERY_GEN_MODELS:
        for m in available:
            if pattern in m:
                model = m
                break
        if model:
            break

    if not model:
        return []

    if debug:
        print(f"   🐛 gap query gen model: {model}")

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=30.0)) as client:
            resp = client.post(
                f"{_OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.7, "num_predict": 400},
                    "stream": False,
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
    except Exception:
        return []

    queries = _parse_query_lines(raw)

    if debug:
        print(f"   🐛 gap queries: {len(queries)}")

    return queries


def _cap_context(web_ctx: str, rag_ctx: str, max_words: int = 4000) -> tuple[str, str]:
    """Cap combined context to max_words. Truncates web results first."""
    rag_words = _word_count(rag_ctx)
    web_words = _word_count(web_ctx)
    total = web_words + rag_words

    if total <= max_words:
        return web_ctx, rag_ctx

    # RAG is generally more relevant — keep it, truncate web
    web_budget = max(max_words - rag_words, max_words // 3)
    if web_words > web_budget:
        words = web_ctx.split()
        web_ctx = " ".join(words[:web_budget]) + "\n[web context truncated]"

    return web_ctx, rag_ctx


def run_pipeline(
    pipeline: dict,
    topic: str,
    workspace_dir: str,
    port: int = 7890,
    debug: bool = False,
) -> str | None:
    """Execute a pipeline sequentially. Returns path to final output file.

    Returns None if a step fails the minimum output gate.
    Each step reads the previous step's output (or the topic for step 1),
    optionally gathers tool context, runs inference, and writes output
    to workspace_dir/{step_name}.md.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    steps = pipeline["steps"]
    total = len(steps)
    daemon_url = _get_daemon_url(port)

    mode = "swarm" if daemon_url else "solo"
    print(f"🍄 Pipeline: {pipeline['name']}")
    print(f"   Mode: {mode} | Steps: {total}")
    print(f"   Topic: {topic}")
    print(f"   Workspace: {workspace_dir}")

    # Resolve routing for all steps upfront via unified Router
    from mycoswarm.router import Router

    route_results = None
    if daemon_url:
        try:
            from mycoswarm.auth import load_token
            token = load_token()
        except Exception:
            token = None
        try:
            router = Router.from_daemon(daemon_url, swarm_token=token)
            route_results = router.build_routing_table(steps)
        except Exception:
            pass  # fall back to solo

    if route_results is None:
        route_results = _build_solo_routing(steps)

    # Pre-warm: fire a quick /health check to each unique remote node
    # so TCP + TLS handshake is done before the first real request.
    _prewarm_nodes = set()
    for rr in route_results:
        if not rr.is_local and rr.node_ip:
            addr = f"http://{rr.node_ip}:{rr.node_port}"
            if addr not in _prewarm_nodes:
                _prewarm_nodes.add(addr)
                try:
                    httpx.get(f"{addr}/health", timeout=3)
                except Exception:
                    pass

    # Print routing table
    max_name = max(len(s["name"]) for s in steps)
    print(f"\n🧭 Routing:")
    for step, rr in zip(steps, route_results):
        name = step["name"].ljust(max_name)
        task_hint = f" [{step['task_type']}]" if step.get("task_type") else ""
        print(f"   {name} → {rr.model} ({rr.node_hostname}){task_hint}")
    print(f"{'─' * 60}")

    previous_output: str | None = None
    last_output_path = ""
    pipeline_start = time.time()

    for i, step in enumerate(steps):
        step_num = i + 1
        step_name = step["name"]
        model = route_results[i].model
        node_host = route_results[i].node_hostname
        print(f"\n📝 Step {step_num}/{total}: {step_name}")

        start = time.time()

        if debug:
            print(f"   🐛 model: {model}")
            tt = step.get("task_type", "none")
            print(f"   🐛 task_type: {tt}, node_affinity: {step.get('node_affinity', 'any')}")

        # --- Gather tool context ---
        tools = step.get("tools", [])
        context_parts: list[str] = []

        if "web_search" in tools:
            web_start = time.time()
            # Gap-filler: targeted search for missing data from synthesizer gaps
            gap_queries = None
            if step_name == "gap-filler":
                synth_path = os.path.join(workspace_dir, "synthesizer.md")
                if os.path.isfile(synth_path):
                    with open(synth_path) as f:
                        gaps_text = _extract_gaps(f.read())
                    if gaps_text:
                        gap_queries = _generate_gap_queries(gaps_text, daemon_url=daemon_url, debug=debug)
                        if debug:
                            print(f"   🐛 gap-fill: {len(gap_queries)} targeted queries from gaps")
            web_ctx, web_stats = _do_web_search(topic, daemon_url, debug=debug, queries=gap_queries)
            web_elapsed = time.time() - web_start
            if web_ctx:
                context_parts.append(web_ctx)
            nq = web_stats.get("num_queries", 0)
            raw = web_stats.get("raw_count", 0)
            rc = web_stats["result_count"]
            pf = web_stats["pages_fetched"]
            print(f"   🔍 Web: {nq} queries → {raw} results ({rc} unique), {pf} pages fetched ({web_elapsed:.1f}s)")
            top_src = web_stats.get("top_sources", [])
            if top_src:
                print(f"      Top sources: {', '.join(top_src)}")
            if debug:
                for q, c in web_stats.get("query_counts", []):
                    print(f"      q: \"{q}\" → {c} results")

        if "rag_search" in tools:
            rag_start = time.time()
            rag_ctx, rag_stats = _do_rag_search(topic, debug=debug)
            rag_elapsed = time.time() - rag_start
            if rag_ctx:
                context_parts.append(rag_ctx)
            dh = rag_stats["doc_hits"]
            sh = rag_stats["session_hits"]
            ph = rag_stats["procedure_hits"]
            rag_parts = []
            if dh:
                rag_parts.append(f"{dh} docs")
            if sh:
                rag_parts.append(f"{sh} sessions")
            if ph:
                rag_parts.append(f"{ph} procedures")
            rag_summary = ", ".join(rag_parts) if rag_parts else "no results"
            print(f"   📚 RAG: {rag_summary}", end="")
            if debug:
                print(f" ({rag_elapsed:.1f}s)")
                for src in rag_stats.get("top_sources", []):
                    print(f"      → {src}")
            else:
                print()

        # --- Cap context to avoid blowing the model's window ---
        if len(context_parts) == 2:
            context_parts[0], context_parts[1] = _cap_context(
                context_parts[0], context_parts[1],
            )
        elif len(context_parts) == 1 and _word_count(context_parts[0]) > 4000:
            words = context_parts[0].split()
            context_parts[0] = " ".join(words[:4000]) + "\n[context truncated]"

        # --- Build user input ---
        # depends_on: load multiple named step outputs
        if step.get("depends_on"):
            input_parts = [f"Topic: {topic}"]
            for dep_name in step["depends_on"]:
                dep_path = os.path.join(workspace_dir, f"{dep_name}.md")
                if os.path.isfile(dep_path):
                    with open(dep_path) as f:
                        dep_text = f.read()
                    input_parts.append(f"## {dep_name.upper()} OUTPUT\n{dep_text}")
                    if debug:
                        dw = _word_count(dep_text)
                        print(f"   🐛 depends_on: {dep_name}.md ({dw} words)")
            if context_parts:
                input_parts.append(
                    "--- RETRIEVED CONTEXT ---\n" + "\n\n".join(context_parts)
                )
        # Special case: editor gets research bundle + draft
        elif step_name == "editor":
            # Prefer synthesizer-v2 (gap-filled), fall back to synthesizer
            research_path = os.path.join(workspace_dir, "synthesizer-v2.md")
            if not os.path.isfile(research_path):
                research_path = os.path.join(workspace_dir, "synthesizer.md")
            writer_path = os.path.join(workspace_dir, "writer.md")
            if os.path.isfile(research_path) and os.path.isfile(writer_path):
                with open(research_path) as f:
                    research_text = f.read()
                with open(writer_path) as f:
                    writer_text = f.read()
                if debug:
                    rname = os.path.basename(research_path)
                    rw = _word_count(research_text)
                    ww = _word_count(writer_text)
                    print(f"   🐛 editor inputs: {rname} ({rw} words) + writer.md ({ww} words)")
                input_parts = [
                    f"Topic: {topic}",
                    "## RESEARCH BUNDLE (ground truth — use this to fact-check)\n"
                    + research_text,
                    "## DRAFT ARTICLE (review this)\n" + writer_text,
                ]
            else:
                input_parts = [f"Topic: {topic}"]
                if previous_output:
                    input_parts.append(previous_output)
        else:
            input_parts: list[str] = [f"Topic: {topic}"]

            if context_parts:
                input_parts.append(
                    "--- RETRIEVED CONTEXT ---\n" + "\n\n".join(context_parts)
                )

            if previous_output:
                input_parts.append(
                    f"--- OUTPUT FROM PREVIOUS STEP ---\n{previous_output}"
                )

        user_content = "\n\n".join(input_parts)
        input_words = _word_count(user_content)

        if input_words > 6000:
            print(f"   ⚠️  Large context: {input_words} words — may degrade output quality")

        if debug:
            print(f"   🐛 input: {input_words} words")

        # --- Run inference ---
        print(f"   🧠 Generating on {node_host} ({model})...", end="", flush=True)
        output_text, metrics = _run_inference(
            system_prompt=step["system_prompt"],
            user_content=user_content,
            model=model,
            daemon_url=daemon_url,
        )

        # Clean garbage tokens
        output_text = _clean_output(output_text)

        # Strip <think>...</think> blocks (deepseek-r1 chain-of-thought)
        think_contents = re.findall(r'<think>(.*?)</think>', output_text, flags=re.DOTALL)
        if think_contents:
            think_words = sum(_word_count(t) for t in think_contents)
            if debug:
                print(f"   🐛 reasoning: {think_words} words (stripped from output)")
        output_text = _strip_think_tags(output_text)

        # --- Post-generation filters ---
        # Source enforcement: strip unsourced bullets from gap-filler
        if step_name == "gap-filler":
            output_text = _filter_unsourced_bullets(output_text, debug=debug)

        # Hard output cap: truncate at last complete bullet before limit
        max_words = step.get("max_output_words")
        if max_words and _word_count(output_text) > max_words:
            output_text = _truncate_to_word_limit(output_text, max_words, debug=debug)

        duration = time.time() - start
        words = _word_count(output_text)
        tps = metrics.get("tokens_per_second", 0)
        actual_node = metrics.get("node_name", node_host)

        # --- Write output ---
        output_path = os.path.join(workspace_dir, f"{step_name}.md")
        with open(output_path, "w") as f:
            f.write(output_text)
        last_output_path = output_path

        print(f" done — {words} words, {duration:.0f}s")

        if debug:
            print(f"   🐛 node: {actual_node}")
            print(f"   🐛 input: {input_words} words → output: {words} words")
            print(f"   🐛 {tps:.1f} tok/s | {duration:.1f}s")
            print(f"   🐛 saved: {output_path}")
            preview = output_text[:200].replace("\n", "\\n")
            print(f"   🐛 output preview: \"{preview}...\"")

        # --- Minimum output gate ---
        min_words = step.get("min_output_words", 50)
        if words < min_words:
            print(f"   ❌ Step failed — output too short ({words} words, minimum {min_words})")
            print(f"   💡 Check the step's system prompt or increase context")
            print(f"   📄 Partial output saved to: {output_path}")
            return None  # halt pipeline

        previous_output = output_text

    # --- Summary ---
    total_time = time.time() - pipeline_start
    print(f"\n{'─' * 60}")
    print(f"🍄 Pipeline complete: {pipeline['name']}")
    print(f"   Steps: {total} | Total: {total_time:.0f}s")
    print(f"   Final output: {last_output_path}")
    print(f"   Workspace: {workspace_dir}")

    return last_output_path
