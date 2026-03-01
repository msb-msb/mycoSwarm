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
        step.setdefault("node_affinity", "any")
        step.setdefault("tools", [])
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
    """Submit a task to the daemon and poll until completion."""
    task_id = task_payload["task_id"]
    try:
        with httpx.Client(headers=_swarm_headers(), timeout=5) as client:
            resp = client.post(f"{url}/task", json=task_payload)
            resp.raise_for_status()
    except httpx.ConnectError:
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


def _do_web_search(
    topic: str, daemon_url: str | None, debug: bool = False,
) -> tuple[str, dict]:
    """Run web search and return (context_string, stats).

    Stats: {"result_count": int, "pages_fetched": int, "top_urls": list[str]}
    """
    results: list[dict] = []
    pages: list[str] = []
    stats: dict = {"result_count": 0, "pages_fetched": 0, "top_urls": []}

    if daemon_url:
        from mycoswarm.solo import generate_search_variants
        from concurrent.futures import ThreadPoolExecutor, as_completed

        variants = generate_search_variants(topic)

        def _search_variant(query: str) -> list[dict]:
            task_id = f"ws-pipe-{uuid.uuid4().hex[:8]}"
            payload = {
                "task_id": task_id,
                "task_type": "web_search",
                "payload": {"query": query, "max_results": 15},
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
            for future in as_completed(futures, timeout=45):
                try:
                    results.extend(future.result())
                except Exception:
                    pass

        # Dedup by URL
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                deduped.append(r)
        results = deduped

        # Fetch top 3 pages
        _SKIP = {"wikipedia.org", "reddit.com", "quora.com", "pinterest.com"}
        candidates = []
        for r in results:
            url_str = r.get("url", "")
            if not any(d in url_str for d in _SKIP):
                candidates.append(url_str)
            if len(candidates) >= 3:
                break

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
            with ThreadPoolExecutor(max_workers=3) as pool:
                for future in as_completed(
                    [pool.submit(_fetch, u) for u in candidates], timeout=15
                ):
                    try:
                        text = future.result()
                        if text and len(text) > 200:
                            pages.append(text)
                    except Exception:
                        pass
    else:
        from mycoswarm.solo import web_search_solo
        results = web_search_solo(topic, max_results=10)

    # Build stats
    stats["result_count"] = len(results)
    stats["pages_fetched"] = len(pages)
    stats["top_urls"] = [r.get("url", "") for r in results[:3]]

    if not results and not pages:
        return "", stats

    parts: list[str] = []
    if results:
        snippet_lines = []
        for i, r in enumerate(results[:10], 1):
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
            text, metrics = _stream_response(daemon_url, task_id, timeout=timeout)
            # Resolve node_id to hostname if we didn't get it from routing
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


def _discover_model(daemon_url: str | None, prefer: str | None = None) -> str:
    """Pick the best model from daemon or local Ollama."""
    if daemon_url:
        try:
            with httpx.Client(headers=_swarm_headers(), timeout=5) as client:
                resp = client.get(f"{daemon_url}/status")
                resp.raise_for_status()
                models = resp.json().get("ollama_models", [])

                if not models:
                    peers = client.get(f"{daemon_url}/peers").json()
                    for p in peers:
                        if p.get("available_models"):
                            models = p["available_models"]
                            break
        except Exception:
            models = []
    else:
        from mycoswarm.solo import check_ollama
        _, models = check_ollama()

    if prefer and prefer in models:
        return prefer

    # Prefer large models
    for m in models:
        if any(s in m for s in ("27b", "32b", "14b")):
            return m
    if models:
        return models[0]

    print("❌ No models available.")
    sys.exit(1)


_GARBAGE_RE = re.compile(r'<unused\d+>|<\|[a-z_]+\|>|<0x[0-9A-Fa-f]+>')


def _word_count(text: str) -> int:
    """Rough word count."""
    return len(text.split())


def _clean_output(text: str) -> str:
    """Strip garbage tokens from model output."""
    return _GARBAGE_RE.sub('', text).strip()


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
    print(f"{'─' * 60}")

    previous_output: str | None = None
    last_output_path = ""
    pipeline_start = time.time()

    for i, step in enumerate(steps):
        step_num = i + 1
        step_name = step["name"]
        print(f"\n📝 Step {step_num}/{total}: {step_name}")

        start = time.time()

        # Resolve model for this step
        model = _discover_model(daemon_url, prefer=step.get("model"))
        if debug:
            print(f"   🐛 model: {model}")
            print(f"   🐛 node_affinity: {step.get('node_affinity', 'any')}")

        # --- Gather tool context ---
        tools = step.get("tools", [])
        context_parts: list[str] = []

        if "web_search" in tools:
            web_start = time.time()
            web_ctx, web_stats = _do_web_search(topic, daemon_url, debug=debug)
            web_elapsed = time.time() - web_start
            if web_ctx:
                context_parts.append(web_ctx)
            rc = web_stats["result_count"]
            pf = web_stats["pages_fetched"]
            print(f"   🔍 Web: {rc} results, {pf} pages fetched", end="")
            if debug:
                print(f" ({web_elapsed:.1f}s)")
                for url in web_stats.get("top_urls", []):
                    print(f"      → {url}")
            else:
                print()

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
        # Special case: editor gets both research bundle and draft
        if step_name == "editor":
            research_path = os.path.join(workspace_dir, "researcher.md")
            writer_path = os.path.join(workspace_dir, "writer.md")
            if os.path.isfile(research_path) and os.path.isfile(writer_path):
                with open(research_path) as f:
                    research_text = f.read()
                with open(writer_path) as f:
                    writer_text = f.read()
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
        node_label = "local"
        if daemon_url:
            try:
                with httpx.Client(headers=_swarm_headers(), timeout=3) as client:
                    status = client.get(f"{daemon_url}/status").json()
                    node_label = status.get("hostname", "local")
            except Exception:
                pass

        print(f"   🧠 Generating on {node_label} ({model})...", end="", flush=True)
        output_text, metrics = _run_inference(
            system_prompt=step["system_prompt"],
            user_content=user_content,
            model=model,
            daemon_url=daemon_url,
        )

        # Clean garbage tokens
        output_text = _clean_output(output_text)

        duration = time.time() - start
        words = _word_count(output_text)
        tps = metrics.get("tokens_per_second", 0)
        actual_node = metrics.get("node_name", node_label)

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
