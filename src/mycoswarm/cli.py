"""mycoSwarm CLI.

Usage:
    mycoswarm chat                Interactive chat (works without daemon)
    mycoswarm chat --resume       Resume the most recent chat session
    mycoswarm chat --session NAME Resume a named session
    mycoswarm chat --list         List saved chat sessions
    mycoswarm ask "your prompt"   Send a prompt for inference (works without daemon)
    mycoswarm detect              Show this node's hardware and capabilities
    mycoswarm detect --json       Output as JSON
    mycoswarm identity            Show full node identity announcement
    mycoswarm daemon              Start the node daemon (announce + discover)
    mycoswarm daemon --port 7890  Start on a specific port
    mycoswarm daemon -v           Verbose logging
    mycoswarm dashboard           Start the web dashboard (http://localhost:8080)
    mycoswarm swarm               Show swarm status (query local daemon)
    mycoswarm ping                Ping all known peers
    mycoswarm search "query"      Search the web via the swarm
    mycoswarm research "query"    Search + synthesize (CPU search → GPU think)
    mycoswarm models              Show all models available across the swarm
    mycoswarm plugins             List installed plugins and their status
    mycoswarm memory              Show stored facts about the user
    mycoswarm memory --add "..."  Remember a fact
    mycoswarm memory --forget N   Forget a fact by ID
    mycoswarm library ingest      Ingest documents into the local library
    mycoswarm library search      Search indexed documents
    mycoswarm library list        List indexed documents
    mycoswarm library remove      Remove a document from the index
    mycoswarm rag "question"      Ask a question with document context (RAG)
"""

import argparse
import sys

import httpx

from mycoswarm.hardware import detect_all
from mycoswarm.capabilities import classify_node
from mycoswarm.node import build_identity


def cmd_detect(args):
    """Detect hardware and classify capabilities."""
    profile = detect_all()
    caps = classify_node(profile)

    if args.json:
        identity = build_identity(profile, caps)
        print(identity.to_json())
        return

    print("🍄 mycoSwarm Node Detection")
    print("=" * 50)

    print(f"\n📍 Host: {profile.hostname}")
    if profile.lan_ip:
        print(f"   LAN:  {profile.lan_ip}")

    if profile.cpu:
        print(f"\n🔧 CPU: {profile.cpu.model}")
        print(
            f"   Cores: {profile.cpu.cores_physical}P / "
            f"{profile.cpu.cores_logical}L @ {profile.cpu.frequency_mhz:.0f} MHz"
        )

    if profile.memory:
        print(
            f"\n💾 RAM: {profile.memory.total_mb:,} MB total, "
            f"{profile.memory.available_mb:,} MB available "
            f"({profile.memory.percent_used:.0f}% used)"
        )

    if profile.gpus:
        for gpu in profile.gpus:
            print(f"\n🎮 GPU {gpu.index}: {gpu.name}")
            print(
                f"   VRAM: {gpu.vram_total_mb:,} MB total, "
                f"{gpu.vram_free_mb:,} MB free"
            )
            if gpu.temperature_c is not None:
                print(f"   Temp: {gpu.temperature_c}°C")
            if gpu.driver_version:
                print(f"   Driver: {gpu.driver_version}  CUDA: {gpu.cuda_version}")
    else:
        print("\n🎮 GPU: None detected")

    if profile.disks:
        print("\n💿 Disk:")
        for disk in profile.disks:
            print(
                f"   {disk.path}: {disk.free_gb:.0f} GB free "
                f"/ {disk.total_gb:.0f} GB ({disk.percent_used:.0f}% used)"
            )

    if profile.ollama_running:
        print(f"\n🦙 Ollama: running ({len(profile.ollama_models)} models)")
        for m in profile.ollama_models:
            quant = f" ({m.quantization})" if m.quantization else ""
            print(f"   • {m.name} [{m.parameter_size}{quant}] {m.size_mb:,} MB")
    else:
        print("\n🦙 Ollama: not detected")

    print(f"\n{'=' * 50}")
    print(f"📊 Node Tier: {caps.node_tier.value.upper()}")
    print(f"   GPU Tier:  {caps.gpu_tier.value}")
    print(
        f"   Max Model: ~{caps.max_model_params_b}B parameters (Q4 quantized)"
        if caps.max_model_params_b > 0
        else "   Max Model: CPU-only (≤3B)"
    )

    print(f"\n🔑 Capabilities:")
    for cap in caps.capabilities:
        print(f"   ✓ {cap.value}")

    if caps.recommended_models:
        print(f"\n📦 Recommended Models:")
        for model in caps.recommended_models:
            print(f"   • {model}")

    if caps.notes:
        print(f"\n📝 Notes:")
        for note in caps.notes:
            print(f"   {note}")


def cmd_identity(args):
    """Show the full node identity announcement."""
    identity = build_identity()
    print(identity.to_json())


def cmd_daemon(args):
    """Start the node daemon."""
    from mycoswarm.daemon import start_daemon

    start_daemon(port=args.port, verbose=args.verbose)


def cmd_dashboard(args):
    """Start the web dashboard."""
    import uvicorn
    from mycoswarm.dashboard import create_app

    app = create_app(daemon_port=args.daemon_port)
    print(f"🍄 Dashboard running at http://localhost:{args.port}")
    print(f"   Querying daemon on port {args.daemon_port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


def cmd_swarm(args):
    """Show swarm status by querying the local daemon."""
    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"

    try:
        with httpx.Client(timeout=5) as client:
            status = client.get(f"{url}/status").json()
            peers_data = client.get(f"{url}/peers").json()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)

    print("🍄 mycoSwarm — Swarm Status")
    print("=" * 60)

    print(f"\n📍 This Node: {status['hostname']} [{status['node_tier'].upper()}]")
    if status.get('gpu'):
        print(f"   GPU: {status['gpu']} ({status['vram_total_mb']} MB VRAM)")
    print(f"   Caps: {', '.join(status['capabilities'])}")
    print(f"   Models: {len(status.get('ollama_models', []))}")
    print(f"   Uptime: {status['uptime_seconds']:.0f}s")

    if peers_data:
        print(f"\n🌐 Peers ({len(peers_data)}):")
        for p in peers_data:
            gpu_info = f" [{p['gpu_name']}]" if p.get('gpu_name') else ""
            tier = p['node_tier'].upper()
            print(f"   • {p['hostname']} ({p['ip']}) [{tier}]{gpu_info}")
            print(f"     Caps: {', '.join(p['capabilities'])}")
            if p['vram_total_mb'] > 0:
                print(f"     VRAM: {p['vram_total_mb']} MB")
    else:
        print("\n🌐 Peers: none discovered yet")

    total_nodes = 1 + len(peers_data)
    gpu_nodes = (1 if status.get('gpu') else 0) + sum(
        1 for p in peers_data if p.get('gpu_name')
    )
    total_vram = status.get('vram_total_mb', 0) + sum(
        p.get('vram_total_mb', 0) for p in peers_data
    )

    print(f"\n{'=' * 60}")
    print(f"📊 Swarm Total:")
    print(f"   Nodes:      {total_nodes}")
    print(f"   GPU Nodes:  {gpu_nodes}")
    print(f"   CPU Nodes:  {total_nodes - gpu_nodes}")
    print(f"   Total VRAM: {total_vram:,} MB")
    print(f"   Tasks:      {status.get('tasks_pending', 0)} pending, "
          f"{status.get('tasks_active', 0)} active")


def cmd_ping(args):
    """Ping all known peers."""
    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"

    try:
        with httpx.Client(timeout=5) as client:
            peers_data = client.get(f"{url}/peers").json()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)

    if not peers_data:
        print("No peers discovered yet.")
        return

    print(f"🏓 Pinging {len(peers_data)} peer(s)...\n")

    import time

    with httpx.Client(timeout=5) as client:
        for p in peers_data:
            peer_url = f"http://{p['ip']}:{p['port']}/health"
            try:
                start = time.time()
                resp = client.get(peer_url)
                elapsed = (time.time() - start) * 1000
                data = resp.json()
                print(
                    f"  ✅ {p['hostname']} ({p['ip']}) — "
                    f"{elapsed:.0f}ms — "
                    f"up {data['uptime_seconds']:.0f}s — "
                    f"{data['peer_count']} peer(s)"
                )
            except Exception as e:
                print(f"  ❌ {p['hostname']} ({p['ip']}) — {e}")


def _discover_model(url: str, prefer: str | None = None) -> str:
    """Pick the best model from the swarm. Returns model name or exits."""
    if prefer:
        return prefer
    try:
        with httpx.Client(timeout=5) as client:
            status = client.get(f"{url}/status").json()
            models = status.get("ollama_models", [])

            # If local node has no models, check peers (best VRAM first)
            if not models:
                peers_data = client.get(f"{url}/peers").json()
                peers_with_models = [
                    p for p in peers_data
                    if p.get("available_models")
                ]
                if peers_with_models:
                    best_peer = max(
                        peers_with_models,
                        key=lambda p: p.get("vram_total_mb", 0),
                    )
                    models = best_peer["available_models"]

            if models:
                # Prefer a 14b+ model, fall back to first available
                model = models[0]
                for m in models:
                    if "14b" in m or "32b" in m or "27b" in m:
                        model = m
                        break
                return model
            else:
                print("❌ No Ollama models available in the swarm.")
                sys.exit(1)
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)


def _stream_response(
    url: str, task_id: str, timeout: int = 300
) -> tuple[str, dict]:
    """Consume SSE stream from /task/{id}/stream.

    Returns (full_text, metrics_dict).  metrics_dict may be empty on error.
    """
    import json

    tokens: list[str] = []
    metrics: dict = {}

    try:
        with httpx.Client(timeout=httpx.Timeout(5.0, read=timeout)) as client:
            with client.stream("GET", f"{url}/task/{task_id}/stream") as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]  # strip "data: "
                    try:
                        event = json.loads(payload)
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
                            for k in (
                                "model",
                                "tokens_per_second",
                                "duration_seconds",
                                "node_id",
                            )
                            if k in event
                        }

    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)
    except httpx.ReadTimeout:
        print("\n❌ Stream timed out.")

    return "".join(tokens), metrics


def _submit_and_poll(url: str, task_payload: dict, timeout: int = 300) -> dict | None:
    """Submit a task and poll until completion. Returns result dict or None."""
    import time

    task_id = task_payload["task_id"]
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.post(f"{url}/task", json=task_payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)

    start = time.time()
    with httpx.Client(timeout=5) as client:
        while time.time() - start < timeout:
            time.sleep(0.5)
            try:
                result_resp = client.get(f"{url}/task/{task_id}")
                data = result_resp.json()

                if data.get("status") == "completed":
                    return data
                elif data.get("status") == "failed":
                    return data
            except Exception:
                pass

    return None


def cmd_ask(args):
    """Send a prompt to the swarm for inference."""
    from mycoswarm.solo import check_daemon, check_ollama, pick_model, ask_direct

    prompt = " ".join(args.prompt)

    # Try daemon first; fall back to direct Ollama
    if check_daemon(args.port):
        import uuid

        profile = detect_all()
        ip = profile.lan_ip or "localhost"
        url = f"http://{ip}:{args.port}"
        model = _discover_model(url, args.model)

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task_payload = {
            "task_id": task_id,
            "task_type": "inference",
            "payload": {
                "model": model,
                "prompt": prompt,
            },
            "source_node": "cli",
            "priority": 5,
            "timeout_seconds": 300,
        }

        print(f"🍄 Asking: {prompt}")
        print(f"   Model: {model}")
        print(f"   Sending to {ip}:{args.port}...\n")

        data = _submit_and_poll(url, task_payload)
        if data is None:
            print("❌ Timed out waiting for response.")
            sys.exit(1)
        if data.get("status") == "failed":
            print(f"❌ Task failed: {data.get('error', 'unknown error')}")
            sys.exit(1)

        result = data.get("result", {})
        print(result.get("response", ""))
        print(f"\n{'─' * 50}")
        print(
            f"  ⏱  {data.get('duration_seconds', 0):.1f}s | "
            f"{result.get('tokens_per_second', 0):.1f} tok/s | "
            f"model: {model}"
        )
        return

    # Single-node mode — direct to Ollama
    running, models = check_ollama()
    if not running:
        print("❌ No daemon running and Ollama is not reachable.")
        print("   Start Ollama with: ollama serve")
        print("   Or start the daemon with: mycoswarm daemon")
        sys.exit(1)

    model = pick_model(models, args.model)
    print(f"🍄 Running in single-node mode. Start the daemon to join a swarm.")
    print(f"   Model: {model}\n")
    ask_direct(prompt, model)


def cmd_search(args):
    """Search the web via the swarm."""
    import uuid

    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"
    query = " ".join(args.query)
    max_results = args.max_results

    task_id = f"task-{uuid.uuid4().hex[:8]}"
    task_payload = {
        "task_id": task_id,
        "task_type": "web_search",
        "payload": {
            "query": query,
            "max_results": max_results,
        },
        "source_node": "cli",
        "priority": 5,
        "timeout_seconds": 60,
    }

    print(f"🔍 Searching: {query}")
    print(f"   Sending to {ip}:{args.port}...\n")

    data = _submit_and_poll(url, task_payload, timeout=60)
    if data is None:
        print("❌ Timed out waiting for search results.")
        sys.exit(1)
    if data.get("status") == "failed":
        print(f"❌ Search failed: {data.get('error', 'unknown error')}")
        sys.exit(1)

    result = data.get("result", {})
    results = result.get("results", [])

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        print(f"     {r['snippet']}")
        print(f"     🔗 {r['url']}")
        print()

    print(f"{'─' * 50}")
    node_id = data.get("node_id", "")
    duration = data.get("duration_seconds", 0)
    print(f"  ⏱  {duration:.1f}s | {len(results)} results | node: {node_id}")


def _resolve_node_name(url: str, node_id: str) -> str:
    """Map a node_id to a hostname via /status and /peers."""
    try:
        with httpx.Client(timeout=3) as client:
            status = client.get(f"{url}/status").json()
            if status.get("node_id") == node_id:
                return status.get("hostname", node_id)
            for p in client.get(f"{url}/peers").json():
                if p.get("node_id") == node_id:
                    return p.get("hostname", node_id)
    except Exception:
        pass
    return node_id


def _do_search(url: str, query: str, task_id: str, max_results: int) -> dict:
    """Submit a web_search task and poll to completion. Thread-safe."""
    import time as _t

    payload = {
        "task_id": task_id,
        "task_type": "web_search",
        "payload": {"query": query, "max_results": max_results},
        "source_node": "cli",
        "priority": 7,
        "timeout_seconds": 60,
    }
    start = _t.time()
    # Submit and capture routing info
    with httpx.Client(timeout=5) as client:
        resp = client.post(f"{url}/task", json=payload)
        resp.raise_for_status()
        submit_data = resp.json()
    routed_to = submit_data.get("message", "")

    # Poll for result
    data = None
    with httpx.Client(timeout=5) as client:
        while _t.time() - start < 60:
            _t.sleep(0.5)
            try:
                r = client.get(f"{url}/task/{task_id}")
                d = r.json()
                if d.get("status") in ("completed", "failed"):
                    data = d
                    break
            except Exception:
                pass

    elapsed = round(_t.time() - start, 1)
    node_id = data.get("node_id", "") if data else ""
    # Use hostname from routing message, else resolve node_id → hostname
    if "Routed to" in routed_to:
        node_name = routed_to.replace("Routed to ", "")
    elif node_id:
        node_name = _resolve_node_name(url, node_id)
    else:
        node_name = ""
    return {
        "query": query, "task_id": task_id, "data": data,
        "node": node_name, "elapsed": elapsed,
    }


def cmd_research(args):
    """Search the web then synthesize results via LLM inference.

    The swarm's signature move: CPU workers search, GPU nodes think.
    Plan → parallel search → synthesize.
    """
    import json as _json
    import uuid
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"
    query = " ".join(args.query)
    model = _discover_model(url, args.model)

    # --- Phase 1: Planning — ask model to decompose into search queries ---
    print("🧠 Planning...")

    plan_id = f"research-plan-{uuid.uuid4().hex[:8]}"
    plan_payload = {
        "task_id": plan_id,
        "task_type": "inference",
        "payload": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Decompose the user's question into 2-4 specific web "
                        "search queries. Respond with ONLY JSON, no explanation: "
                        '{"searches": ["query1", "query2"]}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        },
        "source_node": "cli",
        "priority": 8,
        "timeout_seconds": 30,
    }

    try:
        plan_data = _submit_and_poll(url, plan_payload, timeout=30)
    except SystemExit:
        raise
    except Exception:
        plan_data = None

    # Parse search queries from model response
    search_queries = []
    if plan_data and plan_data.get("status") == "completed":
        raw_response = plan_data.get("result", {}).get("response", "")
        # Extract JSON from response (model might wrap in markdown)
        json_str = raw_response.strip()
        if "```" in json_str:
            # Strip markdown code fences
            for block in json_str.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    json_str = block
                    break
        try:
            parsed = _json.loads(json_str)
            search_queries = parsed.get("searches", [])[:4]
        except _json.JSONDecodeError:
            pass

    # Fallback: use original query if planning failed
    if not search_queries:
        search_queries = [query]
        print("   Using original query (planning skipped)")
    else:
        print(f"   {len(search_queries)} searches planned:")
        for sq in search_queries:
            print(f"     • {sq}")

    # --- Phase 2: Parallel search across CPU workers ---
    print()
    search_start = time.time()
    results_per_query: list[dict] = []

    def _run_search(sq: str) -> dict:
        sid = f"research-s-{uuid.uuid4().hex[:8]}"
        return _do_search(url, sq, sid, args.max_results)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_search, sq): sq for sq in search_queries}
        for future in as_completed(futures):
            rq = future.result()
            results_per_query.append(rq)
            node_label = f" → {rq['node']}" if rq.get("node") else ""
            status = "✅" if rq.get("data", {}).get("status") == "completed" else "❌"
            print(f"  {status} \"{rq['query']}\"{node_label} ({rq['elapsed']}s)")

    search_duration = round(time.time() - search_start, 1)

    # Collect and deduplicate results by URL
    seen_urls: set[str] = set()
    all_results: list[dict] = []
    for rq in results_per_query:
        data = rq.get("data")
        if not data or data.get("status") != "completed":
            continue
        for r in data.get("result", {}).get("results", []):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

    if not all_results:
        print("❌ No search results found.")
        sys.exit(1)

    print(
        f"✅ {len(all_results)} unique results from "
        f"{len(search_queries)} searches ({search_duration}s)"
    )

    # --- Phase 3: Synthesis inference (GPU node) ---
    context_lines = []
    sources = []
    for i, r in enumerate(all_results, 1):
        context_lines.append(f"[{i}] {r['title']}")
        context_lines.append(f"    URL: {r['url']}")
        context_lines.append(f"    {r['snippet']}")
        context_lines.append("")
        sources.append({"num": i, "title": r["title"], "url": r["url"]})
    context_block = "\n".join(context_lines)

    system_prompt = (
        "You are a research assistant. The user asked a question and web search "
        "results are provided below as context. Synthesize the information into "
        "a clear, well-organized answer. Cite sources using [1], [2], etc. "
        "matching the numbered results. Be concise but thorough. If the search "
        "results don't fully answer the question, say so.\n\n"
        f"SEARCH RESULTS:\n{context_block}"
    )

    infer_id = f"research-infer-{uuid.uuid4().hex[:8]}"
    infer_payload = {
        "task_id": infer_id,
        "task_type": "inference",
        "payload": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        },
        "source_node": "cli",
        "priority": 7,
        "timeout_seconds": 300,
    }

    print(f"\n🧠 Synthesizing with {model}...")

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{url}/task", json=infer_payload)
            resp.raise_for_status()
            infer_submit = resp.json()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        print(f"❌ {detail}")
        sys.exit(1)

    infer_node = infer_submit.get("message", "")
    if "Routed to" in infer_node:
        print(f"   {infer_node}...")

    # Stream directly from target for remote inference
    target_ip = infer_submit.get("target_ip")
    target_port = infer_submit.get("target_port")
    if target_ip and target_port:
        stream_url = f"http://{target_ip}:{target_port}"
    else:
        stream_url = url

    print()
    full_text, metrics = _stream_response(stream_url, infer_id)

    if not full_text:
        print("❌ No response from model.")
        sys.exit(1)

    # --- Footer: sources + metrics ---
    tps = metrics.get("tokens_per_second", 0)
    duration = metrics.get("duration_seconds", 0)
    node_id = metrics.get("node_id", "")

    print(f"\n\n{'─' * 50}")
    print("📚 Sources:")
    for s in sources:
        print(f"   [{s['num']}] {s['title']}")
        print(f"       {s['url']}")
    print(f"{'─' * 50}")
    print(
        f"  🔍 {len(search_queries)} searches: {search_duration}s | "
        f"🧠 synthesis: {duration:.1f}s {tps:.1f} tok/s ({node_id}) | "
        f"model: {model}"
    )


def _list_swarm_models(url: str) -> list[str]:
    """Gather all unique models across the swarm."""
    models = set()
    try:
        with httpx.Client(timeout=5) as client:
            status = client.get(f"{url}/status").json()
            models.update(status.get("ollama_models", []))
            peers = client.get(f"{url}/peers").json()
            for p in peers:
                models.update(p.get("available_models", []))
    except httpx.ConnectError:
        pass
    return sorted(models)


def cmd_models(args):
    """Show all models available across the swarm."""
    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"

    try:
        with httpx.Client(timeout=5) as client:
            status = client.get(f"{url}/status").json()
            peers_data = client.get(f"{url}/peers").json()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)

    # Build model → list of nodes mapping
    model_nodes: dict[str, list[dict]] = {}

    # Local node
    for model_name in status.get("ollama_models", []):
        node_info = {
            "hostname": status["hostname"],
            "gpu_name": status.get("gpu"),
            "vram_total_mb": status.get("vram_total_mb", 0),
        }
        model_nodes.setdefault(model_name, []).append(node_info)

    # Peers
    for p in peers_data:
        for model_name in p.get("available_models", []):
            node_info = {
                "hostname": p["hostname"],
                "gpu_name": p.get("gpu_name"),
                "vram_total_mb": p.get("vram_total_mb", 0),
            }
            model_nodes.setdefault(model_name, []).append(node_info)

    if not model_nodes:
        print("❌ No models available in the swarm.")
        print("   Install models with: ollama pull <model>")
        sys.exit(1)

    print("🍄 mycoSwarm — Available Models")
    print("=" * 60)

    for model_name in sorted(model_nodes):
        nodes = model_nodes[model_name]
        print(f"\n  {model_name}")
        for n in nodes:
            gpu = f"{n['gpu_name']}, {n['vram_total_mb'] // 1024}GB" if n["gpu_name"] else "CPU only"
            print(f"    • {n['hostname']} ({gpu})")

    total = len(model_nodes)
    node_count = 1 + len(peers_data)
    print(f"\n{'=' * 60}")
    print(f"  {total} model(s) across {node_count} node(s)")


def cmd_plugins(args):
    """List installed plugins and their status."""
    from mycoswarm.plugins import discover_plugins, PLUGIN_DIR

    plugins = discover_plugins()

    print("🔌 mycoSwarm — Installed Plugins")
    print(f"   Directory: {PLUGIN_DIR}")
    print("=" * 60)

    if not plugins:
        print("\n  No plugins found.")
        print(f"\n  To install a plugin, create a subdirectory in:")
        print(f"    {PLUGIN_DIR}/")
        print(f"  with plugin.yaml + handler.py")
        return

    for p in plugins:
        status = "✅ loaded" if p.loaded else f"❌ {p.error}"
        print(f"\n  {p.name}")
        print(f"    Task type:    {p.task_type or '(none)'}")
        print(f"    Description:  {p.description or '(none)'}")
        if p.capabilities:
            print(f"    Capabilities: {', '.join(p.capabilities)}")
        print(f"    Path:         {p.path}")
        print(f"    Status:       {status}")

    loaded = sum(1 for p in plugins if p.loaded)
    print(f"\n{'=' * 60}")
    print(f"  {loaded}/{len(plugins)} plugin(s) loaded")


SESSIONS_DIR = "~/.config/mycoswarm/sessions"


def _sessions_path() -> "Path":
    from pathlib import Path
    p = Path(SESSIONS_DIR).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_session(
    name: str, messages: list[dict], model: str,
) -> str:
    """Save chat session to disk. Returns the file path."""
    import json
    from datetime import datetime

    path = _sessions_path() / f"{name}.json"
    data = {
        "name": name,
        "model": model,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages,
    }
    # Update timestamps if file already exists
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            data["created"] = existing.get("created", data["created"])
        except (json.JSONDecodeError, KeyError):
            pass

    path.write_text(json.dumps(data, indent=2))
    return str(path)


def _load_session(name: str) -> tuple[list[dict], str] | None:
    """Load a session by name. Returns (messages, model) or None."""
    import json

    path = _sessions_path() / f"{name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("messages", []), data.get("model", "")
    except (json.JSONDecodeError, KeyError):
        return None


def _list_sessions() -> list[dict]:
    """List all saved sessions with metadata."""
    import json

    sessions = []
    for f in sorted(_sessions_path().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "name": f.stem,
                "model": data.get("model", ""),
                "updated": data.get("updated", ""),
                "message_count": data.get("message_count", 0),
            })
        except (json.JSONDecodeError, KeyError):
            pass
    return sessions


def _latest_session_name() -> str | None:
    """Return the name of the most recently updated session."""
    sessions = _list_sessions()
    return sessions[0]["name"] if sessions else None


def cmd_chat(args):
    """Interactive chat with the swarm."""
    from datetime import datetime
    from mycoswarm.solo import check_daemon, check_ollama, pick_model, chat_stream

    # Handle --list
    if args.list_sessions:
        sessions = _list_sessions()
        if not sessions:
            print("No saved sessions.")
            return
        print("🍄 Saved Chat Sessions")
        print("=" * 60)
        for s in sessions:
            updated = s["updated"][:16].replace("T", " ") if s["updated"] else ""
            print(
                f"  {s['name']}  ({s['message_count']} msgs, "
                f"{s['model']}, {updated})"
            )
        return

    debug = getattr(args, "debug", False)

    daemon_up = check_daemon(args.port)

    if daemon_up:
        profile = detect_all()
        ip = profile.lan_ip or "localhost"
        url = f"http://{ip}:{args.port}"
    else:
        running, models = check_ollama()
        if not running:
            print("❌ No daemon running and Ollama is not reachable.")
            print("   Start Ollama with: ollama serve")
            print("   Or start the daemon with: mycoswarm daemon")
            sys.exit(1)

    # Session loading
    session_name = None
    messages: list[dict[str, str]] = []

    if args.resume:
        session_name = _latest_session_name()
        if session_name:
            loaded = _load_session(session_name)
            if loaded:
                messages, saved_model = loaded
                if not args.model and saved_model:
                    args.model = saved_model
        else:
            print("No previous session to resume.")

    elif args.session:
        session_name = args.session
        loaded = _load_session(session_name)
        if loaded:
            messages, saved_model = loaded
            if not args.model and saved_model:
                args.model = saved_model
        else:
            print(f"Session '{session_name}' not found, starting new.")

    if not session_name:
        session_name = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # --- Identity layer: first-run naming ---
    from mycoswarm.identity import load_identity, seed_identity, save_identity

    identity = load_identity()
    if not identity.get("name") and not messages:
        print("🍄 This is your first time running mycoSwarm chat.")
        print("   Would you like to give your AI a name?")
        name = input("   Name (or press Enter to skip): ").strip()
        if name:
            identity = seed_identity(name)
            print(f"   Welcome to the world, {name}. 🍄")
        else:
            print("   No problem — you can name it later with /name")

    if daemon_up:
        model = _discover_model(url, args.model)
    else:
        model = pick_model(models, args.model)

    # Inject persistent memory into messages (identity goes first)
    from mycoswarm.memory import build_memory_system_prompt
    from mycoswarm.identity import build_identity_prompt

    identity_prompt = build_identity_prompt(identity)
    memory_prompt = build_memory_system_prompt()
    system_prompt = identity_prompt + "\n\n" + memory_prompt if memory_prompt else identity_prompt
    if not messages:
        messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        if messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

    print("🍄 mycoSwarm Chat")
    print(f"   Model: {model}")
    print(f"   Session: {session_name}")
    if not daemon_up:
        print("   Running in single-node mode. Start the daemon to join a swarm.")
    if messages:
        print(f"   Resumed: {len(messages)} messages")
    if daemon_up:
        print("   /model /peers /rag /library /auto /remember /memories /stale /forget /identity /name /vitals /timing /clear /quit")
    else:
        print("   /model /rag /library /auto /remember /memories /stale /forget /identity /name /vitals /timing /clear /quit")
    print(f"{'─' * 50}")

    auto_tools = True  # agentic tool routing on by default
    session_rag_context: list[str] = []  # accumulated RAG context for grounding check

    def _save():
        if debug:
            print("   🐛 DEBUG: Skipping session save/summarize in debug mode.")
            return
        _save_session(session_name, messages, model)
        print(f"   Session saved: {session_name}")

        # Summarize session for persistent memory
        from mycoswarm.memory import (
            summarize_session_rich, save_session_summary,
            compute_grounding_score,
        )
        # Only summarize if there are user+assistant messages (skip system-only)
        user_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
        if len(user_msgs) >= 2:
            print("   Reflecting on session...", end=" ", flush=True)
            rich = summarize_session_rich(messages, model)
            if rich:
                user_texts = [m["content"] for m in user_msgs if m["role"] == "user"]
                gs = compute_grounding_score(
                    rich["summary"], user_texts, session_rag_context,
                )
                save_session_summary(
                    session_name, model, rich["summary"], len(user_msgs),
                    grounding_score=gs,
                    decisions=rich.get("decisions"),
                    lessons=rich.get("lessons"),
                    surprises=rich.get("surprises"),
                    emotional_tone=rich.get("emotional_tone"),
                )
                print(f"done. (grounding: {gs:.0%})")
                tone = rich.get("emotional_tone", "neutral")
                lessons = rich.get("lessons", [])
                if tone != "neutral":
                    print(f"   Tone: {tone}")
                if lessons:
                    for le in lessons[:2]:
                        print(f"   Lesson: {le}")
            else:
                print("skipped.")
        else:
            print("   (too short to summarize)")

    intent_result = None  # Updated each turn by auto-tools classification
    _last_vitals = None   # Most recent vital signs for /vitals display
    _last_turn_time = None   # Tracks time between turns (for timing gate)
    _turn_count = 0          # Session turn count (for timing gate)
    _last_timing = None      # Most recent TimingDecision for /timing display

    from datetime import datetime as _dt_timing
    from mycoswarm.timing import evaluate_timing, TimingMode

    while True:
        try:
            sys.stdout.flush()
            user_input = input("\n🍄> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _save()
            print("Bye. 🍄")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                _save()
                print("Bye. 🍄")
                break

            elif cmd == "/clear":
                messages.clear()
                print("   Conversation cleared.")
                continue

            elif cmd == "/save":
                _save()
                continue

            elif cmd == "/model":
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    model = parts[1]
                    print(f"   Model → {model}")
                else:
                    if daemon_up:
                        all_models = _list_swarm_models(url)
                    else:
                        _, all_models = check_ollama()
                    if all_models:
                        print("   Available models:")
                        for m in all_models:
                            marker = " ◀" if m == model else ""
                            print(f"     • {m}{marker}")
                    else:
                        print("   No models found.")
                continue

            elif cmd == "/peers":
                if not daemon_up:
                    print("   No peers (single-node mode). Start the daemon to join a swarm.")
                    continue
                try:
                    with httpx.Client(timeout=5) as client:
                        peers = client.get(f"{url}/peers").json()
                    if peers:
                        for p in peers:
                            gpu = f" [{p['gpu_name']}]" if p.get("gpu_name") else ""
                            print(
                                f"   • {p['hostname']} ({p['ip']}) "
                                f"[{p['node_tier'].upper()}]{gpu}"
                            )
                    else:
                        print("   No peers.")
                except httpx.ConnectError:
                    print("   ❌ Can't reach daemon.")
                continue

            elif cmd == "/remember":
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("   Usage: /remember [type:] <fact>")
                    print("   Types: pref: | project: | temp: (default: fact)")
                    continue
                raw_text = parts[1]
                _type_prefixes = {
                    "pref:": "preference",
                    "preference:": "preference",
                    "project:": "project",
                    "temp:": "ephemeral",
                    "ephemeral:": "ephemeral",
                }
                fact_type = "fact"
                for _pfx, _ft in _type_prefixes.items():
                    if raw_text.lower().startswith(_pfx):
                        fact_type = _ft
                        raw_text = raw_text[len(_pfx):].strip()
                        break
                from mycoswarm.memory import add_fact
                fact = add_fact(raw_text, fact_type=fact_type)
                print(f"   Remembered (#{fact['id']}, {fact_type}): {fact['text']}")
                continue

            elif cmd == "/memories":
                from mycoswarm.memory import load_facts
                facts = load_facts()
                if not facts:
                    print("   No facts stored. Use /remember <fact> to add one.")
                else:
                    for f in facts:
                        ftype = f.get("type", "fact")
                        refs = f.get("reference_count", 0)
                        print(f"   #{f['id']} ({ftype}, {refs} refs): {f['text']}")
                continue

            elif cmd == "/stale":
                from mycoswarm.memory import get_stale_facts
                stale = get_stale_facts(days=30)
                if stale:
                    print(f"   {len(stale)} stale fact(s):")
                    for f in stale:
                        ftype = f.get("type", "fact")
                        last_ref = f.get("last_referenced", "unknown")[:10]
                        print(f"   [{f['id']}] ({ftype}) {f['text']}")
                        print(f"       Last referenced: {last_ref}")
                else:
                    print("   All facts recently referenced.")
                continue

            elif cmd == "/forget":
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2 or not parts[1].isdigit():
                    print("   Usage: /forget <number>")
                    continue
                from mycoswarm.memory import remove_fact
                if remove_fact(int(parts[1])):
                    print(f"   Forgot #{parts[1]}.")
                else:
                    print(f"   Fact #{parts[1]} not found.")
                continue

            elif cmd == "/identity":
                if identity.get("name"):
                    print("   🍄 Identity")
                    print(f"      Name: {identity['name']}")
                    print(f"      Origin: {identity.get('origin', 'unknown')}")
                    print(f"      Substrate: {identity.get('substrate', 'unknown')}")
                    status = "developing" if identity.get("developing") else "stable"
                    print(f"      Status: {status}")
                else:
                    print("   No identity set. Use /name to give me a name.")
                continue

            elif cmd == "/name":
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("   Usage: /name <new_name>")
                    continue
                new_name = parts[1].strip()
                identity["name"] = new_name
                if not identity.get("origin"):
                    identity["origin"] = f"Named by user, {datetime.now().strftime('%B %Y')}"
                    identity["substrate"] = "mycoSwarm distributed network"
                    identity["created"] = datetime.now().isoformat()
                    identity["developing"] = True
                save_identity(identity)
                # Refresh system prompt with new identity
                id_prompt = build_identity_prompt(identity)
                mem_prompt = build_memory_system_prompt()
                sys_prompt = id_prompt + "\n\n" + mem_prompt if mem_prompt else id_prompt
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": sys_prompt}
                print(f"   I'm {new_name} now. 🍄")
                continue

            elif cmd == "/vitals":
                if _last_vitals is not None:
                    _vname = identity.get("name", "Monica")
                    print(f"   {_last_vitals.detailed_display(_vname)}")
                else:
                    print("   No vitals yet — ask me something first.")
                continue

            elif cmd == "/timing":
                from datetime import datetime as _dt_now
                _now = _dt_now.now()
                print("🕐 Timing Gate")
                print("─────────────────────────")
                if _last_timing is not None:
                    print(f"  Mode:     {_last_timing.mode.value.upper()} {_last_timing.status_indicator()}")
                else:
                    print("  Mode:     PROCEED ▶ (no turns yet)")
                print(f"  Time:     {_now.strftime('%I:%M %p').lstrip('0')} ({'late night' if _now.hour >= 23 or _now.hour < 6 else 'early morning' if _now.hour < 9 else 'morning peak' if _now.hour < 12 else 'afternoon' if _now.hour < 17 else 'evening'})")
                print(f"  Session:  turn {_turn_count} of current session")
                if _last_turn_time is not None:
                    _gap = (_now - _last_turn_time).total_seconds()
                    print(f"  Gap:      {_gap:.0f}s since last message")
                else:
                    print("  Gap:      (first message)")
                if _last_timing is not None and _last_timing.reasons:
                    print("  Reasons:")
                    for _r in _last_timing.reasons:
                        print(f"    • {_r}")
                print("─────────────────────────")
                continue

            elif cmd == "/rag":
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("   Usage: /rag <question>")
                    continue
                rag_query = parts[1]
                from mycoswarm.library import search_all
                from mycoswarm.solo import detect_past_reference as _dpr
                _past = _dpr(rag_query)
                doc_hits, session_hits, procedure_hits = search_all(
                    rag_query, n_results=5, session_boost=_past,
                    intent=intent_result,
                )
                if not doc_hits and not session_hits:
                    print("   No documents or sessions indexed. Try: mycoswarm library ingest")
                    continue
                # Build context with distinct labels
                rag_parts = []
                for ri, hit in enumerate(doc_hits, 1):
                    section = hit.get("section", "untitled")
                    rag_parts.append(f"[D{ri}] (From: {hit['source']} > {section}) {hit['text']}")
                for si, hit in enumerate(session_hits, 1):
                    date = hit.get("date", "unknown")
                    topic = hit.get("topic", "")
                    label = f"conversation on {date}"
                    if topic:
                        label += f" — {topic}"
                    rag_parts.append(f"[S{si}] (From {label}) {hit['summary']}")
                if procedure_hits:
                    from mycoswarm.memory import format_procedures_for_prompt, reference_procedure
                    proc_text = format_procedures_for_prompt(procedure_hits)
                    rag_parts.append(f"\nRelevant procedures (past solutions):\n{proc_text}")
                    for p in procedure_hits:
                        reference_procedure(p.get("id", ""))
                rag_context = "\n\n".join(rag_parts)
                cite_parts = []
                if doc_hits:
                    cite_parts.append("[D1], [D2] for documents")
                if session_hits:
                    cite_parts.append("[S1], [S2] for past conversations")
                if procedure_hits:
                    cite_parts.append("[P1], [P2] for known procedures")
                cite_hint = "Cite using " + " and ".join(cite_parts) + "." if cite_parts else ""
                rag_system = (
                    "IMPORTANT: The excerpts below are REAL content retrieved from "
                    "the user's actual files and past conversations. Use them as "
                    "your primary source of truth. Quote specific details from the "
                    "excerpts. Do NOT say you cannot access files — the content is "
                    f"provided below. {cite_hint}\n\nRETRIEVED CONTEXT:\n" + rag_context
                )
                # Inject context into user message for better grounding
                augmented_query = rag_system + "\n\nUSER QUESTION: " + rag_query
                rag_messages = list(messages) + [
                    {"role": "user", "content": augmented_query},
                ]
                total_hits = len(doc_hits) + len(session_hits) + len(procedure_hits)
                print(f"   📚 {total_hits} excerpts found ({len(doc_hits)} docs, {len(session_hits)} sessions).\n")
                if not daemon_up:
                    full_text, metrics = chat_stream(rag_messages, model)
                else:
                    import uuid as _uuid
                    _tid = f"rag-{_uuid.uuid4().hex[:8]}"
                    _tp = {
                        "task_id": _tid,
                        "task_type": "inference",
                        "payload": {"model": model, "messages": rag_messages},
                        "source_node": "cli-chat-rag",
                        "priority": 5,
                        "timeout_seconds": 300,
                    }
                    try:
                        with httpx.Client(timeout=10) as client:
                            _resp = client.post(f"{url}/task", json=_tp)
                            _resp.raise_for_status()
                            _sd = _resp.json()
                    except (httpx.ConnectError, httpx.HTTPStatusError):
                        print("   ❌ Failed to submit to daemon.")
                        continue
                    _tip = _sd.get("target_ip")
                    _tpo = _sd.get("target_port")
                    _surl = f"http://{_tip}:{_tpo}" if _tip and _tpo else url
                    full_text, metrics = _stream_response(_surl, _tid)
                if full_text:
                    messages.append({"role": "user", "content": rag_query})
                    messages.append({"role": "assistant", "content": full_text})
                    source_labels = sorted({h["source"] for h in doc_hits})
                    if session_hits:
                        source_labels.append("sessions")
                    tps = metrics.get("tokens_per_second", 0)
                    duration = metrics.get("duration_seconds", 0)
                    print(
                        f"\n\n{'─' * 50}\n"
                        f"  📚 {', '.join(source_labels)} | "
                        f"⏱  {duration:.1f}s | {tps:.1f} tok/s | {model}"
                    )
                continue

            elif cmd == "/library":
                from mycoswarm.library import list_documents
                docs = list_documents()
                if not docs:
                    print("   No documents indexed. Try: mycoswarm library ingest")
                else:
                    for d in docs:
                        print(f"   {d['file']}  ({d['chunks']} chunks)")
                continue

            elif cmd == "/auto":
                auto_tools = not auto_tools
                state = "on" if auto_tools else "off"
                print(f"   Auto tool use: {state}")
                continue

            elif user_input.startswith("/procedure"):
                parts = user_input.split(maxsplit=1)
                subcmd = parts[1].strip() if len(parts) > 1 else "list"

                if subcmd == "list":
                    from mycoswarm.memory import load_procedures
                    procs = load_procedures()
                    if not procs:
                        print("  No procedures stored yet.")
                        print("  Add one: /procedure add <problem> | <solution>")
                    else:
                        active = [p for p in procs if p.get("status", "active") == "active"]
                        candidates = [p for p in procs if p.get("status") == "candidate"]
                        for p in active:
                            uses = p.get("use_count", 0)
                            print(f"  [{p['id']}] (used {uses}x) {p['problem'][:60]}")
                        print(f"\n  {len(active)} active, {len(candidates)} candidate(s)")
                        if candidates:
                            print(f"  \U0001f4cb {len(candidates)} candidate(s) pending review \u2014 /procedure review")
                    continue

                elif subcmd.startswith("add "):
                    text = subcmd[4:].strip()
                    if "|" in text:
                        problem, solution = text.split("|", 1)
                        from mycoswarm.memory import add_procedure
                        proc = add_procedure(
                            problem=problem.strip(),
                            solution=solution.strip(),
                        )
                        print(f"  Stored: {proc['id']}")
                    else:
                        print("  Format: /procedure add <problem> | <solution>")
                    continue

                elif subcmd.startswith("remove "):
                    proc_id = subcmd[7:].strip()
                    from mycoswarm.memory import remove_procedure
                    if remove_procedure(proc_id):
                        print(f"  Removed: {proc_id}")
                    else:
                        print(f"  Not found: {proc_id}")
                    continue

                elif subcmd.startswith("promote"):
                    # Promote recent lessons to procedures
                    from mycoswarm.memory import load_session_summaries, promote_lesson_to_procedure
                    sessions = load_session_summaries(limit=5)
                    promoted = 0
                    for s in sessions:
                        for lesson in s.get("lessons", []):
                            result = promote_lesson_to_procedure(
                                lesson,
                                session_name=s.get("session_name", ""),
                            )
                            if result:
                                print(f"  Promoted: {result['id']} \u2014 {lesson[:60]}")
                                promoted += 1
                    if promoted == 0:
                        print("  No promotable lessons found in recent sessions.")
                    else:
                        print(f"  Promoted {promoted} lessons to procedures.")
                    continue

                elif subcmd == "review":
                    from mycoswarm.memory import (
                        load_procedure_candidates,
                        approve_procedure,
                        reject_procedure,
                    )
                    candidates = load_procedure_candidates()
                    if not candidates:
                        print("  No procedure candidates to review.")
                        continue

                    print(f"  {len(candidates)} candidate(s) to review:\n")
                    for c in candidates:
                        print(f"  \u250c\u2500 {c['id']} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
                        print(f"  \u2502 Problem:  {c['problem']}")
                        print(f"  \u2502 Solution: {c['solution']}")
                        if c.get("reasoning"):
                            print(f"  \u2502 Why:      {c['reasoning']}")
                        for ap in c.get("anti_patterns", []):
                            print(f"  \u2502 Avoid:    {ap}")
                        if c.get("tags"):
                            print(f"  \u2502 Tags:     {', '.join(c['tags'])}")
                        if c.get("source_lesson"):
                            print(f"  \u2502 Lesson:   {c['source_lesson'][:80]}")
                        print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

                        choice = ""
                        while True:
                            choice = input("  [a]pprove / [r]eject / [s]kip / [q]uit review? ").strip().lower()
                            if choice in ("a", "approve"):
                                if approve_procedure(c["id"]):
                                    print(f"  \u2713 Approved and indexed: {c['id']}")
                                break
                            elif choice in ("r", "reject"):
                                if reject_procedure(c["id"]):
                                    print(f"  \u2717 Rejected: {c['id']}")
                                break
                            elif choice in ("s", "skip"):
                                print(f"  \u2014 Skipped: {c['id']}")
                                break
                            elif choice in ("q", "quit"):
                                print("  Review ended.")
                                break
                            else:
                                print("  Type a/r/s/q")
                        if choice in ("q", "quit"):
                            break
                    continue

                else:
                    print("  /procedure list | add <problem>|<solution> | remove <id> | promote | review")
                    continue

            else:
                print(f"   Unknown command: {cmd}")
                continue

        # --- Refresh memory with semantic session search ---
        refreshed = build_memory_system_prompt(query=user_input)
        if refreshed and messages and messages[0].get("role") == "system":
            id_prompt = build_identity_prompt(identity)
            messages[0] = {"role": "system", "content": id_prompt + "\n\n" + refreshed}

        # --- Agentic classification + tool gathering ---
        tool_context = ""
        tool_sources: list[str] = []
        doc_hits: list[dict] = []
        session_hits: list[dict] = []
        procedure_hits: list[dict] = []

        if auto_tools and len(user_input.split()) >= 5:
            print("   🤔 Classifying...", end="\r", flush=True)

            if daemon_up:
                # Daemon mode: submit intent_classify as distributed task
                import uuid as _uuid
                _ic_id = f"intent-{_uuid.uuid4().hex[:8]}"
                _ic_payload = {
                    "task_id": _ic_id,
                    "task_type": "intent_classify",
                    "payload": {"query": user_input},
                    "source_node": "cli-chat",
                    "priority": 7,
                    "timeout_seconds": 30,
                }
                try:
                    import time as _t
                    _ic_start = _t.time()
                    with httpx.Client(timeout=5) as _ic_client:
                        _ic_client.post(f"{url}/task", json=_ic_payload).raise_for_status()
                    # Poll for result
                    intent_result = None
                    with httpx.Client(timeout=5) as _ic_client:
                        while _t.time() - _ic_start < 15:
                            _t.sleep(0.3)
                            try:
                                _r = _ic_client.get(f"{url}/task/{_ic_id}")
                                _d = _r.json()
                                if _d.get("status") in ("completed", "failed"):
                                    if _d.get("status") == "completed" and _d.get("result"):
                                        intent_result = _d["result"]
                                    break
                            except Exception:
                                pass
                except Exception:
                    intent_result = None

                if intent_result is None:
                    intent_result = {"tool": "answer", "mode": "chat", "scope": "all"}
            else:
                # Solo mode: call intent_classify() directly
                from mycoswarm.solo import intent_classify
                intent_result = intent_classify(user_input)

            # Show intent debug line (replaces "Classifying..." indicator)
            print(f"\r   🤔 intent: {intent_result['tool']}/{intent_result.get('mode', '?')}/{intent_result.get('scope', '?')}", flush=True)

            if debug:
                print(f"🐛 DEBUG: INTENT: {intent_result}", flush=True)

            classification = intent_result["tool"]
            past_ref = intent_result.get("scope") in ("personal", "session")
            # Secondary signal: regex check in case LLM missed it
            if not past_ref:
                from mycoswarm.solo import detect_past_reference
                past_ref = detect_past_reference(user_input)

            need_web = classification in ("web_search", "web_and_rag")
            need_rag = classification in ("rag", "web_and_rag")

            web_context_parts: list[str] = []
            rag_context_parts: list[str] = []

            # --- Web search ---
            if need_web:
                print("   🔍 Searching the web...", end="", flush=True)
                if daemon_up:
                    import uuid as _uuid
                    _sid = f"auto-ws-{_uuid.uuid4().hex[:8]}"
                    ws_data = _do_search(url, user_input, _sid, 5)
                    ws_results = []
                    if ws_data.get("data", {}).get("status") == "completed":
                        ws_results = ws_data["data"].get("result", {}).get("results", [])
                else:
                    from mycoswarm.solo import web_search_solo
                    ws_results = web_search_solo(user_input, max_results=5)

                if ws_results:
                    print(f" {len(ws_results)} results", flush=True)
                    for wi, r in enumerate(ws_results, 1):
                        web_context_parts.append(
                            f"[W{wi}] {r['title']}\n    {r['snippet']}"
                        )
                    tool_sources.append("web")
                else:
                    print(" no results", flush=True)

            # --- RAG search (docs + sessions via search_all with intent) ---
            doc_hits: list[dict] = []
            session_hits: list[dict] = []
            procedure_hits: list[dict] = []
            if need_rag or past_ref:
                scope = (intent_result or {}).get("scope", "all")
                if scope in ("session", "personal"):
                    print("   💭 Searching past conversations...", end="", flush=True)
                elif scope in ("docs", "documents"):
                    print("   📚 Checking your documents...", end="", flush=True)
                else:
                    print("   📚 Searching docs + sessions...", end="", flush=True)
                from mycoswarm.library import search_all
                doc_hits, session_hits, procedure_hits = search_all(
                    user_input, n_results=5, intent=intent_result,
                    do_rerank=False, session_boost=past_ref,
                )
                parts = []
                if doc_hits:
                    parts.append(f"{len(doc_hits)} excerpts")
                if session_hits:
                    parts.append(f"{len(session_hits)} sessions")
                if parts:
                    print(f" {', '.join(parts)}", flush=True)
                else:
                    print(" no results", flush=True)

                if debug:
                    import re as _re
                    _sf = _re.search(r'\b(\w+\.(?:md|txt|py|json|yaml|toml|cfg|csv))\b', user_input, _re.IGNORECASE)
                    print(f"🐛 DEBUG: RETRIEVAL: source_filter={_sf.group(1) if _sf else None} scope={scope} past_ref={past_ref}", flush=True)
                    for _di, _dh in enumerate(doc_hits, 1):
                        print(f"🐛 DEBUG: DOC HIT [{_di}]: source={_dh.get('source')} chunk={_dh.get('chunk_index')} rrf={_dh.get('rrf_score', 'n/a')} text={_dh.get('text', '')[:100]!r}", flush=True)
                    for _si, _sh in enumerate(session_hits, 1):
                        print(f"🐛 DEBUG: SESSION HIT [{_si}]: date={_sh.get('date')} topic={_sh.get('topic', '')} text={_sh.get('summary', '')[:100]!r}", flush=True)

            # --- Procedural retrieval (independent of RAG path) ---
            if not procedure_hits:
                import re as _pre
                _PROBLEM_RE = _pre.compile(
                    r'\b(error|bug|fix|issue|fail|broke|broken|wrong|crash|stuck|slow|missing|'
                    r'unexpected|weird|ignored|uncertain|unsure|confus\w*|not sure|too many|complex|'
                    r'overwhelm\w*|struggl\w*|frustrat\w*|how\s+to|why\s+does|doesn.t\s+work|not\s+working|'
                    r'problem|debug|solve|where\s+do\s+i\s+start|should\s+i)\b',
                    _pre.IGNORECASE,
                )
                if _PROBLEM_RE.search(user_input):
                    try:
                        from mycoswarm.library import search_procedures
                        procedure_hits = search_procedures(user_input, n_results=3)
                    except Exception as e:
                        logger.debug("Procedural retrieval failed: %s", e)

            # --- Format results ---
            for ri, hit in enumerate(doc_hits, 1):
                section = hit.get("section", "untitled")
                rag_context_parts.append(
                    f"[D{ri}] (From: {hit['source']} > {section}) {hit['text']}"
                )
            for si, hit in enumerate(session_hits, 1):
                date = hit.get("date", "unknown")
                topic = hit.get("topic", "")
                label = f"conversation on {date}"
                if topic:
                    label += f" — {topic}"
                rag_context_parts.append(
                    f"[S{si}] (From {label}) {hit['summary']}"
                )

            # Format procedure hits
            if procedure_hits:
                from mycoswarm.memory import format_procedures_for_prompt, reference_procedure
                proc_text = format_procedures_for_prompt(procedure_hits)
                rag_context_parts.append(f"\nRelevant procedures (past solutions):\n{proc_text}")
                for p in procedure_hits:
                    reference_procedure(p.get("id", ""))

            if doc_hits or session_hits or procedure_hits:
                tool_sources.append("docs")

            # --- Build combined context ---
            context_sections: list[str] = []
            if web_context_parts:
                context_sections.append(
                    "WEB SEARCH RESULTS:\n" + "\n\n".join(web_context_parts)
                )
            if rag_context_parts:
                context_sections.append(
                    "RETRIEVED CONTEXT (documents and past conversations):\n"
                    + "\n\n".join(rag_context_parts)
                )
            if context_sections:
                cite_parts: list[str] = []
                if web_context_parts:
                    cite_parts.append("[W1], [W2] for web sources")
                if doc_hits:
                    cite_parts.append("[D1], [D2] for documents")
                if session_hits:
                    cite_parts.append("[S1], [S2] for past conversations")
                if procedure_hits:
                    cite_parts.append("[P1], [P2] for known procedures")
                cite_hint = "Cite using " + " and ".join(cite_parts) + "." if cite_parts else ""
                tool_context = (
                    f"Use the following context to answer. {cite_hint}\n\n"
                    + "\n\n".join(context_sections)
                )

            # Accumulate RAG context for session grounding check
            if rag_context_parts:
                session_rag_context.extend(rag_context_parts)

        # --- Timing gate (always runs, even when auto_tools is off) ---
        _seconds_since_last = None
        if _last_turn_time is not None:
            _seconds_since_last = (_dt_timing.now() - _last_turn_time).total_seconds()
        _timing = evaluate_timing(
            current_time=_dt_timing.now(),
            session_turn_count=_turn_count,
            seconds_since_last_turn=_seconds_since_last,
            user_message_length=len(user_input),
            intent=intent_result,
            frustration_detected=(
                _last_vitals is not None and _last_vitals.compassion < 0.4
            ),
        )
        _last_timing = _timing

        # --- Send message ---
        messages.append({"role": "user", "content": user_input})

        # Swap "no internet" boundary when web results are present
        _send_msgs = list(messages)

        # --- Inject tool context into the user message (not persistent history) ---
        # Merge context into the final user message so the model sees it together.
        # Previous turns' RAG/web results are NOT carried forward.
        if tool_context:
            _send_msgs[-1] = {
                "role": "user",
                "content": tool_context + "\n\nUSER QUESTION: " + _send_msgs[-1]["content"],
            }
        if "web" in tool_sources and _send_msgs and _send_msgs[0].get("role") == "system":
            _no_net = (
                "You are running locally with NO internet access during chat. "
                "You CANNOT look up current weather, news, stock prices, sports "
                "scores, or any real-time information. If asked about something "
                "you're uncertain about or that requires current data, be honest "
                "and say: 'I don't have access to real-time information. You can "
                "try: mycoswarm research <your question> for web-sourced answers.' "
                "Never fabricate current data like weather, prices, or news. "
                "You DO have access to: persistent memory (facts the user has "
                "stored), session history, and your training knowledge."
            )
            _web_aware = (
                "You have web search results available below. "
                "Use them confidently to answer the user's question."
            )
            _send_msgs[0] = {
                **_send_msgs[0],
                "content": _send_msgs[0]["content"].replace(_no_net, _web_aware),
            }

        # --- Inject timing modifier into system prompt ---
        if _timing.prompt_modifier and _send_msgs and _send_msgs[0].get("role") == "system":
            _send_msgs[0] = {
                **_send_msgs[0],
                "content": _send_msgs[0]["content"] + "\n\n" + _timing.prompt_modifier,
            }

        if debug:
            if tool_context:
                print(f"🐛 DEBUG: PROMPT: tool_context ({len(tool_context)} chars):", flush=True)
                for _line in tool_context.split("\n")[:20]:
                    print(f"🐛 DEBUG:   {_line[:200]}", flush=True)
                if tool_context.count("\n") > 20:
                    print(f"🐛 DEBUG:   ... ({tool_context.count(chr(10)) - 20} more lines)", flush=True)
            print(f"🐛 DEBUG: MESSAGES: {len(_send_msgs)} total", flush=True)
            for _mi, _msg in enumerate(_send_msgs):
                _role = _msg.get("role", "?")
                _content = _msg.get("content", "")
                print(f"🐛 DEBUG:   [{_mi}] {_role}: {_content[:200]!r}", flush=True)

        if not daemon_up:
            # Single-node mode — direct to Ollama
            print()
            full_text, metrics = chat_stream(_send_msgs, model)

            if not full_text:
                messages.pop()
                continue

            messages.append({"role": "assistant", "content": full_text})

            tps = metrics.get("tokens_per_second", 0)
            duration = metrics.get("duration_seconds", 0)
            response_tokens = int(tps * duration) if tps and duration else 0
            tools_label = f" | tools: {'+'.join(tool_sources)}" if tool_sources else ""
            print(
                f"\n\n{'─' * 50}\n"
                f"  ⏱  {duration:.1f}s | {tps:.1f} tok/s | {model}{tools_label}"
            )

            # --- Timing indicator ---
            if _timing.mode != TimingMode.PROCEED:
                _t_indicator = _timing.status_indicator()
                _t_reason_text = "; ".join(_timing.reasons[:2])
                print(f"  {_t_indicator} {_timing.mode.value}: {_t_reason_text}")

            # --- Vital signs ---
            from mycoswarm.vitals import compute_vitals
            from mycoswarm.memory import load_facts
            _dont_know = any(
                p in full_text.lower()
                for p in ("i don't know", "i don't recall", "i'm not sure", "i don't have")
            )
            _facts = load_facts()
            # Self-knowledge is grounded in identity
            _grounding = None
            if (intent_result or {}).get("tool", "answer") == "answer" and identity.get("name"):
                if _grounding is None or _grounding == 0:
                    _grounding = 0.7
            # Chat-mode or very short messages: neutral grounding (no alarming alerts)
            if _grounding is None and (
                (intent_result or {}).get("mode") == "chat" or len(user_input) < 10
            ):
                _grounding = 0.6
            _vitals = compute_vitals(
                grounding_score=_grounding,
                source_count=len(doc_hits) + len(session_hits),
                session_hits=len(session_hits),
                doc_hits=len(doc_hits),
                procedure_hits=len(procedure_hits),
                fact_hits=len(_facts),
                intent=intent_result,
                response_tokens=response_tokens,
                said_dont_know=_dont_know,
            )
            _alerts = _vitals.alerts()
            for _a in _alerts:
                print(f"  💭 {_a}")
            print(f"  {_vitals.status_bar()}")
            _last_vitals = _vitals

            # --- Update turn tracking ---
            _last_turn_time = _dt_timing.now()
            _turn_count += 1
            continue

        # Daemon mode — submit via API
        import uuid

        task_id = f"chat-{uuid.uuid4().hex[:8]}"
        task_payload = {
            "task_id": task_id,
            "task_type": "inference",
            "payload": {
                "model": model,
                "messages": _send_msgs,
            },
            "source_node": "cli-chat",
            "priority": 5,
            "timeout_seconds": 300,
        }

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(f"{url}/task", json=task_payload)
                resp.raise_for_status()
                submit_data = resp.json()
        except httpx.ConnectError:
            print("❌ Lost connection to daemon.")
            messages.pop()
            continue
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            print(f"❌ {detail}")
            messages.pop()
            continue

        # Stream directly from target node for remote tasks
        target_ip = submit_data.get("target_ip")
        target_port = submit_data.get("target_port")
        if target_ip and target_port:
            stream_url = f"http://{target_ip}:{target_port}"
        else:
            stream_url = url

        # Stream tokens live
        print()  # newline before response
        full_text, metrics = _stream_response(stream_url, task_id)

        if not full_text:
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": full_text})

        tps = metrics.get("tokens_per_second", 0)
        duration = metrics.get("duration_seconds", 0)
        response_tokens = int(tps * duration) if tps and duration else 0
        node_id = metrics.get("node_id", "")
        tools_label = f" | tools: {'+'.join(tool_sources)}" if tool_sources else ""
        print(
            f"\n\n{'─' * 50}\n"
            f"  ⏱  {duration:.1f}s | {tps:.1f} tok/s | "
            f"{model} | node: {node_id}{tools_label}"
        )

        # --- Timing indicator ---
        if _timing.mode != TimingMode.PROCEED:
            _t_indicator = _timing.status_indicator()
            _t_reason_text = "; ".join(_timing.reasons[:2])
            print(f"  {_t_indicator} {_timing.mode.value}: {_t_reason_text}")

        # --- Vital signs ---
        from mycoswarm.vitals import compute_vitals
        from mycoswarm.memory import load_facts
        _dont_know = any(
            p in full_text.lower()
            for p in ("i don't know", "i don't recall", "i'm not sure", "i don't have")
        )
        _facts = load_facts()
        # Self-knowledge is grounded in identity
        _grounding = None
        if (intent_result or {}).get("tool", "answer") == "answer" and identity.get("name"):
            if _grounding is None or _grounding == 0:
                _grounding = 0.7
        # Chat-mode or very short messages: neutral grounding (no alarming alerts)
        if _grounding is None and (
            (intent_result or {}).get("mode") == "chat" or len(user_input) < 10
        ):
            _grounding = 0.6
        _vitals = compute_vitals(
            grounding_score=_grounding,
            source_count=len(doc_hits) + len(session_hits),
            session_hits=len(session_hits),
            doc_hits=len(doc_hits),
            procedure_hits=len(procedure_hits),
            fact_hits=len(_facts),
            intent=intent_result,
            response_tokens=response_tokens,
            said_dont_know=_dont_know,
        )
        _alerts = _vitals.alerts()
        for _a in _alerts:
            print(f"  💭 {_a}")
        print(f"  {_vitals.status_bar()}")
        _last_vitals = _vitals

        # --- Update turn tracking ---
        _last_turn_time = _dt_timing.now()
        _turn_count += 1


def cmd_library(args):
    """Manage the document library (ingest, search, list, remove, reindex, reindex-sessions, reindex-procedures, auto-update, eval)."""
    from pathlib import Path
    from mycoswarm.library import (
        ingest_file, ingest_directory, search, list_documents, remove_document,
        reindex, reindex_sessions, auto_update, LIBRARY_DIR,
    )

    action = args.action

    if action == "ingest":
        path = Path(args.path).expanduser() if args.path else None

        if path and path.is_file():
            print(f"📄 Ingesting: {path.name}")
            result = ingest_file(path, model=args.model)
            if result.get("skipped"):
                print(f"  ⏭  Skipped (unsupported extension)")
            else:
                print(f"  ✅ {result['chunks']} chunks indexed (model: {result['model']})")
        else:
            target = path if path else LIBRARY_DIR
            print(f"📚 Ingesting directory: {target}")
            results = ingest_directory(path, model=args.model)
            if not results:
                print(f"  No supported files found in {target}")
                print(f"  Supported: .pdf .txt .md .html .csv .json")
                return
            total_chunks = 0
            for r in results:
                if r.get("skipped"):
                    print(f"  ⏭  {r['file']} (unsupported)")
                else:
                    print(f"  ✅ {r['file']} — {r['chunks']} chunks")
                    total_chunks += r["chunks"]
            print(f"\n  📊 {len(results)} file(s), {total_chunks} total chunks")

    elif action == "search":
        query = args.query
        if not query:
            print("❌ Usage: mycoswarm library search --query \"your question\"")
            return
        print(f"🔍 Searching: {query}\n")
        results = search(query, n_results=5, model=args.model)
        if not results:
            print("  No results found. Have you ingested documents?")
            print(f"  Try: mycoswarm library ingest")
            return
        for i, hit in enumerate(results, 1):
            score_pct = max(0, (1 - hit["score"]) * 100)
            print(f"  [{i}] ({hit['source']}) — {score_pct:.0f}% match")
            preview = hit["text"][:200].replace("\n", " ")
            print(f"      {preview}...")
            print()

    elif action == "list":
        docs = list_documents()
        if not docs:
            print("📚 No documents indexed.")
            print(f"   Add files to {LIBRARY_DIR} then run: mycoswarm library ingest")
            return
        print("📚 Indexed Documents")
        print("=" * 40)
        total = 0
        for d in docs:
            print(f"  {d['file']}  ({d['chunks']} chunks)")
            total += d["chunks"]
        print(f"\n  {len(docs)} document(s), {total} total chunks")

    elif action == "remove":
        filename = args.path
        if not filename:
            print("❌ Usage: mycoswarm library remove <filename>")
            return
        if remove_document(filename):
            print(f"✅ Removed: {filename}")
        else:
            print(f"❌ Not found in index: {filename}")

    elif action == "reindex":
        target = Path(args.path).expanduser() if args.path else LIBRARY_DIR
        print(f"🔄 Dropping all chunks and re-indexing from {target}...")
        results = reindex(model=args.model, path=target)
        if not results:
            print(f"  No supported files found in {target}")
            return
        total_chunks = 0
        for r in results:
            if r.get("skipped"):
                print(f"  ⏭  {r['file']} (unsupported)")
            else:
                print(f"  ✅ {r['file']} — {r['chunks']} chunks")
                total_chunks += r["chunks"]
        print(f"\n  📊 {len(results)} file(s), {total_chunks} total chunks (reindexed)")

    elif action == "reindex-sessions":
        print("🔄 Dropping session_memory and re-indexing from sessions.jsonl...")
        stats = reindex_sessions(model=args.model)
        print(f"  📊 {stats['sessions']} session(s), {stats['topics']} topic chunk(s) indexed")
        if stats["failed"]:
            print(f"  ⚠️  {stats['failed']} chunk(s) failed to embed")

    elif action == "reindex-procedures":
        from mycoswarm.library import reindex_procedures
        print("🔄 Dropping procedural_memory and re-indexing from procedures.jsonl...")
        stats = reindex_procedures(model=args.model)
        print(f"  Reindexed: {stats['indexed']}/{stats['procedures']} procedures")
        if stats["failed"]:
            print(f"  Failed: {stats['failed']}")

    elif action == "auto-update":
        target = Path(args.path).expanduser() if args.path else LIBRARY_DIR
        print(f"🔄 Checking for changes in {target}...")
        result = auto_update(docs_dir=target, model=args.model)
        if not result["updated"] and not result["added"] and not result["removed"]:
            print("  ✅ Everything up to date.")
        else:
            for name in result["updated"]:
                print(f"  🔄 Updated: {name}")
            for name in result["added"]:
                print(f"  ➕ Added: {name}")
            for name in result["removed"]:
                print(f"  🗑  Removed: {name}")
            total = len(result["updated"]) + len(result["added"]) + len(result["removed"])
            print(f"\n  📊 {total} change(s) applied")

    elif action == "eval":
        from mycoswarm.rag_eval import run_eval, save_results, print_results, load_previous_results

        previous = load_previous_results()
        print("🔬 Running RAG evaluation...")
        results = run_eval(model=args.model, verbose=True)
        save_results(results)
        print_results(results, previous)


def cmd_rag(args):
    """Ask a question with document context via RAG."""
    from mycoswarm.solo import check_daemon, check_ollama, pick_model, chat_stream, detect_past_reference
    from mycoswarm.library import search_all

    question = " ".join(args.question)
    do_rerank = not getattr(args, "no_rerank", False)
    past_ref = detect_past_reference(question)

    # Search both document library and session memory
    if past_ref:
        print(f"💭 Searching past conversations...", end=" ", flush=True)
    elif do_rerank:
        print(f"🔍 Searching + re-ranking...", end=" ", flush=True)
    else:
        print(f"🔍 Searching library...", end=" ", flush=True)
    doc_hits, session_hits, procedure_hits = search_all(
        question, n_results=5, do_rerank=do_rerank, session_boost=past_ref,
    )

    if not doc_hits and not session_hits:
        print("no results.")
        print("   No documents or sessions indexed. Try: mycoswarm library ingest")
        return

    total = len(doc_hits) + len(session_hits) + len(procedure_hits)
    print(f"{total} result(s) found ({len(doc_hits)} docs, {len(session_hits)} sessions).\n")

    # Build context from hits with distinct labels
    context_parts: list[str] = []
    sources: list[str] = []
    seen_sources: set[str] = set()
    for i, hit in enumerate(doc_hits, 1):
        section = hit.get("section", "untitled")
        context_parts.append(f"[D{i}] (From: {hit['source']} > {section}) {hit['text']}")
        if hit["source"] not in seen_sources:
            sources.append(hit["source"])
            seen_sources.add(hit["source"])
    for i, hit in enumerate(session_hits, 1):
        date = hit.get("date", "unknown")
        topic = hit.get("topic", "")
        label = f"conversation on {date}"
        if topic:
            label += f" — {topic}"
        context_parts.append(f"[S{i}] (From {label}) {hit['summary']}")
    if session_hits:
        sources.append("sessions")
    if procedure_hits:
        from mycoswarm.memory import format_procedures_for_prompt, reference_procedure
        proc_text = format_procedures_for_prompt(procedure_hits)
        context_parts.append(f"\nRelevant procedures (past solutions):\n{proc_text}")
        for p in procedure_hits:
            reference_procedure(p.get("id", ""))
        sources.append("procedures")

    context_block = "\n\n".join(context_parts)
    cite_parts = []
    if doc_hits:
        cite_parts.append("[D1], [D2] for documents")
    if session_hits:
        cite_parts.append("[S1], [S2] for past conversations")
    if procedure_hits:
        cite_parts.append("[P1], [P2] for known procedures")
    cite_hint = "Cite using " + " and ".join(cite_parts) + "." if cite_parts else ""
    system_prompt = (
        "IMPORTANT: The excerpts below are REAL content retrieved from "
        "the user's actual files and past conversations. Use them as "
        "your primary source of truth. Quote specific details from the "
        "excerpts. Do NOT say you cannot access files — the content is "
        f"provided below. {cite_hint}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Route: daemon first, then direct Ollama
    daemon_up = check_daemon(args.port)

    if daemon_up:
        import uuid

        profile = detect_all()
        ip = profile.lan_ip or "localhost"
        url = f"http://{ip}:{args.port}"
        model = _discover_model(url, args.model)

        task_id = f"rag-{uuid.uuid4().hex[:8]}"
        task_payload = {
            "task_id": task_id,
            "task_type": "inference",
            "payload": {
                "model": model,
                "messages": messages,
            },
            "source_node": "cli-rag",
            "priority": 5,
            "timeout_seconds": 300,
        }

        print(f"🧠 Asking {model}...\n")
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(f"{url}/task", json=task_payload)
                resp.raise_for_status()
                submit_data = resp.json()
        except httpx.ConnectError:
            print("❌ Lost connection to daemon.")
            return

        target_ip = submit_data.get("target_ip")
        target_port = submit_data.get("target_port")
        stream_url = f"http://{target_ip}:{target_port}" if target_ip and target_port else url

        full_text, metrics = _stream_response(stream_url, task_id)
    else:
        running, models = check_ollama()
        if not running:
            print("❌ No daemon running and Ollama is not reachable.")
            sys.exit(1)

        model = pick_model(models, args.model)
        print(f"🧠 Asking {model}...\n")
        full_text, metrics = chat_stream(messages, model)

    if not full_text:
        return

    # Footer with sources
    tps = metrics.get("tokens_per_second", 0)
    duration = metrics.get("duration_seconds", 0)
    print(f"\n\n{'─' * 50}")
    print(f"📚 Sources: {', '.join(sources)}")
    print(f"  ⏱  {duration:.1f}s | {tps:.1f} tok/s | {model}")


def cmd_memory(args):
    """Manage persistent memory (facts)."""
    from mycoswarm.memory import load_facts, add_fact, remove_fact

    if args.add:
        fact = add_fact(args.add)
        print(f"✅ Added #{fact['id']}: {fact['text']}")
        return

    if args.forget is not None:
        if remove_fact(args.forget):
            print(f"✅ Forgot #{args.forget}.")
        else:
            print(f"❌ Fact #{args.forget} not found.")
        return

    # Default: list all facts
    facts = load_facts()
    if not facts:
        print("🧠 No facts stored.")
        print("   Add one with: mycoswarm memory --add \"your fact\"")
        return

    print("🧠 mycoSwarm Memory — Known Facts")
    print("=" * 40)
    for f in facts:
        added = f.get("added", "")[:10]
        print(f"  #{f['id']}: {f['text']}  ({added})")
    print(f"\n  {len(facts)} fact(s). Forget with: mycoswarm memory --forget <number>")


def main():
    parser = argparse.ArgumentParser(
        prog="mycoswarm",
        description="🍄 mycoSwarm — Distributed AI framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    # detect
    detect_parser = subparsers.add_parser(
        "detect", help="Detect hardware and classify capabilities"
    )
    detect_parser.add_argument("--json", action="store_true", help="Output as JSON")
    detect_parser.set_defaults(func=cmd_detect)

    # identity
    identity_parser = subparsers.add_parser(
        "identity", help="Show full node identity announcement"
    )
    identity_parser.set_defaults(func=cmd_identity)

    # daemon
    daemon_parser = subparsers.add_parser(
        "daemon", help="Start the node daemon (announce + discover peers)"
    )
    daemon_parser.add_argument(
        "--port", type=int, default=7890, help="API port (default: 7890)"
    )
    daemon_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    daemon_parser.set_defaults(func=cmd_daemon)

    # dashboard
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Start the web dashboard"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8080, help="Dashboard port (default: 8080)"
    )
    dashboard_parser.add_argument(
        "--daemon-port", type=int, default=7890,
        help="Daemon API port to query (default: 7890)",
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # swarm
    swarm_parser = subparsers.add_parser(
        "swarm", help="Show swarm status"
    )
    swarm_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    swarm_parser.set_defaults(func=cmd_swarm)

    # ping
    ping_parser = subparsers.add_parser(
        "ping", help="Ping all discovered peers"
    )
    ping_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    ping_parser.set_defaults(func=cmd_ping)

    # ask
    ask_parser = subparsers.add_parser(
        "ask", help="Send a prompt to the swarm for inference"
    )
    ask_parser.add_argument("prompt", nargs="+", help="The prompt text")
    ask_parser.add_argument(
        "--model", type=str, default=None, help="Ollama model name"
    )
    ask_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    ask_parser.set_defaults(func=cmd_ask)

    # search
    search_parser = subparsers.add_parser(
        "search", help="Search the web via the swarm"
    )
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument(
        "-n", "--max-results", type=int, default=5,
        help="Number of results (default: 5, max: 20)",
    )
    search_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    search_parser.set_defaults(func=cmd_search)

    # research
    research_parser = subparsers.add_parser(
        "research", help="Search + synthesize (CPU search → GPU think)"
    )
    research_parser.add_argument("query", nargs="+", help="Research question")
    research_parser.add_argument(
        "--model", type=str, default=None, help="Ollama model name"
    )
    research_parser.add_argument(
        "-n", "--max-results", type=int, default=5,
        help="Number of search results to feed to LLM (default: 5)",
    )
    research_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    research_parser.set_defaults(func=cmd_research)

    # models
    models_parser = subparsers.add_parser(
        "models", help="Show all models available across the swarm"
    )
    models_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    models_parser.set_defaults(func=cmd_models)

    # plugins
    plugins_parser = subparsers.add_parser(
        "plugins", help="List installed plugins and their status"
    )
    plugins_parser.set_defaults(func=cmd_plugins)

    # chat
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat with the swarm"
    )
    chat_parser.add_argument(
        "--model", type=str, default=None, help="Ollama model name"
    )
    chat_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    chat_parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume the most recent chat session",
    )
    chat_parser.add_argument(
        "--session", type=str, default=None,
        help="Resume a named session (or start new with that name)",
    )
    chat_parser.add_argument(
        "--list", dest="list_sessions", action="store_true", default=False,
        help="List saved chat sessions",
    )
    chat_parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Show full processing pipeline debug output",
    )
    chat_parser.set_defaults(func=cmd_chat)

    # library
    library_parser = subparsers.add_parser(
        "library", help="Manage the document library (ingest, search, list, remove, reindex, reindex-sessions, reindex-procedures, auto-update, eval)"
    )
    library_parser.add_argument(
        "action", choices=["ingest", "search", "list", "remove", "reindex", "reindex-sessions", "reindex-procedures", "auto-update", "eval"],
        help="Action to perform",
    )
    library_parser.add_argument(
        "path", nargs="?", default=None,
        help="File or directory path (for ingest) or filename (for remove)",
    )
    library_parser.add_argument(
        "--query", type=str, default=None,
        help="Search query (for search action)",
    )
    library_parser.add_argument(
        "--model", type=str, default=None,
        help="Embedding model override",
    )
    library_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    library_parser.set_defaults(func=cmd_library)

    # rag
    rag_parser = subparsers.add_parser(
        "rag", help="Ask a question with document context (RAG)"
    )
    rag_parser.add_argument("question", nargs="+", help="Your question")
    rag_parser.add_argument(
        "--model", type=str, default=None, help="Inference model"
    )
    rag_parser.add_argument(
        "--no-rerank", action="store_true", default=False,
        help="Skip LLM re-ranking of retrieved chunks",
    )
    rag_parser.add_argument(
        "--port", type=int, default=7890, help="Local daemon port"
    )
    rag_parser.set_defaults(func=cmd_rag)

    # memory
    memory_parser = subparsers.add_parser(
        "memory", help="Manage persistent memory (facts about the user)"
    )
    memory_parser.add_argument(
        "--add", type=str, default=None,
        help="Add a fact (e.g. --add \"I teach Tai Chi\")",
    )
    memory_parser.add_argument(
        "--forget", type=int, default=None,
        help="Remove a fact by ID (e.g. --forget 1)",
    )
    memory_parser.set_defaults(func=cmd_memory)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
