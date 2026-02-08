"""mycoSwarm CLI.

Usage:
    mycoswarm detect              Show this node's hardware and capabilities
    mycoswarm detect --json       Output as JSON
    mycoswarm identity            Show full node identity announcement
    mycoswarm daemon              Start the node daemon (announce + discover)
    mycoswarm daemon --port 7890  Start on a specific port
    mycoswarm daemon -v           Verbose logging
    mycoswarm swarm               Show swarm status (query local daemon)
    mycoswarm ping                Ping all known peers
    mycoswarm ask "your prompt"   Send a prompt for inference
    mycoswarm search "query"      Search the web via the swarm
    mycoswarm research "query"    Search + synthesize (CPU search → GPU think)
    mycoswarm models              Show all models available across the swarm
    mycoswarm chat                Interactive chat with the swarm
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
    import uuid

    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"
    prompt = " ".join(args.prompt)

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


def cmd_research(args):
    """Search the web then synthesize results via LLM inference.

    The swarm's signature move: CPU workers search, GPU nodes think.
    """
    import uuid

    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"
    query = " ".join(args.query)
    model = _discover_model(url, args.model)

    # --- Phase 1: Web search (CPU worker) ---
    search_id = f"research-search-{uuid.uuid4().hex[:8]}"
    search_payload = {
        "task_id": search_id,
        "task_type": "web_search",
        "payload": {"query": query, "max_results": args.max_results},
        "source_node": "cli",
        "priority": 7,
        "timeout_seconds": 60,
    }

    print(f"🔍 Searching: {query}")

    # Submit and get routing info
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{url}/task", json=search_payload)
            resp.raise_for_status()
            submit_data = resp.json()
    except httpx.ConnectError:
        print("❌ Daemon not running. Start it with: mycoswarm daemon")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        print(f"❌ {detail}")
        sys.exit(1)

    search_node = submit_data.get("message", "")
    if "Routed to" in search_node:
        print(f"   {search_node}...")
    else:
        print(f"   Searching locally...")

    # Poll for search results
    import time
    start = time.time()
    search_data = None
    with httpx.Client(timeout=5) as client:
        while time.time() - start < 60:
            time.sleep(0.5)
            try:
                r = client.get(f"{url}/task/{search_id}")
                data = r.json()
                if data.get("status") in ("completed", "failed"):
                    search_data = data
                    break
            except Exception:
                pass

    if search_data is None:
        print("❌ Search timed out.")
        sys.exit(1)
    if search_data.get("status") == "failed":
        print(f"❌ Search failed: {search_data.get('error', 'unknown')}")
        sys.exit(1)

    results = search_data.get("result", {}).get("results", [])
    if not results:
        print("❌ No search results found.")
        sys.exit(1)

    search_duration = search_data.get("duration_seconds", 0)
    search_node_id = search_data.get("node_id", "local")
    print(f"   Found {len(results)} results ({search_duration:.1f}s, node: {search_node_id})")

    # --- Phase 2: Inference with context (GPU node) ---
    # Build context block from search results
    context_lines = []
    sources = []
    for i, r in enumerate(results, 1):
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

    print(f"\n🧠 Thinking with {model}...")

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{url}/task", json=infer_payload)
            resp.raise_for_status()
            infer_submit = resp.json()
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
        f"  🔍 search: {search_duration:.1f}s ({search_node_id}) | "
        f"🧠 inference: {duration:.1f}s {tps:.1f} tok/s ({node_id}) | "
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


def cmd_chat(args):
    """Interactive chat with the swarm."""
    import uuid

    profile = detect_all()
    ip = profile.lan_ip or "localhost"
    url = f"http://{ip}:{args.port}"
    model = _discover_model(url, args.model)
    messages: list[dict[str, str]] = []

    print("🍄 mycoSwarm Chat")
    print(f"   Model: {model}")
    print("   /model to switch, /peers to show swarm, /clear to reset, /quit to exit")
    print(f"{'─' * 50}")

    while True:
        try:
            user_input = input("\n🍄> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye. 🍄")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print("Bye. 🍄")
                break

            elif cmd == "/clear":
                messages.clear()
                print("   Conversation cleared.")
                continue

            elif cmd == "/model":
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    model = parts[1]
                    print(f"   Model → {model}")
                else:
                    all_models = _list_swarm_models(url)
                    if all_models:
                        print("   Available models:")
                        for m in all_models:
                            marker = " ◀" if m == model else ""
                            print(f"     • {m}{marker}")
                    else:
                        print("   No models found.")
                continue

            elif cmd == "/peers":
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

            else:
                print(f"   Unknown command: {cmd}")
                continue

        # --- Send message ---
        messages.append({"role": "user", "content": user_input})

        task_id = f"chat-{uuid.uuid4().hex[:8]}"
        task_payload = {
            "task_id": task_id,
            "task_type": "inference",
            "payload": {
                "model": model,
                "messages": list(messages),
            },
            "source_node": "cli-chat",
            "priority": 5,
            "timeout_seconds": 300,
        }

        # Submit task
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(f"{url}/task", json=task_payload)
                resp.raise_for_status()
                submit_data = resp.json()
        except httpx.ConnectError:
            print("❌ Daemon not running. Start it with: mycoswarm daemon")
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
        node_id = metrics.get("node_id", "")
        print(
            f"\n\n{'─' * 50}\n"
            f"  ⏱  {duration:.1f}s | {tps:.1f} tok/s | "
            f"{model} | node: {node_id}"
        )


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
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
