"""mycoSwarm node daemon.

Runs on every machine in the swarm. Detects hardware, announces
capabilities via mDNS, listens for peers, serves the node API,
executes tasks from the queue, and periodically refreshes status.

Usage:
    mycoswarm daemon              Start the daemon
    mycoswarm daemon --port 7890  Start on a specific port
"""

import asyncio
import logging
import os
import signal
import sys
import time

import httpx
import uvicorn

from mycoswarm.hardware import detect_all
from mycoswarm.capabilities import classify_node
from mycoswarm.node import build_identity, NodeIdentity
from mycoswarm.discovery import Discovery, PeerRegistry, DEFAULT_PORT
from mycoswarm.api import create_api, TaskQueue
from mycoswarm.worker import TaskWorker, HANDLERS
from mycoswarm.orchestrator import Orchestrator
from mycoswarm.plugins import discover_plugins, register_plugins

logger = logging.getLogger(__name__)

STATUS_REFRESH_INTERVAL = 30


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _print_banner(identity: NodeIdentity, port: int):
    print(f"""
🍄 mycoSwarm Daemon
{'=' * 50}
  Node ID:  {identity.node_id}
  Host:     {identity.hostname}
  LAN IP:   {identity.lan_ip}
  Port:     {port}
  Tier:     {identity.node_tier.upper()}
  GPU:      {identity.gpu_name or 'None'}
  VRAM:     {identity.vram_total_mb} MB
  Caps:     {', '.join(identity.capabilities)}
  Models:   {len(identity.available_models)}
{'=' * 50}
  API:      http://{identity.lan_ip}:{port}
  Health:   http://{identity.lan_ip}:{port}/health
  Status:   http://{identity.lan_ip}:{port}/status
  Peers:    http://{identity.lan_ip}:{port}/peers
{'=' * 50}
  Announcing on LAN... (Ctrl+C to stop)
""")


async def _status_refresh_loop(
    discovery: Discovery,
    registry: PeerRegistry,
    interval: int = STATUS_REFRESH_INTERVAL,
):
    """Periodically refresh hardware status, ping peers, and update announcement."""
    while True:
        await asyncio.sleep(interval)
        try:
            profile = detect_all()
            caps = classify_node(profile)
            identity = build_identity(profile, caps)
            await discovery.update_identity(identity)

            # Ping all known peers — keeps last_seen fresh between mDNS events
            peers = await registry.get_all()
            async with httpx.AsyncClient(timeout=5.0) as client:
                for peer in peers:
                    try:
                        resp = await client.get(
                            f"http://{peer.ip}:{peer.port}/health"
                        )
                        if resp.status_code == 200:
                            registry.record_success(peer.node_id)
                    except (httpx.ConnectError, httpx.TimeoutException):
                        registry.record_failure(peer.node_id)

            active = [p for p in peers if not p.is_stale]
            logger.debug(
                f"Status refresh: {len(active)} active peer(s), "
                f"VRAM free: {identity.vram_free_mb}MB"
            )
        except Exception as e:
            logger.error(f"Status refresh failed: {e}")


async def _peer_logger(event: str, peer):
    """Default callback for peer changes."""
    if event == "added":
        gpu_info = f" [{peer.gpu_name}]" if peer.gpu_name else ""
        print(
            f"  🟢 Node joined: {peer.hostname} ({peer.ip})"
            f" [{peer.node_tier}]{gpu_info}"
        )
    elif event == "removed":
        print(f"  🔴 Node left: {peer.hostname} ({peer.ip})")
    elif event == "updated":
        logger.debug(f"Peer updated: {peer.hostname}")


async def run_daemon(port: int = DEFAULT_PORT, verbose: bool = False):
    """Main daemon loop."""
    _setup_logging(verbose)
    start_time = time.time()

    logger.info("Detecting hardware...")
    profile = detect_all()
    caps = classify_node(profile)
    identity = build_identity(profile, caps)

    if not identity.lan_ip:
        print("❌ No LAN IP detected. Is your network connected?")
        sys.exit(1)

    # Load plugins before announcing capabilities
    plugins = discover_plugins()
    registered = register_plugins(plugins, HANDLERS, identity.capabilities)
    if registered:
        logger.info(f"🔌 {len(registered)} plugin(s) loaded")

    # Auto-update document index on startup (disabled by default)
    # To enable: set MYCOSWARM_AUTO_UPDATE=1 in environment
    if os.environ.get("MYCOSWARM_AUTO_UPDATE") == "1":
        try:
            from mycoswarm.library import auto_update
            result = auto_update()
            changes = len(result["updated"]) + len(result["added"]) + len(result["removed"])
            if changes:
                logger.info(f"📚 Auto-update: {changes} document change(s) applied")
        except Exception as e:
            logger.warning(f"📚 Auto-update failed: {e}")

    _print_banner(identity, port)

    # Set up components
    registry = PeerRegistry()
    registry.on_change(_peer_logger)
    task_queue = TaskQueue()
    discovery = Discovery(identity, registry, port=port)
    worker = TaskWorker(task_queue, identity.node_id)
    orchestrator = Orchestrator(identity, registry)

    # Create FastAPI app
    app = create_api(identity, registry, task_queue, start_time, orchestrator, port=port)

    # Handle shutdown
    stop_event = asyncio.Event()

    def _signal_handler():
        print("\n🛑 Shutting down...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Start discovery
    await discovery.start()

    # Start API server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning" if not verbose else "info",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Start task worker
    worker_task = asyncio.create_task(worker.run())

    # Start status refresh
    refresh_task = asyncio.create_task(
        _status_refresh_loop(discovery, registry)
    )

    logger.info(f"🌐 API listening on http://{identity.lan_ip}:{port}")
    logger.info("👷 Task worker running")
    logger.info("🎯 Orchestrator active — tasks route to best node")

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    worker.stop()
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    server.should_exit = True
    await server_task
    await orchestrator.close()
    await discovery.stop()

    stats = worker.stats
    print(
        f"  Tasks: {stats['tasks_completed']} completed, "
        f"{stats['tasks_failed']} failed"
    )
    print(f"  Saw {registry.count} peer(s) during session")
    print("  Goodbye. 🍄")


def start_daemon(port: int = DEFAULT_PORT, verbose: bool = False):
    """Entry point for CLI."""
    asyncio.run(run_daemon(port=port, verbose=verbose))
