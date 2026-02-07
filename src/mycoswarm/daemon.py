"""mycoSwarm node daemon.

Runs on every machine in the swarm. Detects hardware, announces
capabilities via mDNS, listens for peers, and periodically
refreshes status.

Usage:
    mycoswarm daemon              Start the daemon
    mycoswarm daemon --port 7890  Start on a specific port
"""

import asyncio
import logging
import signal
import sys

from mycoswarm.hardware import detect_all
from mycoswarm.capabilities import classify_node
from mycoswarm.node import build_identity, NodeIdentity
from mycoswarm.discovery import Discovery, PeerRegistry, DEFAULT_PORT

logger = logging.getLogger(__name__)

# How often to refresh hardware status (seconds)
STATUS_REFRESH_INTERVAL = 30


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )


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
  Announcing on LAN... (Ctrl+C to stop)
""")


async def _status_refresh_loop(
    discovery: Discovery,
    registry: PeerRegistry,
    interval: int = STATUS_REFRESH_INTERVAL,
):
    """Periodically refresh hardware status and update announcement."""
    while True:
        await asyncio.sleep(interval)
        try:
            profile = detect_all()
            caps = classify_node(profile)
            identity = build_identity(profile, caps)
            await discovery.update_identity(identity)

            peers = await registry.get_all()
            active = [p for p in peers if not p.is_stale]
            logger.debug(
                f"Status refresh: {len(active)} active peer(s), "
                f"VRAM free: {identity.vram_free_mb}MB"
            )
        except Exception as e:
            logger.error(f"Status refresh failed: {e}")


async def _peer_logger(event: str, peer):
    """Default callback for peer changes — just log them."""
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

    # Detect hardware and build identity
    logger.info("Detecting hardware...")
    profile = detect_all()
    caps = classify_node(profile)
    identity = build_identity(profile, caps)

    if not identity.lan_ip:
        print("❌ No LAN IP detected. Is your network connected?")
        sys.exit(1)

    _print_banner(identity, port)

    # Set up peer registry and discovery
    registry = PeerRegistry()
    registry.on_change(_peer_logger)

    discovery = Discovery(identity, registry, port=port)

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

    # Start status refresh loop
    refresh_task = asyncio.create_task(
        _status_refresh_loop(discovery, registry)
    )

    # Wait for shutdown signal
    await stop_event.wait()

    # Cleanup
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass

    await discovery.stop()

    peers = await registry.get_all()
    print(f"  Saw {registry.count} peer(s) during session")
    print("  Goodbye. 🍄")


def start_daemon(port: int = DEFAULT_PORT, verbose: bool = False):
    """Entry point for CLI."""
    asyncio.run(run_daemon(port=port, verbose=verbose))
