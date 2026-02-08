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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
