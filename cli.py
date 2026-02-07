"""mycoSwarm CLI.

Usage:
    mycoswarm detect              Show this node's hardware and capabilities
    mycoswarm detect --json       Output as JSON
    mycoswarm identity            Show full node identity announcement
    mycoswarm daemon              Start the node daemon (announce + discover)
    mycoswarm daemon --port 7890  Start on a specific port
    mycoswarm daemon -v           Verbose logging
"""

import argparse
import sys

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

    # Human-readable output
    print("🍄 mycoSwarm Node Detection")
    print("=" * 50)

    # Hostname & Network
    print(f"\n📍 Host: {profile.hostname}")
    if profile.lan_ip:
        print(f"   LAN:  {profile.lan_ip}")

    # CPU
    if profile.cpu:
        print(f"\n🔧 CPU: {profile.cpu.model}")
        print(
            f"   Cores: {profile.cpu.cores_physical}P / "
            f"{profile.cpu.cores_logical}L @ {profile.cpu.frequency_mhz:.0f} MHz"
        )

    # Memory
    if profile.memory:
        print(
            f"\n💾 RAM: {profile.memory.total_mb:,} MB total, "
            f"{profile.memory.available_mb:,} MB available "
            f"({profile.memory.percent_used:.0f}% used)"
        )

    # GPU
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

    # Disk
    if profile.disks:
        print("\n💿 Disk:")
        for disk in profile.disks:
            print(
                f"   {disk.path}: {disk.free_gb:.0f} GB free "
                f"/ {disk.total_gb:.0f} GB ({disk.percent_used:.0f}% used)"
            )

    # Ollama
    if profile.ollama_running:
        print(f"\n🦙 Ollama: running ({len(profile.ollama_models)} models)")
        for m in profile.ollama_models:
            quant = f" ({m.quantization})" if m.quantization else ""
            print(f"   • {m.name} [{m.parameter_size}{quant}] {m.size_mb:,} MB")
    else:
        print("\n🦙 Ollama: not detected")

    # Classification
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
    detect_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
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
        "--port",
        type=int,
        default=7890,
        help="API port (default: 7890)",
    )
    daemon_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    daemon_parser.set_defaults(func=cmd_daemon)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
