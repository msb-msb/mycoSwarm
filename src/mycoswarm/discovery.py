"""mDNS discovery for mycoSwarm nodes.

Uses zeroconf to announce this node and discover peers on the LAN.
Each node registers as a _mycoswarm._tcp.local. service with its
capabilities encoded in TXT records.

No configuration required — just start the daemon and it finds peers.
"""

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field

from zeroconf import IPVersion, Zeroconf, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser

from mycoswarm.node import NodeIdentity

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_mycoswarm._tcp.local."
DEFAULT_PORT = 7890  # mycoSwarm API port


def _all_lan_addresses() -> list[bytes]:
    """Return packed IPv4 addresses for all non-loopback interfaces."""
    import psutil

    addrs: list[bytes] = []
    for _iface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == socket.AF_INET and not snic.address.startswith("127."):
                addrs.append(socket.inet_aton(snic.address))
    return addrs


@dataclass
class Peer:
    """A discovered peer node."""

    node_id: str
    hostname: str
    ip: str
    port: int
    node_tier: str
    capabilities: list[str]
    gpu_tier: str
    gpu_name: str | None
    vram_total_mb: int
    available_models: list[str]
    version: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_seen

    @property
    def is_stale(self) -> bool:
        """Peer hasn't been seen in over 300 seconds."""
        return self.age_seconds > 300


UNHEALTHY_THRESHOLD = 3  # consecutive failures before marking unhealthy


class PeerRegistry:
    """Thread-safe registry of discovered peers."""

    def __init__(self):
        self._peers: dict[str, Peer] = {}  # node_id -> Peer
        self._lock = asyncio.Lock()
        self._callbacks: list[callable] = []
        self._failure_counts: dict[str, int] = {}  # node_id -> consecutive failures

    def on_change(self, callback: callable):
        """Register a callback for peer changes. callback(event, peer)."""
        self._callbacks.append(callback)

    async def _notify(self, event: str, peer: Peer):
        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event, peer)
                else:
                    cb(event, peer)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def add_or_update(self, peer: Peer):
        async with self._lock:
            # Peer responded via mDNS — it's alive, reset failures
            self._failure_counts.pop(peer.node_id, None)
            existing = self._peers.get(peer.node_id)
            if existing:
                peer.first_seen = existing.first_seen
                self._peers[peer.node_id] = peer
                await self._notify("updated", peer)
                logger.info(f"Updated peer: {peer.hostname} ({peer.ip})")
            else:
                self._peers[peer.node_id] = peer
                await self._notify("added", peer)
                logger.info(
                    f"Discovered peer: {peer.hostname} ({peer.ip}) "
                    f"[{peer.node_tier}] {peer.capabilities}"
                )

    async def remove(self, node_id: str):
        async with self._lock:
            peer = self._peers.pop(node_id, None)
            if peer:
                await self._notify("removed", peer)
                logger.info(f"Lost peer: {peer.hostname} ({peer.ip})")

    async def get_all(self) -> list[Peer]:
        async with self._lock:
            return list(self._peers.values())

    async def get_by_capability(self, capability: str) -> list[Peer]:
        async with self._lock:
            return [
                p for p in self._peers.values() if capability in p.capabilities
            ]

    async def get_by_tier(self, tier: str) -> list[Peer]:
        async with self._lock:
            return [p for p in self._peers.values() if p.node_tier == tier]

    def record_failure(self, node_id: str):
        """Record a failed request to a peer."""
        count = self._failure_counts.get(node_id, 0) + 1
        self._failure_counts[node_id] = count
        if count >= UNHEALTHY_THRESHOLD:
            peer = self._peers.get(node_id)
            name = peer.hostname if peer else node_id
            logger.warning(
                f"🏥 Peer {name} marked unhealthy "
                f"({count} consecutive failures)"
            )

    def record_success(self, node_id: str):
        """Record a successful request to a peer.

        Clears failure count AND refreshes last_seen — successful HTTP
        communication is proof the peer is alive, independent of mDNS.
        """
        if node_id in self._failure_counts:
            peer = self._peers.get(node_id)
            name = peer.hostname if peer else node_id
            logger.info(f"💚 Peer {name} is healthy again")
            del self._failure_counts[node_id]
        # Update last_seen — prevents stale-marking when mDNS updates
        # don't arrive (identical TXT records = no zeroconf Updated event)
        peer = self._peers.get(node_id)
        if peer:
            peer.last_seen = time.time()

    def is_healthy(self, node_id: str) -> bool:
        """Check if a peer is considered healthy (< UNHEALTHY_THRESHOLD failures)."""
        return self._failure_counts.get(node_id, 0) < UNHEALTHY_THRESHOLD

    @property
    def count(self) -> int:
        return len(self._peers)


def _identity_to_txt(identity: NodeIdentity) -> dict[str, str]:
    """Encode node identity into mDNS TXT record properties.

    TXT records have a 255-byte limit per key-value pair,
    so we keep values short and split large lists.
    """
    return {
        "node_id": identity.node_id,
        "tier": identity.node_tier,
        "caps": ",".join(identity.capabilities),
        "gpu_tier": identity.gpu_tier,
        "gpu": identity.gpu_name or "none",
        "vram": str(identity.vram_total_mb),
        "models": ",".join(identity.available_models[:10]),  # Cap at 10
        "version": identity.version,
    }


def _txt_to_peer(properties: dict[bytes, bytes], ip: str, port: int) -> Peer | None:
    """Decode mDNS TXT record into a Peer object."""
    try:
        # zeroconf returns bytes keys and values
        props = {
            k.decode("utf-8"): v.decode("utf-8") if v else ""
            for k, v in properties.items()
        }

        return Peer(
            node_id=props.get("node_id", "unknown"),
            hostname=props.get("node_id", "unknown"),  # Updated from ServiceInfo
            ip=ip,
            port=port,
            node_tier=props.get("tier", "light"),
            capabilities=props.get("caps", "").split(",") if props.get("caps") else [],
            gpu_tier=props.get("gpu_tier", "none"),
            gpu_name=props.get("gpu") if props.get("gpu") != "none" else None,
            vram_total_mb=int(props.get("vram", "0")),
            available_models=(
                props.get("models", "").split(",") if props.get("models") else []
            ),
            version=props.get("version", "unknown"),
        )
    except Exception as e:
        logger.error(f"Failed to parse peer TXT record: {e}")
        return None


class Discovery:
    """mDNS-based discovery for mycoSwarm nodes.

    Usage:
        discovery = Discovery(my_identity, registry)
        await discovery.start()
        # ... nodes find each other automatically ...
        await discovery.stop()
    """

    def __init__(
        self,
        identity: NodeIdentity,
        registry: PeerRegistry,
        port: int = DEFAULT_PORT,
    ):
        self.identity = identity
        self.registry = registry
        self.port = port
        self._zeroconf: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None

    async def start(self):
        """Start announcing this node and listening for peers."""
        self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)

        # Build our service info
        ip = self.identity.lan_ip
        if not ip:
            logger.error("No LAN IP detected — cannot start discovery")
            return

        service_name = f"{self.identity.node_id}.{SERVICE_TYPE}"
        txt_props = _identity_to_txt(self.identity)

        all_addrs = _all_lan_addresses()
        if not all_addrs:
            all_addrs = [socket.inet_aton(ip)]

        self._service_info = ServiceInfo(
            type_=SERVICE_TYPE,
            name=service_name,
            addresses=all_addrs,
            port=self.port,
            properties=txt_props,
            server=f"{self.identity.hostname}.local.",
        )

        # Register ourselves (allow_name_change prevents NonUniqueNameException on restart)
        await self._zeroconf.async_register_service(
            self._service_info, allow_name_change=True
        )
        addr_strs = [socket.inet_ntoa(a) for a in all_addrs]
        logger.info(
            f"📡 Announcing: {self.identity.node_id} on {addr_strs} "
            f"port {self.port} [{self.identity.node_tier}]"
        )

        # Browse for peers
        self._browser = AsyncServiceBrowser(
            self._zeroconf.zeroconf,
            SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )
        logger.info(f"👀 Listening for peers on {SERVICE_TYPE}")

    def _on_service_state_change(self, zeroconf: Zeroconf, service_type: str,
                                  name: str, state_change) -> None:
        """Callback when a service is added, removed, or updated."""
        asyncio.ensure_future(
            self._handle_state_change(zeroconf, service_type, name, state_change)
        )

    async def _handle_state_change(self, zeroconf: Zeroconf, service_type: str,
                                    name: str, state_change) -> None:
        """Process service state changes."""
        from zeroconf import ServiceStateChange

        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):
            info = AsyncServiceInfo(service_type, name)
            await info.async_request(zeroconf, 3000)

            if not info.addresses:
                return

            ip = socket.inet_ntoa(info.addresses[0])
            port = info.port

            # Skip ourselves
            peer = _txt_to_peer(info.properties, ip, port)
            if peer and peer.node_id != self.identity.node_id:
                peer.hostname = info.server.rstrip(".").replace(".local", "")
                await self.registry.add_or_update(peer)

        elif state_change == ServiceStateChange.Removed:
            # Extract node_id from service name
            node_id = name.replace(f".{SERVICE_TYPE}", "")
            if node_id != self.identity.node_id:
                await self.registry.remove(node_id)

    async def update_identity(self, identity: NodeIdentity):
        """Update our announced identity (e.g., after load changes)."""
        self.identity = identity
        if self._service_info and self._zeroconf:
            txt_props = _identity_to_txt(identity)
            ip = identity.lan_ip
            if ip:
                all_addrs = _all_lan_addresses()
                if not all_addrs:
                    all_addrs = [socket.inet_aton(ip)]

                new_info = ServiceInfo(
                    type_=SERVICE_TYPE,
                    name=self._service_info.name,
                    addresses=all_addrs,
                    port=self.port,
                    properties=txt_props,
                    server=f"{identity.hostname}.local.",
                )
                await self._zeroconf.async_update_service(new_info)
                self._service_info = new_info
                logger.debug("Updated service announcement")

    async def stop(self):
        """Stop discovery and unregister from the network."""
        if self._browser:
            await self._browser.async_cancel()
            self._browser = None

        if self._service_info and self._zeroconf:
            await self._zeroconf.async_unregister_service(self._service_info)
            logger.info("Unregistered from network")

        if self._zeroconf:
            await self._zeroconf.async_close()
            self._zeroconf = None


# Need to import this for the handler
from zeroconf.asyncio import AsyncServiceInfo
