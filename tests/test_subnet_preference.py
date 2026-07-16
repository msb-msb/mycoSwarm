"""Tests for MYCOSWARM_SWARM_SUBNET soft-prefer address selection.

Covers the shared helper (in-CIDR / out-of-CIDR / malformed / unset), the
lan_ip property's preference, and _all_lan_addresses ordering. The invariant
throughout: with the env var UNSET, behavior is byte-for-byte the historical
first-enumerated order — the zero-config default must not change.
"""

import ipaddress
import socket
from collections import namedtuple
from unittest.mock import patch

from mycoswarm.hardware import (
    HardwareProfile,
    NetworkInterface,
    prefer_subnet,
    _swarm_subnet,
    SWARM_SUBNET_ENV,
)
from mycoswarm.discovery import _all_lan_addresses


# --- _swarm_subnet: env parsing ------------------------------------------

def test_swarm_subnet_unset_is_none(monkeypatch):
    monkeypatch.delenv(SWARM_SUBNET_ENV, raising=False)
    assert _swarm_subnet() is None


def test_swarm_subnet_valid_cidr(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    assert _swarm_subnet() == ipaddress.ip_network("192.168.50.0/24")


def test_swarm_subnet_malformed_is_none(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "not-a-cidr")
    assert _swarm_subnet() is None


def test_swarm_subnet_empty_is_none(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "   ")
    assert _swarm_subnet() is None


def test_swarm_subnet_host_bits_tolerated(monkeypatch):
    # strict=False — a host address with a prefix is accepted as its network.
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.13/24")
    assert _swarm_subnet() == ipaddress.ip_network("192.168.50.0/24")


# --- prefer_subnet: reordering -------------------------------------------

def test_prefer_subnet_unset_is_identity(monkeypatch):
    """UNSET env → list returned unchanged (the zero-config guarantee)."""
    monkeypatch.delenv(SWARM_SUBNET_ENV, raising=False)
    ips = ["192.168.1.25", "192.168.50.13"]
    assert prefer_subnet(ips) == ips


def test_prefer_subnet_leads_in_subnet(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    # .1 enumerated first, but .50 should lead after preference.
    assert prefer_subnet(["192.168.1.25", "192.168.50.13"]) == [
        "192.168.50.13",
        "192.168.1.25",
    ]


def test_prefer_subnet_soft_keeps_fallback(monkeypatch):
    """Out-of-subnet address is retained (soft-prefer), not dropped."""
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    out = prefer_subnet(["192.168.1.25", "192.168.50.13"])
    assert "192.168.1.25" in out
    assert set(out) == {"192.168.50.13", "192.168.1.25"}


def test_prefer_subnet_already_first_is_noop(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    ips = ["192.168.50.13", "192.168.1.25"]
    assert prefer_subnet(ips) == ips


def test_prefer_subnet_stable_within_groups(monkeypatch):
    """Relative order preserved within in- and out-of-subnet groups."""
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    ips = ["10.0.0.9", "192.168.50.13", "192.168.1.25", "192.168.50.20"]
    assert prefer_subnet(ips) == [
        "192.168.50.13",
        "192.168.50.20",  # in-subnet, original relative order
        "10.0.0.9",
        "192.168.1.25",  # out-of-subnet, original relative order
    ]


def test_prefer_subnet_malformed_env_is_identity(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50/nonsense")
    ips = ["192.168.1.25", "192.168.50.13"]
    assert prefer_subnet(ips) == ips


def test_prefer_subnet_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.delenv(SWARM_SUBNET_ENV, raising=False)
    subnet = ipaddress.ip_network("192.168.50.0/24")
    assert prefer_subnet(["192.168.1.25", "192.168.50.13"], subnet) == [
        "192.168.50.13",
        "192.168.1.25",
    ]


def test_prefer_subnet_empty_list(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    assert prefer_subnet([]) == []


# --- lan_ip: property preference -----------------------------------------

def _dual_homed_profile():
    """.50 wired fabric enumerated first, .1 wifi second (current fleet order)."""
    return HardwareProfile(
        hostname="node",
        gpus=[],
        network=[
            NetworkInterface(name="lo", ipv4="127.0.0.1", is_loopback=True),
            NetworkInterface(name="enp0s31f6", ipv4="192.168.50.13"),
            NetworkInterface(name="wlp2s0", ipv4="192.168.1.25"),
        ],
    )


def _wifi_first_profile():
    """Pathological boot: wifi enumerated before the wired fabric."""
    return HardwareProfile(
        hostname="node",
        gpus=[],
        network=[
            NetworkInterface(name="lo", ipv4="127.0.0.1", is_loopback=True),
            NetworkInterface(name="wlp2s0", ipv4="192.168.1.25"),
            NetworkInterface(name="enp0s31f6", ipv4="192.168.50.13"),
        ],
    )


def test_lan_ip_unset_is_first_enumerated(monkeypatch):
    """UNSET → first non-loopback, exactly as before."""
    monkeypatch.delenv(SWARM_SUBNET_ENV, raising=False)
    assert _dual_homed_profile().lan_ip == "192.168.50.13"
    # And when wifi enumerates first, unset behavior yields .1 (the old bug's root).
    assert _wifi_first_profile().lan_ip == "192.168.1.25"


def test_lan_ip_prefers_subnet(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    # Fabric preferred regardless of enumeration order.
    assert _dual_homed_profile().lan_ip == "192.168.50.13"
    assert _wifi_first_profile().lan_ip == "192.168.50.13"


def test_lan_ip_none_when_no_interfaces(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    assert HardwareProfile(hostname="x", gpus=[], network=[]).lan_ip is None


# --- _all_lan_addresses: announced ordering ------------------------------

_Snic = namedtuple("snic", ["family", "address"])


def _mock_addrs():
    """psutil.net_if_addrs() shape: wifi enumerated before wired fabric."""
    return {
        "lo": [_Snic(socket.AF_INET, "127.0.0.1")],
        "wlp2s0": [_Snic(socket.AF_INET, "192.168.1.25")],
        "enp0s31f6": [_Snic(socket.AF_INET, "192.168.50.13")],
    }


def test_all_lan_addresses_unset_is_enumeration_order(monkeypatch):
    monkeypatch.delenv(SWARM_SUBNET_ENV, raising=False)
    with patch("psutil.net_if_addrs", return_value=_mock_addrs()):
        packed = _all_lan_addresses()
    ips = [socket.inet_ntoa(a) for a in packed]
    # loopback dropped; raw psutil order otherwise (wifi first here).
    assert ips == ["192.168.1.25", "192.168.50.13"]


def test_all_lan_addresses_prefers_subnet_first(monkeypatch):
    monkeypatch.setenv(SWARM_SUBNET_ENV, "192.168.50.0/24")
    with patch("psutil.net_if_addrs", return_value=_mock_addrs()):
        packed = _all_lan_addresses()
    ips = [socket.inet_ntoa(a) for a in packed]
    # Fabric leads; wifi retained as fallback.
    assert ips == ["192.168.50.13", "192.168.1.25"]
