"""mycoSwarm Dashboard — web UI for swarm monitoring.

Queries the existing daemon API (/status + /peers) and renders
a live-updating dashboard with node cards.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

logger = logging.getLogger(__name__)


def _load_swarm_headers() -> dict:
    """Load swarm auth headers. Returns empty dict if no token found."""
    try:
        from mycoswarm.auth import load_token, get_auth_header
        token = load_token()
        if token:
            return get_auth_header(token)
        logger.warning("No swarm token found — dashboard requests will be unauthenticated")
    except Exception as e:
        logger.warning("Could not load swarm token: %s", e)
    return {}


def create_app(daemon_port: int = 7890) -> FastAPI:
    """Create the dashboard FastAPI application."""

    app = FastAPI(title="mycoSwarm Dashboard", version="0.1.0")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    daemon_url = f"http://localhost:{daemon_port}"
    _auth_headers = _load_swarm_headers()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/api/status")
    async def api_status():
        try:
            async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers) as client:
                status_resp = await client.get(f"{daemon_url}/status")
                status = status_resp.json()

                peers_resp = await client.get(f"{daemon_url}/peers")
                peers_raw = peers_resp.json()
        except (httpx.ConnectError, httpx.TimeoutException):
            return {
                "error": "Daemon not reachable",
                "local": None,
                "peers": [],
                "summary": {
                    "total_nodes": 0,
                    "gpu_nodes": 0,
                    "cpu_nodes": 0,
                    "total_vram": 0,
                    "total_ram": 0,
                },
            }

        now = time.time()

        # Query each peer's /status endpoint for system stats
        async def _fetch_peer_status(client, ip, port):
            try:
                resp = await client.get(f"http://{ip}:{port}/status")
                return resp.json()
            except Exception:
                return None

        peer_statuses = []
        if peers_raw:
            async with httpx.AsyncClient(timeout=3.0, headers=_auth_headers) as peer_client:
                peer_statuses = await asyncio.gather(
                    *[
                        _fetch_peer_status(peer_client, p["ip"], p["port"])
                        for p in peers_raw
                    ]
                )

        # Build peer list enriched with system stats
        peers = []
        for p, ps in zip(peers_raw, peer_statuses):
            last_seen = p.get("last_seen", 0)
            online = (now - last_seen) < 60
            peer_data = {
                "node_id": p.get("node_id", ""),
                "hostname": p.get("hostname", ""),
                "ip": p.get("ip", ""),
                "port": p.get("port", 0),
                "node_tier": p.get("node_tier", ""),
                "gpu_name": p.get("gpu_name"),
                "vram_total_mb": p.get("vram_total_mb", 0),
                "models": p.get("available_models", []),
                "version": p.get("version", ""),
                "last_seen": last_seen,
                "online": online,
            }

            # Enrich with live system stats from peer's /status
            if ps and not ps.get("error"):
                peer_data.update({
                    "vram_free_mb": ps.get("vram_free_mb", 0),
                    "cpu_model": ps.get("cpu_model", ""),
                    "cpu_cores": ps.get("cpu_cores", 0),
                    "cpu_usage_percent": ps.get("cpu_usage_percent", 0),
                    "ram_total_mb": ps.get("ram_total_mb", 0),
                    "ram_used_mb": ps.get("ram_used_mb", 0),
                    "disk_total_gb": ps.get("disk_total_gb", 0),
                    "disk_used_gb": ps.get("disk_used_gb", 0),
                    "os": ps.get("os", ""),
                    "architecture": ps.get("architecture", ""),
                    "uptime": ps.get("uptime_seconds", 0),
                })
                # Prefer version from live /status over mDNS
                if ps.get("version"):
                    peer_data["version"] = ps["version"]

            peers.append(peer_data)

        local = {
            "node_id": status.get("node_id", ""),
            "hostname": status.get("hostname", ""),
            "version": status.get("version", ""),
            "node_tier": status.get("node_tier", ""),
            "gpu": status.get("gpu"),
            "vram_total_mb": status.get("vram_total_mb", 0),
            "vram_free_mb": status.get("vram_free_mb", 0),
            "models": status.get("ollama_models", []),
            "uptime": status.get("uptime_seconds", 0),
            "cpu_model": status.get("cpu_model", ""),
            "cpu_cores": status.get("cpu_cores", 0),
            "cpu_usage_percent": status.get("cpu_usage_percent", 0),
            "ram_total_mb": status.get("ram_total_mb", 0),
            "ram_used_mb": status.get("ram_used_mb", 0),
            "disk_total_gb": status.get("disk_total_gb", 0),
            "disk_used_gb": status.get("disk_used_gb", 0),
            "os": status.get("os", ""),
            "architecture": status.get("architecture", ""),
            "ip": status.get("ip", ""),
            "port": status.get("port", 0),
        }

        # Summary
        total_nodes = 1 + len(peers)
        gpu_nodes = (1 if local["gpu"] else 0) + sum(
            1 for p in peers if p["gpu_name"]
        )
        total_vram = local["vram_total_mb"] + sum(
            p["vram_total_mb"] for p in peers
        )
        total_ram = local["ram_total_mb"] + sum(
            p.get("ram_total_mb", 0) for p in peers
        )

        return {
            "local": local,
            "peers": peers,
            "summary": {
                "total_nodes": total_nodes,
                "gpu_nodes": gpu_nodes,
                "cpu_nodes": total_nodes - gpu_nodes,
                "total_vram": total_vram,
                "total_ram": total_ram,
            },
        }

    @app.get("/api/memory")
    async def api_memory():
        """Return memory stats: facts, sessions, ChromaDB counts."""
        from mycoswarm.memory import FACTS_PATH, SESSIONS_PATH
        from mycoswarm.library import CHROMA_DIR

        # Facts count
        facts_count = 0
        try:
            import json as _json
            data = _json.loads(FACTS_PATH.read_text())
            facts_count = len(data.get("facts", []))
        except (FileNotFoundError, _json.JSONDecodeError, OSError):
            pass

        # Sessions count
        sessions_count = 0
        try:
            for line in SESSIONS_PATH.read_text().splitlines():
                if line.strip():
                    sessions_count += 1
        except (FileNotFoundError, OSError):
            pass

        # ChromaDB collection stats
        session_chunks = 0
        doc_chunks = 0
        docs_indexed = 0
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))

            try:
                sess_col = client.get_collection("session_memory")
                session_chunks = sess_col.count()
            except (ValueError, Exception):
                pass

            try:
                doc_col = client.get_collection("mycoswarm_docs")
                doc_chunks = doc_col.count()
                # Count unique source files
                all_meta = doc_col.get(include=["metadatas"])
                if all_meta and all_meta["metadatas"]:
                    docs_indexed = len({
                        m.get("source", "") for m in all_meta["metadatas"]
                    })
            except (ValueError, Exception):
                pass
        except Exception:
            pass

        return {
            "facts_count": facts_count,
            "sessions_count": sessions_count,
            "session_chunks": session_chunks,
            "doc_chunks": doc_chunks,
            "docs_indexed": docs_indexed,
        }

    return app
