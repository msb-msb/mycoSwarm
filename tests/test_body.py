"""Tests for the hardware body reader (Phase 31c Step 1)."""

from unittest.mock import MagicMock, patch

from mycoswarm.body import get_body_state, _read_gpu, _read_nodes


NVIDIA_SMI_OUTPUT = "58, 4821, 24576\n"

MOCK_STATUS = {
    "hostname": "Miu",
    "gpu": "NVIDIA GeForce RTX 3090",
    "node_tier": "executive",
    "vram_total_mb": 24576,
}

MOCK_PEERS = [
    {"hostname": "rushuna", "gpu_name": "NVIDIA GeForce RTX 3060", "node_tier": "specialist"},
    {"hostname": "naru", "gpu_name": None, "node_tier": "light"},
]


class TestReadGpu:
    @patch("mycoswarm.body.subprocess.run")
    @patch("mycoswarm.body.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_parses_nvidia_smi(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=NVIDIA_SMI_OUTPUT)

        result = _read_gpu()

        assert result["gpu_temp"] == 58.0
        assert result["vram_used_gb"] == round(4821 / 1024, 2)
        assert result["vram_total_gb"] == round(24576 / 1024, 2)
        assert result["vram_percent"] == round((4821 / 24576) * 100, 1)

    @patch("mycoswarm.body.shutil.which", return_value=None)
    def test_no_nvidia_smi(self, mock_which):
        result = _read_gpu()

        assert result["gpu_temp"] is None
        assert result["vram_used_gb"] is None
        assert result["vram_total_gb"] is None
        assert result["vram_percent"] is None

    @patch("mycoswarm.body.subprocess.run")
    @patch("mycoswarm.body.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_failure(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        result = _read_gpu()

        assert result["gpu_temp"] is None

    @patch("mycoswarm.body.subprocess.run")
    @patch("mycoswarm.body.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_timeout(self, mock_which, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        result = _read_gpu()

        assert result["gpu_temp"] is None


class TestReadNodes:
    @patch("mycoswarm.body.httpx.get")
    def test_reads_local_and_peers(self, mock_get):
        def _side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/status" in url:
                resp.json.return_value = MOCK_STATUS
            elif "/peers" in url:
                resp.json.return_value = MOCK_PEERS
            return resp

        mock_get.side_effect = _side_effect

        nodes = _read_nodes("http://localhost:7890")

        assert len(nodes) == 3
        assert nodes[0]["name"] == "Miu"
        assert nodes[0]["tier"] == "executive"
        assert nodes[0]["online"] is True
        assert nodes[1]["name"] == "rushuna"
        assert nodes[2]["name"] == "naru"
        assert nodes[2]["gpu"] is None

    def test_no_daemon_url(self):
        nodes = _read_nodes(None)
        assert nodes == []

    @patch("mycoswarm.body.httpx.get")
    def test_daemon_unreachable(self, mock_get):
        mock_get.side_effect = Exception("connection refused")

        nodes = _read_nodes("http://localhost:7890")

        assert nodes == []

    @patch("mycoswarm.body.httpx.get")
    def test_peers_fail_but_local_ok(self, mock_get):
        call_count = 0

        def _side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/status" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = MOCK_STATUS
                return resp
            raise Exception("peers endpoint down")

        mock_get.side_effect = _side_effect

        nodes = _read_nodes("http://localhost:7890")

        assert len(nodes) == 1
        assert nodes[0]["name"] == "Miu"


class TestGetBodyState:
    @patch("mycoswarm.body._read_nodes")
    @patch("mycoswarm.body._read_gpu")
    def test_combines_gpu_and_nodes(self, mock_gpu, mock_nodes):
        mock_gpu.return_value = {
            "gpu_temp": 62.0,
            "vram_used_gb": 5.0,
            "vram_total_gb": 24.0,
            "vram_percent": 20.8,
        }
        mock_nodes.return_value = [
            {"name": "Miu", "online": True, "gpu": "RTX 3090", "tier": "executive"},
        ]

        state = get_body_state(daemon_url="http://localhost:7890")

        assert state["gpu_temp"] == 62.0
        assert state["vram_used_gb"] == 5.0
        assert len(state["nodes"]) == 1
        assert state["nodes"][0]["name"] == "Miu"

    @patch("mycoswarm.body._read_nodes")
    @patch("mycoswarm.body._read_gpu")
    def test_solo_mode_no_daemon(self, mock_gpu, mock_nodes):
        mock_gpu.return_value = {
            "gpu_temp": 55.0,
            "vram_used_gb": 3.0,
            "vram_total_gb": 24.0,
            "vram_percent": 12.5,
        }
        mock_nodes.return_value = []

        state = get_body_state()

        assert state["gpu_temp"] == 55.0
        assert state["nodes"] == []

    @patch("mycoswarm.body._read_nodes")
    @patch("mycoswarm.body._read_gpu")
    def test_all_keys_present(self, mock_gpu, mock_nodes):
        mock_gpu.return_value = {
            "gpu_temp": None, "vram_used_gb": None,
            "vram_total_gb": None, "vram_percent": None,
        }
        mock_nodes.return_value = []

        state = get_body_state()

        assert "gpu_temp" in state
        assert "vram_used_gb" in state
        assert "vram_total_gb" in state
        assert "vram_percent" in state
        assert "nodes" in state


import subprocess  # noqa: E402 — needed for TimeoutExpired in test
