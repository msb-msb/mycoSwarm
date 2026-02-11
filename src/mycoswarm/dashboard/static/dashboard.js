/* mycoSwarm Dashboard — auto-refreshing node status */

(function () {
  var REFRESH_MS = 5000;

  var $nodes = document.getElementById("nodes-container");
  var $totalNodes = document.getElementById("total-nodes");
  var $gpuNodes = document.getElementById("gpu-nodes");
  var $cpuNodes = document.getElementById("cpu-nodes");
  var $totalVram = document.getElementById("total-vram");
  var $totalRam = document.getElementById("total-ram");
  var $connStatus = document.getElementById("connection-status");
  var $lastUpdate = document.getElementById("last-update");

  function esc(str) {
    var d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  function formatUptime(seconds) {
    if (!seconds) return "-";
    if (seconds < 60) return Math.round(seconds) + "s";
    if (seconds < 3600) return Math.round(seconds / 60) + "m";
    var h = Math.floor(seconds / 3600);
    var m = Math.round((seconds % 3600) / 60);
    return h + "h " + m + "m";
  }

  function formatLastSeen(ts) {
    if (!ts) return "never";
    var ago = Math.round(Date.now() / 1000 - ts);
    if (ago < 5) return "just now";
    if (ago < 60) return ago + "s ago";
    if (ago < 3600) return Math.round(ago / 60) + "m ago";
    return Math.round(ago / 3600) + "h ago";
  }

  function formatGB(mb) {
    if (!mb) return "0";
    return (mb / 1024).toFixed(1);
  }

  function pct(used, total) {
    if (!total) return 0;
    return Math.round((used / total) * 100);
  }

  function tierClass(tier) {
    if (!tier) return "cpu";
    var t = tier.toLowerCase();
    if (t === "executive") return "executive";
    if (t.includes("gpu")) return "gpu";
    if (t.includes("hybrid")) return "hybrid";
    return "cpu";
  }

  function statRow(label, value, pctVal) {
    var s = '<div class="stat-row"><span class="label">' + label + "</span>";
    s += '<span class="value">' + value + "</span>";
    if (pctVal !== undefined) {
      s += '<span class="pct">(' + pctVal + "%)</span>";
    }
    return s + "</div>";
  }

  function renderModels(models) {
    if (!models || !models.length) return "";
    var first = esc(models[0]);
    if (models.length === 1)
      return (
        '<div class="node-models">Models: <span class="model-primary">' +
        first +
        "</span></div>"
      );
    return (
      '<div class="node-models">Models: <span class="model-primary">' +
      first +
      "</span> +" +
      (models.length - 1) +
      " more</div>"
    );
  }

  function renderCard(node, isLocal) {
    var online = isLocal ? true : node.online;
    var statusClass = online ? "online" : "offline";
    var statusText = online ? "online" : "offline";
    var tc = tierClass(node.node_tier);

    // Address line
    var addr = "";
    var ip = node.ip || "";
    var port = node.port || 0;
    if (ip && port) addr = ip + ":" + port;
    else if (ip) addr = ip;

    // GPU / VRAM
    var gpu = isLocal ? node.gpu : node.gpu_name;
    var vramUsed = (node.vram_total_mb || 0) - (node.vram_free_mb || 0);

    // CPU
    var cpuModel = node.cpu_model || "";
    var cpuCores = node.cpu_cores || 0;
    var cpuUsage = node.cpu_usage_percent;

    // RAM
    var ramTotal = node.ram_total_mb || 0;
    var ramUsed = node.ram_used_mb || 0;

    // Disk
    var diskTotal = node.disk_total_gb || 0;
    var diskUsed = node.disk_used_gb || 0;

    // Build stats
    var stats = "";

    if (gpu) {
      stats += statRow("GPU:", esc(gpu));
      stats += statRow(
        "VRAM:",
        vramUsed.toLocaleString() +
          " / " +
          (node.vram_total_mb || 0).toLocaleString() +
          " MB",
        pct(vramUsed, node.vram_total_mb)
      );
    }

    if (cpuModel) {
      var cpuText = esc(cpuModel);
      if (cpuCores) cpuText += " &mdash; " + cpuCores + " cores";
      stats += statRow("CPU:", cpuText);
    }

    if (cpuUsage !== undefined && cpuUsage !== null) {
      stats += statRow("CPU Use:", cpuUsage + "%");
    }

    if (ramTotal) {
      stats += statRow(
        "RAM:",
        formatGB(ramUsed) + " / " + formatGB(ramTotal) + " GB",
        pct(ramUsed, ramTotal)
      );
    }

    if (diskTotal) {
      stats += statRow(
        "Disk:",
        diskUsed + " / " + diskTotal + " GB",
        pct(diskUsed, diskTotal)
      );
    }

    // Uptime / last seen
    if (isLocal) {
      stats += statRow("Uptime:", formatUptime(node.uptime));
    } else {
      stats += statRow("Seen:", formatLastSeen(node.last_seen));
      if (node.uptime) stats += statRow("Uptime:", formatUptime(node.uptime));
    }

    // OS / arch
    if (node.os || node.architecture) {
      var osText = [node.os, node.architecture].filter(Boolean).join(" / ");
      stats += statRow("OS:", esc(osText));
    }

    var html =
      '<div class="node-card' +
      (isLocal ? " local" : "") +
      '">' +
      '<div class="node-header">' +
      '<span class="node-hostname">' +
      esc(node.hostname) +
      (isLocal ? " (this node)" : "") +
      "</span>" +
      '<span class="node-tier ' +
      tc +
      '">' +
      esc((node.node_tier || "").toUpperCase()) +
      "</span>" +
      "</div>" +
      '<div class="node-subheader">' +
      '<span class="status-dot ' +
      statusClass +
      '"></span>' +
      "<span>" +
      statusText +
      "</span>" +
      (addr ? "<span>&middot; " + esc(addr) + "</span>" : "") +
      "</div>" +
      '<div class="node-stats">' +
      stats +
      "</div>" +
      renderModels(node.models) +
      "</div>";

    return html;
  }

  function updateSummary(summary) {
    $totalNodes.textContent = summary.total_nodes;
    $gpuNodes.textContent = summary.gpu_nodes;
    $cpuNodes.textContent = summary.cpu_nodes;

    var vram = summary.total_vram;
    $totalVram.textContent =
      vram >= 1024
        ? (vram / 1024).toFixed(1) + " GB"
        : vram.toLocaleString() + " MB";

    var ram = summary.total_ram;
    $totalRam.textContent =
      ram >= 1024
        ? (ram / 1024).toFixed(1) + " GB"
        : ram.toLocaleString() + " MB";
  }

  function setConnected(ok) {
    $connStatus.textContent = ok ? "connected" : "disconnected";
    $connStatus.className = "status-badge " + (ok ? "online" : "offline");
  }

  async function fetchStatus() {
    try {
      var resp = await fetch("/api/status");
      var data = await resp.json();

      if (data.error) {
        setConnected(false);
        $nodes.innerHTML =
          '<p class="placeholder">&#x26a0; ' +
          esc(data.error) +
          ". Is the daemon running?</p>";
        return;
      }

      setConnected(true);
      updateSummary(data.summary);

      var html = "";
      if (data.local) {
        html += renderCard(data.local, true);
      }
      (data.peers || []).forEach(function (peer) {
        html += renderCard(peer, false);
      });

      $nodes.innerHTML = html || '<p class="placeholder">No nodes found.</p>';
      $lastUpdate.textContent =
        "Updated: " + new Date().toLocaleTimeString();
    } catch (e) {
      setConnected(false);
      $nodes.innerHTML =
        '<p class="placeholder">&#x26a0; Cannot reach dashboard API.</p>';
    }
  }

  fetchStatus();
  setInterval(fetchStatus, REFRESH_MS);
})();
