"""mycoSwarm plugin loader.

Scans ~/.config/mycoswarm/plugins/ for subdirectories containing:
  plugin.yaml  — metadata (name, task_type, capabilities, description)
  handler.py   — must export: async def handle(task) -> TaskResult

Plugins are loaded once at daemon startup. No hot reloading.
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from mycoswarm.api import TaskRequest, TaskResult

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path("~/.config/mycoswarm/plugins").expanduser()

# Required fields in plugin.yaml
REQUIRED_FIELDS = {"name", "task_type"}


@dataclass
class PluginInfo:
    """Metadata for a loaded plugin."""

    name: str
    task_type: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    path: Path = field(default_factory=Path)
    handler: Callable[[TaskRequest], Awaitable[TaskResult]] | None = None
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.handler is not None and self.error is None


def _parse_yaml_simple(path: Path) -> dict:
    """Minimal YAML parser for flat plugin.yaml files.

    Handles simple key: value pairs and lists (- item).
    No dependency on PyYAML.
    """
    result: dict = {}
    current_key: str | None = None

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item
        if stripped.startswith("- ") and current_key is not None:
            if not isinstance(result.get(current_key), list):
                result[current_key] = []
            result[current_key].append(stripped[2:].strip().strip('"').strip("'"))
            continue

        # Key: value
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val:
                result[key] = val
                current_key = key
            else:
                # Next lines may be list items
                result[key] = []
                current_key = key

    return result


def _load_plugin(plugin_dir: Path) -> PluginInfo:
    """Load a single plugin from a directory."""
    yaml_path = plugin_dir / "plugin.yaml"
    handler_path = plugin_dir / "handler.py"

    # Parse metadata
    if not yaml_path.exists():
        return PluginInfo(
            name=plugin_dir.name, task_type="",
            path=plugin_dir, error="Missing plugin.yaml",
        )

    meta = _parse_yaml_simple(yaml_path)

    name = meta.get("name", plugin_dir.name)
    task_type = meta.get("task_type", "")
    description = meta.get("description", "")
    capabilities = meta.get("capabilities", [])
    if isinstance(capabilities, str):
        capabilities = [capabilities]

    if not task_type:
        return PluginInfo(
            name=name, task_type="",
            path=plugin_dir, error="plugin.yaml missing 'task_type'",
        )

    # Load handler
    if not handler_path.exists():
        return PluginInfo(
            name=name, task_type=task_type, description=description,
            capabilities=capabilities, path=plugin_dir,
            error="Missing handler.py",
        )

    try:
        spec = importlib.util.spec_from_file_location(
            f"mycoswarm_plugin_{plugin_dir.name}", handler_path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        handler = getattr(module, "handle", None)
        if handler is None:
            return PluginInfo(
                name=name, task_type=task_type, description=description,
                capabilities=capabilities, path=plugin_dir,
                error="handler.py missing 'handle' function",
            )

        return PluginInfo(
            name=name, task_type=task_type, description=description,
            capabilities=capabilities, path=plugin_dir,
            handler=handler,
        )
    except Exception as e:
        return PluginInfo(
            name=name, task_type=task_type, description=description,
            capabilities=capabilities, path=plugin_dir,
            error=f"Failed to load handler: {e}",
        )


def discover_plugins(plugin_dir: Path = PLUGIN_DIR) -> list[PluginInfo]:
    """Scan the plugins directory and load all valid plugins."""
    if not plugin_dir.exists():
        return []

    plugins: list[PluginInfo] = []

    for child in sorted(plugin_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue

        plugin = _load_plugin(child)
        plugins.append(plugin)

        if plugin.loaded:
            logger.info(
                f"🔌 Plugin loaded: {plugin.name} "
                f"(task_type={plugin.task_type})"
            )
        else:
            logger.warning(
                f"⚠️ Plugin failed: {plugin.name} — {plugin.error}"
            )

    return plugins


def register_plugins(
    plugins: list[PluginInfo],
    handlers: dict,
    identity_capabilities: list[str],
) -> list[PluginInfo]:
    """Register loaded plugins into the worker HANDLERS and node capabilities.

    Returns the list of successfully registered plugins.
    """
    registered = []

    for plugin in plugins:
        if not plugin.loaded:
            continue

        if plugin.task_type in handlers:
            logger.warning(
                f"⚠️ Plugin {plugin.name}: task_type '{plugin.task_type}' "
                f"already registered, skipping"
            )
            plugin.error = f"task_type '{plugin.task_type}' already exists"
            continue

        handlers[plugin.task_type] = plugin.handler
        for cap in plugin.capabilities:
            if cap not in identity_capabilities:
                identity_capabilities.append(cap)

        registered.append(plugin)
        logger.info(
            f"🔌 Registered: {plugin.name} → task_type={plugin.task_type}"
        )

    return registered
