"""Tests for mycoswarm.plugins — plugin discovery and loading."""

import pytest

from mycoswarm.api import TaskRequest, TaskResult, TaskStatus
from mycoswarm.plugins import discover_plugins, register_plugins, PluginInfo


def test_discover_plugins_finds_valid_dirs(tmp_path):
    """Create tmp plugin dir with valid plugin.yaml + handler.py, verify discovered."""
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()

    (plugin_dir / "plugin.yaml").write_text(
        "name: my_plugin\n"
        "task_type: my_task\n"
        "description: A test plugin\n"
        "capabilities:\n"
        "  - cpu_worker\n"
    )
    (plugin_dir / "handler.py").write_text(
        "from mycoswarm.api import TaskRequest, TaskResult, TaskStatus\n"
        "async def handle(task: TaskRequest) -> TaskResult:\n"
        "    return TaskResult(task_id=task.task_id, status=TaskStatus.COMPLETED, result={'ok': True})\n"
    )

    plugins = discover_plugins(tmp_path)

    assert len(plugins) == 1
    assert plugins[0].name == "my_plugin"
    assert plugins[0].task_type == "my_task"
    assert plugins[0].loaded is True
    assert plugins[0].handler is not None


def test_invalid_yaml_skipped_gracefully(tmp_path):
    """Plugin dir with malformed plugin.yaml → PluginInfo with error, no crash."""
    plugin_dir = tmp_path / "bad_plugin"
    plugin_dir.mkdir()

    # Missing task_type field
    (plugin_dir / "plugin.yaml").write_text("name: bad_plugin\n")
    (plugin_dir / "handler.py").write_text(
        "async def handle(task): pass\n"
    )

    plugins = discover_plugins(tmp_path)

    assert len(plugins) == 1
    assert plugins[0].error is not None
    assert plugins[0].loaded is False


def test_duplicate_task_type_rejected(tmp_path):
    """Register two plugins with same task_type → second gets error."""
    handlers = {"existing_type": lambda t: None}
    identity_caps = ["cpu_worker"]

    plugin = PluginInfo(
        name="dup_plugin",
        task_type="existing_type",
        handler=lambda t: None,
    )

    registered = register_plugins([plugin], handlers, identity_caps)

    assert len(registered) == 0
    assert plugin.error is not None
    assert "already exists" in plugin.error


def test_missing_handler_py(tmp_path):
    """Plugin dir with yaml but no handler.py → error message."""
    plugin_dir = tmp_path / "no_handler"
    plugin_dir.mkdir()

    (plugin_dir / "plugin.yaml").write_text(
        "name: no_handler\n"
        "task_type: some_task\n"
    )

    plugins = discover_plugins(tmp_path)

    assert len(plugins) == 1
    assert plugins[0].loaded is False
    assert "handler.py" in plugins[0].error.lower()


def test_empty_plugin_dir(tmp_path):
    """No subdirectories → empty list."""
    plugins = discover_plugins(tmp_path)
    assert plugins == []
