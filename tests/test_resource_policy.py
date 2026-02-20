"""Tests for resource policy enforcement."""

import os
import pytest
from mycoswarm.resource_policy import (
    check_access, AccessLevel, AccessResult, ResourceOwner,
    format_access_check, log_access,
)


class TestMonicaSpace:
    """Monica's own directories — full access."""

    def test_drafts_write(self):
        r = check_access("~/insiderllm-drafts/test.md", "write")
        assert r.allowed
        assert r.level == AccessLevel.FULL
        assert r.owner == ResourceOwner.MONICA

    def test_drafts_read(self):
        r = check_access("~/insiderllm-drafts/deepseek-guide.md", "read")
        assert r.allowed

    def test_sessions_write(self):
        r = check_access("~/.config/mycoswarm/sessions/chat-123.jsonl", "write")
        assert r.allowed
        assert r.level == AccessLevel.FULL

    def test_library_write(self):
        r = check_access("~/.config/mycoswarm/library/chroma.sqlite3", "write")
        assert r.allowed

    def test_memory_write(self):
        r = check_access("~/.config/mycoswarm/memory/procedures.jsonl", "write")
        assert r.allowed

    def test_tmp_workspace(self):
        r = check_access("/tmp/mycoswarm/scratch.txt", "write")
        assert r.allowed
        assert r.owner == ResourceOwner.MONICA


class TestSharedSpace:
    """Shared directories — read-write with logging or read-only."""

    def test_mycoswarm_docs_read(self):
        r = check_access("~/mycoswarm-docs/content-plan.md", "read")
        assert r.allowed
        assert r.level == AccessLevel.READ_WRITE

    def test_mycoswarm_docs_write(self):
        r = check_access("~/mycoswarm-docs/new-doc.md", "write")
        assert r.allowed
        assert r.level == AccessLevel.READ_WRITE

    def test_project_docs_read(self):
        r = check_access("~/Desktop/mycoSwarm/docs/SKILL.md", "read")
        assert r.allowed
        assert r.level == AccessLevel.READ_ONLY

    def test_project_docs_write_denied(self):
        r = check_access("~/Desktop/mycoSwarm/docs/SKILL.md", "write")
        assert not r.allowed
        assert r.level == AccessLevel.READ_ONLY

    def test_plan_md_read(self):
        r = check_access("~/Desktop/mycoSwarm/PLAN.md", "read")
        assert r.allowed

    def test_plan_md_write_denied(self):
        r = check_access("~/Desktop/mycoSwarm/PLAN.md", "write")
        assert not r.allowed

    def test_claude_md_read(self):
        r = check_access("~/Desktop/mycoSwarm/CLAUDE.md", "read")
        assert r.allowed

    def test_readme_read(self):
        r = check_access("~/Desktop/mycoSwarm/README.md", "read")
        assert r.allowed


class TestGuardianSpace:
    """Guardian-owned code — requires approval."""

    def test_source_code_ask(self):
        r = check_access("~/Desktop/mycoSwarm/src/mycoswarm/cli.py", "read")
        assert not r.allowed
        assert r.needs_approval
        assert r.level == AccessLevel.ASK
        assert r.owner == ResourceOwner.GUARDIAN

    def test_scripts_ask(self):
        r = check_access("~/Desktop/mycoSwarm/scripts/install-safety-procedures.py", "write")
        assert not r.allowed
        assert r.needs_approval

    def test_tests_ask(self):
        r = check_access("~/Desktop/mycoSwarm/tests/test_instinct.py", "read")
        assert not r.allowed
        assert r.needs_approval

    def test_insiderllm_site_ask(self):
        r = check_access("~/Desktop/InsiderLLM/content/posts/test.md", "write")
        assert not r.allowed
        assert r.needs_approval


class TestSystemSpace:
    """System paths — always denied."""

    def test_ssh_denied(self):
        r = check_access("~/.ssh/id_rsa", "read")
        assert not r.allowed
        assert r.level == AccessLevel.DENY
        assert not r.needs_approval

    def test_gnupg_denied(self):
        r = check_access("~/.gnupg/pubring.kbx", "read")
        assert not r.allowed

    def test_etc_denied(self):
        r = check_access("/etc/passwd", "read")
        assert not r.allowed
        assert r.level == AccessLevel.DENY

    def test_usr_denied(self):
        r = check_access("/usr/bin/python3", "execute")
        assert not r.allowed

    def test_var_denied(self):
        r = check_access("/var/log/syslog", "read")
        assert not r.allowed


class TestDefaultDeny:
    """Paths that don't match any rule — denied."""

    def test_random_path(self):
        r = check_access("/some/random/path", "read")
        assert not r.allowed
        assert r.owner == ResourceOwner.SYSTEM

    def test_home_root(self):
        r = check_access("~/random-file.txt", "read")
        assert not r.allowed

    def test_other_desktop_dir(self):
        r = check_access("~/Desktop/SomeOtherProject/file.py", "write")
        assert not r.allowed


class TestFormatAccessCheck:
    """Display formatting for /access command."""

    def test_format_full(self):
        out = format_access_check("~/insiderllm-drafts/test.md")
        assert "monica" in out
        assert "full" in out

    def test_format_deny(self):
        out = format_access_check("/etc/passwd")
        assert "system" in out
        assert "deny" in out

    def test_format_ask(self):
        out = format_access_check("~/Desktop/mycoSwarm/src/mycoswarm/cli.py")
        assert "guardian" in out
        assert "ask" in out

    def test_format_read_only(self):
        out = format_access_check("~/Desktop/mycoSwarm/docs/SKILL.md")
        assert "ro" in out


class TestAuditLog:
    """Audit logging."""

    def test_log_access_writes_entry(self, tmp_path, monkeypatch):
        log_file = tmp_path / "access.log"
        monkeypatch.setattr(
            "mycoswarm.resource_policy.os.path.expanduser",
            lambda p: str(log_file) if "access.log" in p else os.path.expanduser(p),
        )
        result = AccessResult(
            allowed=True, level=AccessLevel.FULL,
            owner=ResourceOwner.MONICA, reason="test",
        )
        log_access("/tmp/mycoswarm/test.txt", "write", result)
        assert log_file.exists()
        import json
        entry = json.loads(log_file.read_text().strip())
        assert entry["allowed"] is True
        assert entry["operation"] == "write"
        assert entry["path"] == "/tmp/mycoswarm/test.txt"
