"""Tests for ``_default_workspace_root`` — security-critical refusal logic.

A workspace root of ``/`` or ``$HOME`` gives the agent reach over the
entire machine. The default factory must refuse to start, not just warn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.loop.agent import WORKSPACE_ROOT_ENV, _default_workspace_root

if TYPE_CHECKING:
    from pathlib import Path


class TestWorkspaceRootRefusal:
    """Pins the broad-workspace refusal behavior."""

    def test_refuses_filesystem_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WHEN WORKSPACE_ROOT is "/" THEN startup raises with a fix hint."""
        monkeypatch.setenv(WORKSPACE_ROOT_ENV, "/")
        with pytest.raises(RuntimeError, match="too broad"):
            _default_workspace_root()

    def test_refuses_home_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """A monkey-patched $HOME is refused — pins that we re-read HOME at call time."""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(fake_home))
        with pytest.raises(RuntimeError, match="too broad"):
            _default_workspace_root()

    def test_refuses_users_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``/Users`` (macOS) gives every user's files; refuse it."""
        monkeypatch.setenv(WORKSPACE_ROOT_ENV, "/Users")
        with pytest.raises(RuntimeError, match="too broad"):
            _default_workspace_root()

    def test_refuses_system_directories(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """System paths like /etc and /tmp are refused as workspace roots."""
        for system_path in ("/etc", "/tmp", "/usr", "/var"):  # noqa: S108
            monkeypatch.setenv(WORKSPACE_ROOT_ENV, system_path)
            with pytest.raises(RuntimeError, match="too broad"):
                _default_workspace_root()

    def test_accepts_project_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """WHEN WORKSPACE_ROOT is a real project path THEN startup succeeds."""
        monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(tmp_path))
        assert _default_workspace_root() == tmp_path.resolve()
