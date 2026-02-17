"""Tests for development script helpers in ``inga._scripts``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import inga._scripts as scripts


def test_run_commands_unknown_command_exits() -> None:
    """Unknown command names should fail fast with exit code 1."""
    with pytest.raises(SystemExit) as exc:
        scripts._run_commands(["does_not_exist"])

    assert exc.value.code == 1


def test_run_commands_stops_on_failed_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero subprocess exit should be propagated."""

    def fake_run(_cmd: list[str]) -> SimpleNamespace:
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(scripts.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        scripts._run_commands(["fmt"])

    assert exc.value.code == 7


def test_run_commands_executes_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requested commands should execute in the given order."""
    seen: list[list[str]] = []

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scripts.subprocess, "run", fake_run)

    scripts._run_commands(["fmt", "lint"])

    assert seen == [
        scripts.COMMANDS["fmt"][0],
        scripts.COMMANDS["lint"][0],
    ]


def test_dev_runs_default_pipeline_without_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``dev`` with no CLI args should run default steps."""
    called_with: list[list[str]] = []

    def fake_run_commands(names: list[str]) -> None:
        called_with.append(names)

    monkeypatch.setattr(scripts, "_run_commands", fake_run_commands)
    monkeypatch.setattr(scripts.sys, "argv", ["dev"])

    scripts.dev()

    assert called_with == [["fmt", "lint", "typecheck", "test"]]


def test_dev_runs_explicit_requested_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling ``dev`` with args should pass them through unchanged."""
    called_with: list[list[str]] = []

    def fake_run_commands(names: list[str]) -> None:
        called_with.append(names)

    monkeypatch.setattr(scripts, "_run_commands", fake_run_commands)
    monkeypatch.setattr(scripts.sys, "argv", ["dev", "test", "test_integration"])

    scripts.dev()

    assert called_with == [["test", "test_integration"]]
