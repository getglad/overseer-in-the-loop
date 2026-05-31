"""Streaming service — a result frame per attack then a scorecard, fault-isolated."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.protocol import MessageType
from src.redteam import service as svc
from src.redteam.attacks import Attack, Expected
from src.redteam.models import AttackResult
from src.redteam.service import run_redteam

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from .conftest import StubRails

_BLOCKED = ("bash", {"command": "rm -rf ."})
_ALLOWED = ("read_file", {"file_path": "README.md"})


def _attack(attack_id: str, action: tuple[str, dict], expected: Expected) -> Attack:
    """Build a minimal Attack from a (tool_name, tool_args) action."""
    tool_name, tool_args = action
    return Attack(
        attack_id=attack_id,
        category="test",
        description="probe",
        tool_name=tool_name,
        tool_args=tool_args,
        user_context="[current] do a thing",
        expected=expected,
    )


class TestStreaming:
    """run_redteam streams one result per attack, then the final scorecard."""

    async def test_emits_result_per_attack_then_scorecard(
        self,
        make_rails: Callable[..., StubRails],
    ) -> None:
        """Two attacks → two REDTEAM_RESULT frames and one REDTEAM_COMPLETE."""
        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        corpus = [_attack("a", _BLOCKED, Expected.BLOCK), _attack("b", _ALLOWED, Expected.ALLOW)]
        await run_redteam(send, make_rails(), corpus)

        results = [m for m in sent if m["type"] == MessageType.REDTEAM_RESULT]
        completes = [m for m in sent if m["type"] == MessageType.REDTEAM_COMPLETE]
        assert len(results) == 2
        assert len(completes) == 1

        assert results[0]["content"]["index"] == 0
        assert results[0]["content"]["total"] == 2
        assert results[0]["content"]["attack_id"] == "a"
        assert results[1]["content"]["index"] == 1

        scorecard = completes[0]["content"]
        assert scorecard["total"] == 2
        assert scorecard["passed"] == 2
        assert scorecard["false_allows"] == 0
        assert completes[0]["status"] == "complete"

    async def test_per_attack_error_isolation(
        self,
        make_rails: Callable[..., StubRails],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A probe that raises becomes a fail-closed result; the run still completes."""

        async def fake_run_attack(_rails: object, attack: Attack) -> AttackResult:
            if attack.attack_id == "boom":
                msg = "kaboom"
                raise RuntimeError(msg)
            return AttackResult(
                attack=attack,
                observed_allowed=attack.expected == Expected.ALLOW,
                observed_layer="rules",
                observed_reason="ok",
            )

        monkeypatch.setattr(svc, "run_attack", fake_run_attack)

        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        corpus = [
            _attack("before", _BLOCKED, Expected.BLOCK),
            _attack("boom", _BLOCKED, Expected.BLOCK),
            _attack("after", _ALLOWED, Expected.ALLOW),
        ]
        await run_redteam(send, make_rails(), corpus)

        results = [m for m in sent if m["type"] == MessageType.REDTEAM_RESULT]
        completes = [m for m in sent if m["type"] == MessageType.REDTEAM_COMPLETE]
        assert len(results) == 3
        assert len(completes) == 1

        errored = next(m for m in results if m["content"]["attack_id"] == "boom")
        assert errored["content"]["layer"] == "error"
        assert errored["content"]["observed_blocked"] is True
        # A block-expected probe that errored counts as caught — never a false-allow.
        assert errored["content"]["false_allow"] is False
        assert completes[0]["content"]["false_allows"] == 0
