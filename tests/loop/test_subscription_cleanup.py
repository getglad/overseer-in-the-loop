"""Regression test for the IntermediateStepManager subscription contract.

``run_agent`` (src/loop/service.py) subscribes to the per-run
IntermediateStep stream to forward intermediate events to the
WebSocket client. NAT disposes the per-run ``Subject`` automatically
on completion, but the ``Subscription`` instance returned by
``subscribe()`` retains references to its observer callback — and via
that callback, to the run's ``step_queue`` — until something explicitly
calls ``unsubscribe()``.

These tests pin the capture+dispose contract so future refactors of
``run_agent`` can't silently regress. They exercise NAT's ``Subject``
directly because that's the primitive ``IntermediateStepManager`` uses
internally; using ``Subject`` keeps the test deterministic and
free of NAT runtime/Context plumbing.

Tests assert by observed behavior (do callbacks fire?) rather than by
inspecting Subject internals — that way the tests stay valid even if
NAT changes its private representation.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from nat.utils.reactive.subject import Subject

from src.core.conversation import get_recent_user_prompts
from src.core.protocol import MessageType
from src.loop import service as service_module
from src.loop.service import run_agent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _raise_in_try_with_unsubscribe(subject: Subject[int]) -> None:
    """Subscribe to the subject, raise inside the body, unsubscribe in finally.

    Mirrors the service.py pattern's exception path. Extracted so the
    ``pytest.raises`` block holds a single statement (satisfies PT012).
    """
    subscription = subject.subscribe(on_next=lambda _: None)
    try:
        msg = "simulated agent failure"
        raise RuntimeError(msg)
    finally:
        subscription.unsubscribe()


class TestSubscriptionLifecycle:
    """The capture+dispose pattern used by service.py for the step subscription."""

    def test_subscribe_without_unsubscribe_keeps_observer_active(self) -> None:
        """WITHOUT unsubscribe(), every subscriber keeps receiving events.

        Documents the pre-fix behavior. A regression of the fix would
        let ``run_agent`` match this pattern again — every prior run's
        ``step_queue`` would receive every event from every later run.
        """
        subject: Subject[int] = Subject()
        received_by: list[list[int]] = [[] for _ in range(5)]
        for sink in received_by:
            subject.subscribe(on_next=sink.append)

        subject.on_next(42)
        assert all(sink == [42] for sink in received_by), (
            f"expected all 5 sinks to receive the event, got {received_by}"
        )

    def test_subscribe_with_unsubscribe_in_finally_releases_observer(self) -> None:
        """WITH the service.py pattern, callbacks no longer fire after the scope exits."""
        subject: Subject[int] = Subject()
        received_by: list[list[int]] = [[] for _ in range(5)]
        for sink in received_by:
            subscription = subject.subscribe(on_next=sink.append)
            try:
                pass  # placeholder for the per-run agent work
            finally:
                subscription.unsubscribe()

        subject.on_next(42)
        assert all(sink == [] for sink in received_by), (
            f"expected no sinks to receive events post-unsubscribe, got {received_by}"
        )

    def test_unsubscribe_runs_even_when_body_raises(self) -> None:
        """The try/finally guarantees cleanup on the exception path.

        ``run_agent``'s outer ``except Exception`` is OUTSIDE the
        ``try/finally`` so a raise inside the run still triggers
        ``subscription.unsubscribe()`` before the exception propagates.
        """
        subject: Subject[int] = Subject()
        received: list[int] = []
        subscription = subject.subscribe(on_next=received.append)
        try:
            with pytest.raises(RuntimeError, match="simulated"):
                _raise_in_try_with_unsubscribe(subject)
            # After the raised body, the inner subscription is disposed.
            # The outer subscription (received) is still active — drop it
            # cleanly to mirror the per-run lifecycle.
        finally:
            subscription.unsubscribe()

        subject.on_next(42)
        assert received == [], (
            f"expected no events post-unsubscribe even on raise, got {received}"
        )

    def test_observer_callback_no_longer_fires_after_unsubscribe(self) -> None:
        """After unsubscribe, the observer's on_next is not invoked on new events.

        The functional invariant the fix protects: events from a later
        run cannot reach a previous run's ``step_queue``.
        """
        subject: Subject[int] = Subject()
        received: list[int] = []
        subscription = subject.subscribe(on_next=received.append)

        subject.on_next(1)
        assert received == [1]

        subscription.unsubscribe()
        subject.on_next(2)
        assert received == [1]


def _fake_session_manager(runner: object) -> MagicMock:
    """A session_manager whose session()/run() are async context managers."""

    @contextlib.asynccontextmanager
    async def _session(**_kwargs: object) -> AsyncIterator[None]:
        yield

    @contextlib.asynccontextmanager
    async def _run(_query: str) -> AsyncIterator[object]:
        yield runner

    sm = MagicMock()
    sm.session = _session
    sm.run = _run
    return sm


def _patch_context(
    monkeypatch: pytest.MonkeyPatch, subscription: MagicMock, *, complete: bool,
) -> None:
    """Patch service.Context so subscribe() returns `subscription`.

    When `complete`, on_complete fires immediately so forward_steps drains and
    the success path runs; otherwise the run must raise to exit.
    """
    def _subscribe(on_next: Any, on_complete: Any) -> MagicMock:  # noqa: ARG001
        if complete:
            on_complete()
        return subscription

    ism = MagicMock()
    ism.subscribe.side_effect = _subscribe
    fake_context = MagicMock()
    fake_context.get.return_value = MagicMock(intermediate_step_manager=ism)
    monkeypatch.setattr(service_module, "Context", fake_context)


class TestRunAgentSubscriptionCleanup:
    """Pins the ACTUAL run_agent cleanup, not just the Subject primitive pattern."""

    async def test_success_path_unsubscribes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run_agent disposes the step subscription on the success path."""
        class _Runner:
            async def result(self, *, to_type: type) -> str:  # noqa: ARG002
                return "done"

        subscription = MagicMock()
        _patch_context(monkeypatch, subscription, complete=True)
        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await run_agent(send, "q", MagicMock(), _fake_session_manager(_Runner()))

        subscription.unsubscribe.assert_called_once()
        assert any(m["type"] == MessageType.SYSTEM_RESPONSE for m in sent)

    async def test_error_path_unsubscribes_and_cancels(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On a failing run, the subscription is still disposed and HITL cancelled."""
        class _BoomRunner:
            async def result(self, *, to_type: type) -> str:  # noqa: ARG002
                msg = "boom"
                raise RuntimeError(msg)

        subscription = MagicMock()
        _patch_context(monkeypatch, subscription, complete=False)
        bridge = MagicMock()
        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await run_agent(send, "q", bridge, _fake_session_manager(_BoomRunner()))

        subscription.unsubscribe.assert_called_once()
        bridge.cancel_all.assert_called_once()
        assert any(m["type"] == MessageType.ERROR for m in sent)


class TestRunAgentConversationWindow:
    """run_agent binds the prior-prompt window for the run, then clears it."""

    async def test_window_bound_during_run_then_cleared(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """prior_user_prompts is visible to code running inside the run, then reset.

        ``_Runner.result`` stands in for the agent loop (where the classifier
        middleware fires), so reading the window there proves the binding
        reaches the classifier; the post-run assertion proves the reset.
        """
        seen: dict[str, tuple[str, ...]] = {}

        class _Runner:
            async def result(self, *, to_type: type) -> str:  # noqa: ARG002
                seen["window"] = get_recent_user_prompts()
                return "done"

        _patch_context(monkeypatch, MagicMock(), complete=True)

        async def send(_msg: dict[str, Any]) -> None:
            return

        await run_agent(
            send,
            "deploy it",
            MagicMock(),
            _fake_session_manager(_Runner()),
            prior_user_prompts=["read the config"],
        )

        assert seen["window"] == ("read the config",)  # bound during the run
        assert get_recent_user_prompts() == ()  # cleared after
