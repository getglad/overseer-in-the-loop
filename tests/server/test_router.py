"""Tests for the FastAPI gateway REST + WebSocket endpoints."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from src.core.conversation import MAX_PROMPT_HISTORY
from src.loop.hitl import APPROVE_OPTION
from src.server.router import APPROVE_LITERAL, _start_agent_run, _ws_authorized

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class _FakeWS:
    """Minimal stand-in exposing the case-insensitive ``.headers`` _ws_authorized reads."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class TestHealthEndpoints:
    """Tests for REST health and status endpoints."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """WHEN GET /health THEN returns 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_status_returns_ready_when_session_manager_set(self, client: TestClient) -> None:
        """WHEN lifespan has run THEN /status reports workflow as ready."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "0.1.0"
        assert data["workflow"]["status"] == "ready"


class TestApprovalLiteralPin:
    """Pins router's inlined ``APPROVE_LITERAL`` to the canonical option value.

    The router intentionally inlines the literal instead of importing
    ``APPROVE_OPTION`` (to keep module-load NAT-free). If anyone changes
    ``APPROVE_OPTION.value`` without updating ``APPROVE_LITERAL``, every
    user approval would silently become a rejection. This test catches it.
    """

    def test_literal_matches_approve_option_value(self) -> None:
        """WHEN APPROVE_OPTION changes THEN router's literal MUST follow."""
        assert APPROVE_OPTION.value == APPROVE_LITERAL, (
            f"router.APPROVE_LITERAL ({APPROVE_LITERAL!r}) drifted from "
            f"APPROVE_OPTION.value ({APPROVE_OPTION.value!r}). Update one."
        )


class TestWebSocketAuthorization:
    """CSWSH origin allowlist (browsers) + shared token (non-browser clients)."""

    def test_allowlisted_origin_authorized(self) -> None:
        """A browser from an allowlisted origin is authorized (no token needed)."""
        assert _ws_authorized(_FakeWS({"origin": "http://localhost:3000"})) is True

    def test_cross_site_origin_rejected(self) -> None:
        """A browser from an off-allowlist origin is rejected (the CSWSH case)."""
        assert _ws_authorized(_FakeWS({"origin": "http://evil.example"})) is False

    def test_no_origin_no_token_allowed(self) -> None:
        """A non-browser client (no Origin) is allowed when no token is configured."""
        assert _ws_authorized(_FakeWS({})) is True

    def test_no_origin_token_required_but_missing_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With a token configured, a non-browser client must present it."""
        monkeypatch.setenv("GATEWAY_WS_TOKEN", "s3cret")
        assert _ws_authorized(_FakeWS({})) is False

    def test_no_origin_token_presented_authorized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-browser client presenting the token via subprotocol is authorized."""
        monkeypatch.setenv("GATEWAY_WS_TOKEN", "s3cret")
        assert _ws_authorized(_FakeWS({"sec-websocket-protocol": "other, s3cret"})) is True

    def test_browser_origin_skips_token_requirement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token configured for non-browser auth must NOT lock out the browser UI."""
        monkeypatch.setenv("GATEWAY_WS_TOKEN", "s3cret")
        assert _ws_authorized(_FakeWS({"origin": "http://localhost:3000"})) is True

    def test_custom_allowlist_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GATEWAY_ALLOWED_ORIGINS overrides the localhost default."""
        monkeypatch.setenv("GATEWAY_ALLOWED_ORIGINS", "https://app.example.com")
        assert _ws_authorized(_FakeWS({"origin": "https://app.example.com"})) is True
        assert _ws_authorized(_FakeWS({"origin": "http://localhost:3000"})) is False


class TestWebSocketHandshakeAndFraming:
    """End-to-end handshake rejection + malformed-frame handling over the live WS."""

    def test_cross_site_origin_closes_connection(self, client: TestClient) -> None:
        """A cross-site Origin is rejected at the handshake, not upgraded."""
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws,
        ):
            ws.receive_json()

    def test_malformed_frame_returns_error_keeps_connection(self, client: TestClient) -> None:
        """A non-JSON frame yields an error message instead of tearing down the socket."""
        headers = {"origin": "http://localhost:3000"}  # allowlisted, so the handshake passes
        with client.websocket_connect("/ws", headers=headers) as ws:
            ws.send_text("this is not json{")
            reply = ws.receive_json()
            assert reply["type"] == "error_message"


def _user_message(text: str) -> dict[str, Any]:
    """Build a minimal user_message envelope that ``extract_query`` accepts."""
    return {"content": {"messages": [{"content": [{"type": "text", "text": text}]}]}}


class _StubWS:
    """WebSocket stand-in for ``_start_agent_run``: app.state + async send_json."""

    def __init__(self) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(session_manager=object()))
        self.send_json = AsyncMock()


class TestConversationWindowPlumbing:
    """The router's per-connection prompt window feeds the guardrail classifier.

    This is the riskiest new orchestration in the change: the snapshot-prior-
    THEN-append ordering is what keeps the current request out of its own prior
    context, and the per-connection deque is what keeps one conversation's turns
    out of another's guard decisions. A reorder/scope regression here would pass
    the rest of the suite, so it gets a direct test.
    """

    @staticmethod
    def _capture_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
        """Stub run_agent to record (query, prior_user_prompts); neutralize the evil toggle."""
        captured: list[tuple[str, tuple[str, ...]]] = []

        async def _stub_run_agent(
            _send: Any, query: str, _bridge: Any, _sm: Any, *, prior_user_prompts: Any = (),
        ) -> None:
            captured.append((query, tuple(prior_user_prompts)))

        monkeypatch.setattr("src.loop.service.run_agent", _stub_run_agent)
        monkeypatch.setattr("src.guardrails.middleware.set_evil_toggle", lambda **_: None)
        return captured

    @staticmethod
    async def _run_turn(
        ws: _StubWS, history: deque[str], text: str, prior_task: asyncio.Task[None] | None = None,
    ) -> asyncio.Task[None] | None:
        """Drive one user_message through the real _start_agent_run and let it complete."""
        task = await _start_agent_run(ws, _user_message(text), MagicMock(), prior_task, history)  # type: ignore[arg-type]
        if task is not None:
            await task
        return task

    async def test_prior_window_excludes_the_current_turn(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Turn N's prior window is the EARLIER turns — never the current request itself."""
        captured = self._capture_runs(monkeypatch)
        ws, history = _StubWS(), deque(maxlen=MAX_PROMPT_HISTORY)
        for text in ("read config", "deploy it", "check status"):
            await self._run_turn(ws, history, text)
        assert [prior for _q, prior in captured] == [
            (),
            ("read config",),
            ("read config", "deploy it"),
        ]

    async def test_window_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prior window never exceeds MAX_PROMPT_HISTORY (oldest turns drop)."""
        captured = self._capture_runs(monkeypatch)
        ws, history = _StubWS(), deque(maxlen=MAX_PROMPT_HISTORY)
        for i in range(MAX_PROMPT_HISTORY + 3):
            await self._run_turn(ws, history, f"turn{i}")
        assert all(len(prior) <= MAX_PROMPT_HISTORY for _q, prior in captured)
        assert len(captured[-1][1]) == MAX_PROMPT_HISTORY

    async def test_each_connection_starts_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second connection's deque is independent — no cross-connection history bleed."""
        captured = self._capture_runs(monkeypatch)
        ws = _StubWS()
        await self._run_turn(ws, deque(maxlen=MAX_PROMPT_HISTORY), "conn-1 turn")
        await self._run_turn(ws, deque(maxlen=MAX_PROMPT_HISTORY), "conn-2 turn")
        assert captured[1] == ("conn-2 turn", ())

    async def test_superseded_turn_stays_in_history(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A turn whose run is superseded still appears as prior context next turn."""
        captured = self._capture_runs(monkeypatch)
        ws, history = _StubWS(), deque(maxlen=MAX_PROMPT_HISTORY)
        hanging: asyncio.Task[None] = asyncio.create_task(asyncio.Event().wait())  # type: ignore[arg-type]
        try:
            await self._run_turn(ws, history, "first request")
            await self._run_turn(ws, history, "second request", prior_task=hanging)
        finally:
            hanging.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hanging
        assert captured[1] == ("second request", ("first request",))
        assert hanging.cancelled()
