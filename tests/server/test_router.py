"""Tests for the FastAPI gateway REST + WebSocket endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import WebSocketDisconnect

from src.loop.hitl import APPROVE_OPTION
from src.server.router import APPROVE_LITERAL, _ws_authorized

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
