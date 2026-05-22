"""Agent loop endpoints — REST health/status + WebSocket agent interaction."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from src.core.protocol import MessageType, extract_query, ws_msg
from src.server.hitl_bridge import WebSocketHITLBridge

if TYPE_CHECKING:
    from nat.runtime.session import SessionManager

# Literal value of `src.loop.hitl.APPROVE_OPTION.value`. Inlined so this
# module loads without pulling NAT through the hitl chain — the watcher
# process of `uvicorn --reload` doesn't need NAT at all.
APPROVE_LITERAL = "yes"

# Literal value of `src.loop.react_steps.REACT_WITH_STEPS_TYPE`. Inlined for the
# same reason as APPROVE_LITERAL — importing it would pull NAT into the watcher.
# Reported by /status; keep in sync with the constant if it ever changes.
WORKFLOW_TYPE_LITERAL = "react_agent_with_steps"

# WebSocket policy-violation close code (RFC 6455).
_WS_POLICY_VIOLATION = 1008

# RFC 6455 subprotocol tokens use the RFC 7230 token charset — no comma,
# whitespace, or base64 +/=. A token outside it can't be presented as a
# subprotocol, so it would silently never authenticate.
_WS_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

logger = structlog.get_logger()

router = APIRouter()


def warn_if_ws_token_malformed() -> None:
    """Warn at startup if GATEWAY_WS_TOKEN can't be a valid WS subprotocol."""
    token = os.environ.get("GATEWAY_WS_TOKEN")
    if token and not _WS_TOKEN_RE.match(token):
        logger.warning(
            "gateway_ws_token_invalid",
            reason="GATEWAY_WS_TOKEN must be an RFC 6455 subprotocol token "
            "(no comma, whitespace, or +/=); non-browser auth will fail",
        )


def _allowed_origins() -> set[str]:
    """Origins permitted to open the WS, from env (default: the Next dev server)."""
    raw = os.environ.get(
        "GATEWAY_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3001",
    )
    return {o.strip() for o in raw.split(",") if o.strip()}


def _offered_subprotocols(websocket: WebSocket) -> set[str]:
    """The Sec-WebSocket-Protocol values the client offered."""
    raw = websocket.headers.get("sec-websocket-protocol", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _ws_authorized(websocket: WebSocket) -> bool:
    """Authorize a WS handshake by client type.

    Browser clients send a truthful ``Origin`` a malicious page cannot forge, so
    an origin allowlist fully defends against Cross-Site WebSocket Hijacking — no
    token (a token in the browser bundle couldn't be secret). Non-browser clients
    send no ``Origin``; authenticate them with the server-only ``GATEWAY_WS_TOKEN``
    (if set) presented via ``Sec-WebSocket-Protocol`` (kept out of the URL/logs).
    """
    origin = websocket.headers.get("origin")
    if origin is not None:
        if origin not in _allowed_origins():
            logger.warning("ws_rejected_origin", origin=origin)
            return False
        return True
    token = os.environ.get("GATEWAY_WS_TOKEN")
    if token and token not in _offered_subprotocols(websocket):
        logger.warning("ws_rejected_token")
        return False
    return True


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check — returns ok if the server is running."""
    return {"status": "ok"}


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    """Server and workflow metadata."""
    sm = getattr(request.app.state, "session_manager", None)
    return {
        "version": request.app.version,
        "workflow": {
            "type": WORKFLOW_TYPE_LITERAL,
            "status": "ready" if sm is not None else "starting",
        },
    }


async def _start_agent_run(
    websocket: WebSocket,
    data: dict[str, Any],
    bridge: WebSocketHITLBridge,
    prior_task: asyncio.Task[None] | None,
) -> asyncio.Task[None] | None:
    """Validate a user_message and spawn the agent run. Returns the new task (or prior on error)."""
    query = extract_query(data)
    if query is None:
        await websocket.send_json(ws_msg(MessageType.ERROR, "Invalid message structure."))
        return prior_task
    if not query:
        await websocket.send_json(ws_msg(MessageType.ERROR, "Empty query."))
        return prior_task

    sm: SessionManager | None = getattr(websocket.app.state, "session_manager", None)
    if sm is None:
        await websocket.send_json(
            ws_msg(MessageType.ERROR, "Server not ready — workflow still building."),
        )
        return prior_task

    if prior_task is not None and not prior_task.done():
        prior_task.cancel()
        # Clear the superseded run's pending HITL futures so a late reply to the
        # old prompt can't resolve a cancelled future (and so they don't linger).
        bridge.cancel_all()

    # Deferred so router import doesn't cascade into NAT — see APPROVE_LITERAL.
    from src.loop.service import run_agent

    return asyncio.create_task(run_agent(websocket.send_json, query, bridge, sm))


async def _accept_ws(websocket: WebSocket) -> bool:
    """Authorize (CSWSH origin / non-browser token) then accept. False ⇒ rejected."""
    if not _ws_authorized(websocket):
        await websocket.close(code=_WS_POLICY_VIOLATION)
        return False
    # Echo the token subprotocol only when a non-browser client actually offered
    # it — the handshake requires the server to select an offered subprotocol.
    token = os.environ.get("GATEWAY_WS_TOKEN")
    if token and token in _offered_subprotocols(websocket):
        await websocket.accept(subprotocol=token)
    else:
        await websocket.accept()
    return True


async def _handle_client_message(
    websocket: WebSocket,
    data: dict[str, Any],
    bridge: WebSocketHITLBridge,
    agent_task: asyncio.Task[None] | None,
) -> asyncio.Task[None] | None:
    """Dispatch one validated client message; returns the (maybe updated) agent task."""
    msg_type = data.get("type")
    if msg_type == MessageType.USER_MESSAGE:
        return await _start_agent_run(websocket, data, bridge, agent_task)
    if msg_type == MessageType.USER_INTERACTION:
        parent_id = data.get("parent_id", "")
        response_text = extract_query(data) or ""
        approved = response_text.strip().lower() == APPROVE_LITERAL
        bridge.resolve_approval(parent_id, approved=approved)
        return agent_task
    await websocket.send_json(ws_msg(MessageType.ERROR, f"Unknown message type: {msg_type}"))
    return agent_task


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for agent loop interaction.

    Protocol (matches NAT message types):
    - Client sends: user_message, user_interaction_message
    - Server sends: system_intermediate_message, system_interaction_message,
                    system_response_message, error_message
    """
    if not await _accept_ws(websocket):
        return

    session_id = str(uuid.uuid4())
    bridge = WebSocketHITLBridge()
    agent_task: asyncio.Task[None] | None = None

    logger.info("ws_connected", session_id=session_id)

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                # Malformed text or a binary frame — reject the message, keep the
                # connection (WebSocketDisconnect still propagates to the outer except).
                await websocket.send_json(
                    ws_msg(MessageType.ERROR, "Message must be valid JSON."),
                )
                continue
            if not isinstance(data, dict):
                await websocket.send_json(
                    ws_msg(MessageType.ERROR, "Message must be a JSON object."),
                )
                continue
            agent_task = await _handle_client_message(websocket, data, bridge, agent_task)

    except WebSocketDisconnect:
        logger.info("ws_disconnected", session_id=session_id)
    finally:
        bridge.cancel_all()
        if agent_task is not None and not agent_task.done():
            agent_task.cancel()
            # Let NAT session teardown (and OTel flush) complete before returning.
            # Bounded — if teardown hangs we abandon the task rather than block the WS handler.
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.gather(agent_task, return_exceptions=True)),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning("agent_task_teardown_timeout", session_id=session_id)
