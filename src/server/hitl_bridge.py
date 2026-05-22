"""WebSocket HITL bridge — translates between NAT callbacks and WebSocket clients."""

from __future__ import annotations

import asyncio

import structlog

logger = structlog.get_logger()


class WebSocketHITLBridge:
    """Bridges NAT's HITL callback system to WebSocket clients.

    When the agent requests human approval, this bridge:
    1. Creates an asyncio.Future keyed by message ID
    2. The WebSocket handler sends the prompt to the client
    3. When the client responds, resolve_approval() completes the Future
    4. The NAT callback awaiting the Future gets the result

    Concurrency: all methods mutate ``_pending`` without a lock. They are
    safe only when called from the same event-loop task (the WebSocket
    dispatch loop). Don't call from a threadpool or a sibling task.
    """

    def __init__(self) -> None:
        """Initialize with an empty set of pending approvals."""
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def create_pending_approval(self, message_id: str) -> asyncio.Future[bool]:
        """Register a pending HITL approval and return the Future to await."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[message_id] = future
        logger.debug("hitl_pending", message_id=message_id)
        return future

    def resolve_approval(self, message_id: str, *, approved: bool) -> None:
        """Resolve a pending HITL approval with the user's decision."""
        future = self._pending.pop(message_id, None)
        if future is None:
            logger.warning("hitl_unknown_message", message_id=message_id)
            return
        if future.done():
            # The awaiting task was cancelled (e.g. a superseded agent run), so
            # the future is already settled. set_result() would raise
            # InvalidStateError and tear down the WS dispatch loop — skip it.
            logger.warning("hitl_already_settled", message_id=message_id)
            return
        future.set_result(approved)
        logger.info("hitl_resolved", message_id=message_id, approved=approved)

    def cancel_all(self) -> None:
        """Cancel all pending Futures — called on WebSocket disconnect."""
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
