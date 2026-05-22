"""WebSocket protocol helpers shared across all domains."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MessageType(StrEnum):
    """WebSocket message type discriminator — matches NAT's protocol."""

    USER_MESSAGE = "user_message"
    USER_INTERACTION = "user_interaction_message"
    SYSTEM_RESPONSE = "system_response_message"
    SYSTEM_INTERMEDIATE = "system_intermediate_message"
    SYSTEM_INTERACTION = "system_interaction_message"
    ERROR = "error_message"


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(tz=UTC).isoformat()


def ws_msg(
    msg_type: MessageType,
    content: str | dict[str, Any],
    *,
    parent_id: str | None = None,
    status: str | None = None,
    msg_id: str | None = None,
) -> dict[str, Any]:
    """Build a WebSocket message envelope."""
    envelope: dict[str, Any] = {
        "type": msg_type,
        "id": msg_id or str(uuid.uuid4()),
        "content": content,
        "timestamp": now_iso(),
    }
    if parent_id is not None:
        envelope["parent_id"] = parent_id
    if status is not None:
        envelope["status"] = status
    return envelope


def extract_query(data: dict[str, Any]) -> str | None:
    """Pull the user's text from a user_message envelope.

    Returns:
        The text content if found, or None if the message structure is invalid.
    """
    content = data.get("content")
    if not isinstance(content, dict):
        return None
    messages = content.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    return text
    return None
