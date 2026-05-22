"""Tests for the service layer's IntermediateStep → WS-wire translation.

NAT 1.7.0 emits LLM-tier ``IntermediateStep``s natively (model latency +
token-by-token), where ``payload.name`` is the *model* name and the token text
lives in ``data.chunk``. The gateway translates those into the stable wire
names the UI routes on (``agent:thinking`` for the chat bubble, ``LLM_START`` /
``LLM_END`` for the trace), so the frontend never sees raw model names.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

from src.core.protocol import MessageType
from src.loop.service import _make_forward_steps, _wire_step


def _data(**fields: Any) -> SimpleNamespace:
    """A stand-in for NAT's StreamEventData with only the fields under test."""
    fields.setdefault("payload", None)
    fields.setdefault("input", None)
    fields.setdefault("output", None)
    return SimpleNamespace(**fields)


def _step(
    event_type: str,
    name: str,
    *,
    data: SimpleNamespace,
    span: str = "u1",
) -> SimpleNamespace:
    """A stand-in for a NAT IntermediateStep (only ``.payload`` is read)."""
    return SimpleNamespace(
        payload=SimpleNamespace(
            event_type=SimpleNamespace(name=event_type),
            name=name,
            data=data,
            UUID=span,
        ),
    )


class TestWireStep:
    """``_wire_step`` maps NAT event types to UI-routable wire names."""

    def test_llm_new_token_becomes_thinking_delta(self) -> None:
        """A token event routes to the chat bubble carrying only the new chunk."""
        step = _step("LLM_NEW_TOKEN", "z-ai/glm-5.1", data=_data(input="full prompt", chunk="hel"))
        name, payload = _wire_step(step.payload)
        assert name == "agent:thinking"
        assert payload == "hel"

    def test_llm_start_is_content_free(self) -> None:
        """LLM_START routes to the trace LLM category with NO payload.

        The native event's data.input is the full ReAct prompt (system prompt +
        tool schemas + conversation); forwarding it would leak the prompt into
        the browser trace, so the boundary card carries only the latency pairing.
        """
        step = _step("LLM_START", "z-ai/glm-5.1", data=_data(input="SYSTEM PROMPT"))
        name, payload = _wire_step(step.payload)
        assert name == "LLM_START"
        assert payload == ""

    def test_llm_end_is_content_free(self) -> None:
        """LLM_END routes to the trace LLM category with no payload (answer is the response)."""
        step = _step("LLM_END", "z-ai/glm-5.1", data=_data(output="done"))
        name, payload = _wire_step(step.payload)
        assert name == "LLM_END"
        assert payload == ""

    def test_tool_event_keeps_payload_name(self) -> None:
        """Tool/classifier events already carry a UI-meaningful name; pass it through."""
        name, _ = _wire_step(
            _step("FUNCTION_START", "getglad_tools__read_file", data=_data(input="x")).payload,
        )
        assert name == "getglad_tools__read_file"

    def test_classifier_event_keeps_payload_name(self) -> None:
        """The classifier's own CUSTOM_START name (rules:allow) passes through."""
        name, _ = _wire_step(_step("CUSTOM_START", "rules:allow", data=_data()).payload)
        assert name == "rules:allow"

    def test_empty_token_yields_empty_payload(self) -> None:
        """An empty token still maps to thinking with an empty delta (caller skips it)."""
        name, payload = _wire_step(_step("LLM_NEW_TOKEN", "m", data=_data(chunk="")).payload)
        assert name == "agent:thinking"
        assert payload == ""


class TestForwardSteps:
    """``forward_steps`` drains the step queue and emits translated WS frames."""

    async def _drain(self, *steps: SimpleNamespace | None) -> list[dict[str, Any]]:
        """Run forward_steps over ``steps`` (None = stream-complete sentinel)."""
        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        queue: asyncio.Queue[Any] = asyncio.Queue()
        for step in steps:
            queue.put_nowait(step)
        await _make_forward_steps(send, "run-1", queue)()
        return sent

    async def test_token_translated_to_thinking_with_span_id(self) -> None:
        """A token frame is an in-progress thinking delta keyed by the LLM run id."""
        sent = await self._drain(
            _step("LLM_NEW_TOKEN", "z-ai/glm-5.1", data=_data(chunk="hi"), span="llm-7"),
            None,
        )
        inter = [m for m in sent if m["type"] == MessageType.SYSTEM_INTERMEDIATE]
        assert len(inter) == 1
        assert inter[0]["content"] == {"name": "agent:thinking", "payload": "hi"}
        assert inter[0]["id"] == "llm-7"
        assert inter[0]["status"] == "in_progress"

    async def test_empty_token_is_skipped(self) -> None:
        """An empty token delta is dropped so it can't open a blank thinking bubble."""
        sent = await self._drain(
            _step("LLM_NEW_TOKEN", "m", data=_data(chunk="")),
            None,
        )
        assert [m for m in sent if m["type"] == MessageType.SYSTEM_INTERMEDIATE] == []

    async def test_llm_end_marked_complete(self) -> None:
        """LLM_END forwards as a complete trace step under its event-type name."""
        sent = await self._drain(
            _step("LLM_END", "z-ai/glm-5.1", data=_data(output="done"), span="llm-7"),
            None,
        )
        inter = [m for m in sent if m["type"] == MessageType.SYSTEM_INTERMEDIATE]
        assert len(inter) == 1
        assert inter[0]["content"]["name"] == "LLM_END"
        assert inter[0]["status"] == "complete"
        assert inter[0]["id"] == "llm-7"
