"""Agent execution service — runs the NAT agent and streams results.

Transport-agnostic: takes a `send` callback, not a WebSocket object.
The router passes `websocket.send_json`; tests pass a list-accumulating mock.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog
from nat.builder.context import Context
from nat.data_models.interactive import (
    HumanPromptBinary,
    HumanResponse,
    HumanResponseBinary,
    InteractionPrompt,
)
from openai import APIConnectionError, APIStatusError

from src.core.conversation import reset_recent_user_prompts, set_recent_user_prompts
from src.core.protocol import MessageType, ws_msg
from src.loop.hitl import APPROVAL_TIMEOUT, APPROVE_OPTION, REJECT_OPTION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nat.runtime.session import SessionManager

    from src.server.hitl_bridge import WebSocketHITLBridge

logger = structlog.get_logger()

type SendFn = Callable[[dict[str, Any]], Awaitable[None]]

# Max chars of LLM provider's error message we forward to the client.
_MAX_DETAIL_CHARS = 200

# Strip credentials a misconfigured upstream proxy might echo back in error
# bodies before we relay them to the WebSocket client.
_CREDENTIAL_REDACTORS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
)


def _redact_credentials(text: str) -> str:
    for pattern in _CREDENTIAL_REDACTORS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _explain_llm_failure(exc: BaseException) -> str | None:
    """Walk an exception chain for a known LLM-side error and translate it.

    NAT/LangGraph wraps LLM API failures inside several retry/aggregator
    layers, so the user-relevant cause usually lives in ``__cause__`` or
    ``__context__``. Returns a short, actionable message if we recognize
    the error type, or None if we don't (caller falls back to a generic
    message).
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, APIStatusError):
            # Redact credentials BEFORE truncating — otherwise a token straddling
            # the char cap could leak a partial. Trim runaway tracebacks/JSON
            # dumps LiteLLM sometimes attaches after redaction.
            first_line = current.message.strip().splitlines()[0]
            detail = _redact_credentials(first_line)[:_MAX_DETAIL_CHARS]
            hint = (
                "Check LLM_MODEL is currently available and that LLM_API_KEY / "
                "LLM_BASE_URL in mise.local.toml are correct."
            )
            return f"LLM API returned {current.status_code}: {detail}. {hint}"

        if isinstance(current, APIConnectionError):
            return (
                "Could not reach the LLM endpoint. Check LLM_BASE_URL in "
                "mise.local.toml and your network connection."
            )

        current = current.__cause__ or current.__context__

    return None


# Wire name for the streamed LLM token deltas. The UI's AgentLoopPanel routes
# frames with this name into ONE growing "thinking" chat bubble (merged by the
# message id) instead of the tool/guardrail trace panel.
_THINKING_NAME = "agent:thinking"

# LLM-tier event types whose wire name must be the event type, not the model
# name NAT puts in ``payload.name`` — the UI's step categories key on these.
_LLM_BOUNDARY_EVENTS = frozenset({"LLM_START", "LLM_END"})


def _wire_step(payload: Any) -> tuple[str, str]:  # noqa: ANN401 — NAT IntermediateStepPayload (untyped)
    """Translate a NAT IntermediateStep payload to the (name, payload) the UI expects.

    NAT's native LLM events carry the *model* in ``payload.name`` (e.g.
    ``z-ai/glm-5.1``) and the token in ``data.chunk``. The UI routes on the
    event *type*: token streams build one growing thinking bubble; start/end
    bound a paired trace row. Tool and guardrail events already carry a
    UI-meaningful ``payload.name`` (``read_file``, ``rules:allow``) and pass
    through untouched.
    """
    event = payload.event_type.name
    if event == "LLM_NEW_TOKEN":
        chunk = getattr(payload.data, "chunk", None)
        return _THINKING_NAME, str(chunk) if chunk else ""
    if event in _LLM_BOUNDARY_EVENTS:
        # No payload: the native LLM_START's data.input is the FULL ReAct prompt
        # (system prompt + tool schemas + conversation). The reasoning streams via
        # agent:thinking and the answer is the response, so the boundary card only
        # needs to pair START/END for latency — forwarding the prompt would leak it
        # into the browser trace.
        return event, ""
    return (payload.name or event), _format_step_data(payload.data)


def _format_step_data(data: object) -> str:
    """Extract readable fields from a NAT StreamEventData object.

    Produces "key: value" lines that the UI can parse into labeled fields.
    """
    if data is None:
        return ""

    payload_val = getattr(data, "payload", None)
    if payload_val:
        return str(payload_val)

    parts: list[str] = []
    input_val = getattr(data, "input", None)
    output_val = getattr(data, "output", None)

    if input_val is not None:
        parts.append(f"input: {input_val}")
    if output_val is not None:
        parts.append(f"output: {output_val}")

    if parts:
        return "\n".join(parts)

    return str(data)


def _make_hitl_callback(
    send: SendFn,
    run_id: str,
    bridge: WebSocketHITLBridge,
) -> Callable[[InteractionPrompt], Awaitable[HumanResponse]]:
    """Build the HITL callback NAT invokes when the agent requests approval."""

    async def hitl_callback(prompt: InteractionPrompt) -> HumanResponse:
        msg_id = str(uuid.uuid4())

        if isinstance(prompt.content, HumanPromptBinary):
            content: dict[str, Any] = {
                "input_type": "binary_choice",
                "text": prompt.content.text,
                "options": [o.model_dump() for o in prompt.content.options],
            }
        else:
            content = {"input_type": "text", "text": prompt.content.text}

        await send(
            ws_msg(
                MessageType.SYSTEM_INTERACTION,
                content,
                parent_id=run_id,
                status="in_progress",
                msg_id=msg_id,
            ),
        )

        future = bridge.create_pending_approval(msg_id)
        try:
            # Enforce APPROVAL_TIMEOUT here — the prompt carries it but NAT
            # doesn't enforce it for the WS callback, so without this a
            # never-answered approval hangs the run forever. Timeout ⇒ reject.
            approved = await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT)
        except TimeoutError:
            logger.warning("hitl_timeout", msg_id=msg_id, timeout_s=APPROVAL_TIMEOUT)
            bridge.resolve_approval(msg_id, approved=False)  # pop the cancelled future
            approved = False

        selected = APPROVE_OPTION if approved else REJECT_OPTION
        return HumanResponseBinary(selected_option=selected)

    return hitl_callback


def _make_forward_steps(
    send: SendFn,
    run_id: str,
    step_queue: asyncio.Queue[Any],
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Build the task that drains the step queue and streams events to the client."""

    async def forward_steps() -> None:
        while True:
            step = await step_queue.get()
            if step is None:
                break
            try:
                name, payload = _wire_step(step.payload)
                # Empty token deltas (NAT emits a few) would open a blank
                # thinking bubble — drop them before they reach the client.
                if name == _THINKING_NAME and not payload:
                    continue
                logger.debug(
                    "step_received",
                    name=name,
                    event_type=step.payload.event_type.name,
                    payload_preview=payload[:200] if payload else "",
                )
                is_complete = step.payload.event_type.name.endswith("_END")
                await send(
                    ws_msg(
                        MessageType.SYSTEM_INTERMEDIATE,
                        {"name": name, "payload": payload},
                        parent_id=run_id,
                        status="complete" if is_complete else "in_progress",
                        # Carry the step's span UUID as the message id so the UI
                        # merges deltas/updates into one bubble and pairs
                        # LLM_START/LLM_END — without it every event gets a fresh
                        # id and renders as a separate fragment.
                        msg_id=step.payload.UUID,
                    ),
                )
            except Exception:
                logger.exception("step_forward_error")

    return forward_steps


async def run_agent(
    send: SendFn,
    query: str,
    bridge: WebSocketHITLBridge,
    session_manager: SessionManager,
    *,
    prior_user_prompts: Sequence[str] = (),
) -> None:
    """Run the NAT agent and stream steps to the client.

    Args:
        send: Async callable that delivers a JSON-serializable dict to the client.
        query: The user's query text.
        bridge: HITL bridge for managing approval Futures.
        session_manager: NAT session manager for running the agent.
        prior_user_prompts: Earlier USER turns in this conversation (oldest first),
            bound task-locally so the guardrail classifier can judge the current
            action against the conversation's goal. Excludes agent reasoning and
            tool output by design (those can carry injected content).
    """
    run_id = str(uuid.uuid4())
    step_queue: asyncio.Queue[Any] = asyncio.Queue()
    hitl_callback = _make_hitl_callback(send, run_id, bridge)
    forward_steps = _make_forward_steps(send, run_id, step_queue)
    # Bind the conversation window for this run's task; the classifier middleware
    # (running inside the runner below) reads it via core.conversation.
    history_token = set_recent_user_prompts(prior_user_prompts)

    try:
        async with (
            session_manager.session(user_input_callback=hitl_callback),
            session_manager.run(query) as runner,
        ):
            ctx = Context.get()
            # Capture for unsubscribe in finally — observer closure holds
            # step_queue ref otherwise (regression test pins this).
            subscription = ctx.intermediate_step_manager.subscribe(
                on_next=step_queue.put_nowait,
                on_complete=lambda: step_queue.put_nowait(None),
            )

            forward_task = asyncio.create_task(forward_steps())
            try:
                result = await runner.result(to_type=str)
                await forward_task

                await send(
                    ws_msg(
                        MessageType.SYSTEM_RESPONSE,
                        result,
                        parent_id=run_id,
                        status="complete",
                    ),
                )
                logger.info("agent_run_complete", run_id=run_id)
            finally:
                subscription.unsubscribe()
                # Guarantee the forwarder is torn down before we leave (on error
                # or cancellation runner.result raised, so it's still awaiting the
                # queue) — otherwise it leaks and can emit frames after the run.
                if not forward_task.done():
                    forward_task.cancel()
                await asyncio.gather(forward_task, return_exceptions=True)

    except Exception as exc:
        logger.exception("agent_run_error", run_id=run_id)
        # Cancel any HITL Futures awaiting a reply — otherwise they leak on this run.
        bridge.cancel_all()
        # Translate known LLM API failures into actionable client messages
        # so the UI surfaces the cause instead of "check server logs".
        message = _explain_llm_failure(exc) or "Agent run failed. Check server logs."
        await send(ws_msg(MessageType.ERROR, message))
    finally:
        reset_recent_user_prompts(history_token)
