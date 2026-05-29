"""NeMo Guardrails action classifier — Layer 2 of the action evaluator.

Layer 1 (``rules.py``) makes the instant call when the answer is obvious;
this module handles the ambiguous cases by routing them through NeMo's
``self_check_output`` output rail. The LLM evaluates the proposed tool
call against the A1-A7 action-safety taxonomy and the user's stated goal.

NeMo convention: ``output_mapping=lambda v: not v`` means the LLM's
"yes" inverts to ``blocked``. The prompt template in ``prompts.py``
phrases the question accordingly ("is this unsafe?", not "is this safe?").
"""

from __future__ import annotations

import contextlib
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from nat.builder.context import Context
from nat.data_models.intermediate_step import (
    IntermediateStepPayload,
    IntermediateStepType,
    StreamEventData,
)
from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.config import Model, RailsConfig, TaskPrompt
from nemoguardrails.rails.llm.options import RailType

from src.core.conversation import MAX_PROMPT_CHARS
from src.guardrails.prompts import ACTION_EVAL_TEMPLATE
from src.guardrails.rules import Decision, check_rules

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger()

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_EVAL_MODEL = "z-ai/glm-5.1"

# Layer labels for ClassifyResult and trace events — single source of truth.
LAYER_RULES = "rules"
LAYER_GUARDRAIL_AGENT = "guardrail-agent"

# Pre-flight cap on the args payload sent to the LLM. A multi-MB write would
# blow the model's context window AND bill us for the input tokens on the way
# to an error — so we gate on our side and escalate to HITL without a NIM call.
_MAX_LLM_EVAL_BYTES = 32 * 1024

# Per-value cap for args echoed into the audit trail (which reaches the WS
# client). The LLM sees full content; the browser sees a bounded preview so a
# write of a secrets-laden file doesn't mirror its bytes back to the UI.
_MAX_AUDIT_VALUE_CHARS = 200


def build_rails(
    *,
    api_key: str,
    model_name: str = DEFAULT_EVAL_MODEL,
    base_url: str = NVIDIA_NIM_BASE_URL,
) -> LLMRails:
    """Build the NeMo Guardrails instance with typed Python config.

    Args:
        api_key: API key for the evaluation model. Caller reads from env
            so this library stays env-free.
        model_name: Evaluation model for the output rail.
        base_url: LLM API base URL.

    Returns:
        Configured ``LLMRails`` instance ready for ``check_async`` calls.
    """
    config = RailsConfig.from_content(config={
        "models": [
            Model(
                type="main",
                engine="openai",
                model=model_name,
                parameters={"base_url": base_url, "api_key": api_key},
            ).model_dump(),
        ],
        "rails": {
            "output": {"flows": ["self check output"]},
        },
        "prompts": [
            TaskPrompt(
                task="self_check_output",
                content=ACTION_EVAL_TEMPLATE,
            ).model_dump(),
        ],
    })

    return LLMRails(config)


@dataclass(frozen=True)
class AuditTrail:
    """Reasoning chain for an LLM-layer classification event.

    Only LLM-tier decisions carry this — rules decisions are deterministic
    and have no model or prompt to audit. Bundled rather than splayed across
    individual kwargs so the emit signature stays cohesive.
    """

    prompt: str
    response: str
    model: str


async def emit_classification_event(
    tool_name: str,
    *,
    decision: str,
    layer: str,
    reason: str,
    audit: AuditTrail | None = None,
) -> None:
    """Push a classification decision into the agent's intermediate-step stream.

    Event name is ``{layer}:{decision}`` — e.g. ``rules:allow`` or
    ``guardrail-agent:block``. Layer values are constrained to
    ``LAYER_RULES`` or ``LAYER_GUARDRAIL_AGENT`` to keep the trace
    vocabulary consistent across UI and backend.

    LLM-tier events carry an ``AuditTrail`` so the trace panel can show
    the prompt, model response, and model name; rules-tier events have
    no such reasoning chain.

    A classification is atomic, so we emit a matched START+END pair with
    a shared UUID — without the END the trace UI shows the badge as
    "running" forever; without the shared UUID the START and END land as
    two separate cards (the UI merges by message id, which the service
    layer maps from this UUID).
    """
    parts = [f"{tool_name} → {decision}", f"reason: {reason}"]
    if audit is not None:
        parts.extend([
            f"prompt: {audit.prompt}",
            f"response: {audit.response}",
            f"model: {audit.model}",
        ])

    span_id = str(uuid.uuid4())
    payload = "\n".join(parts)
    name = f"{layer}:{decision}"

    try:
        ctx = Context.get()
        manager = ctx.intermediate_step_manager
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=span_id,
                event_type=IntermediateStepType.CUSTOM_START,
                name=name,
                data=StreamEventData(payload=payload),
            ),
        )
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=span_id,
                event_type=IntermediateStepType.CUSTOM_END,
                name=name,
                data=StreamEventData(payload=payload),
            ),
        )
    except (LookupError, AttributeError):
        logger.debug("classifier_event_skip", tool=tool_name, reason="no context")


@dataclass(frozen=True)
class ClassifyResult:
    """Outcome of a single classification call."""

    allowed: bool
    layer: str
    reason: str


_MARKER_RE = re.compile(r"\[(current|earlier)\]", re.IGNORECASE)


def _as_turn(text: str) -> str:
    """Reduce one user turn to a single safe line for the classifier prompt.

    Collapses all whitespace (so a turn can't span rendered lines) AND
    neutralizes any literal ``[current]``/``[earlier]`` token in the user's
    text to ``(current)``/``(earlier)`` — otherwise a user could embed the
    marker convention this shares with ``ACTION_EVAL_TEMPLATE`` and forge a
    second turn inline. Length is capped to ``MAX_PROMPT_CHARS`` so a pasted
    blob can't blow the classifier's context/budget.
    """
    collapsed = _MARKER_RE.sub(r"(\1)", " ".join(text.split()))
    return collapsed[:MAX_PROMPT_CHARS]


def format_user_side(current: str, prior: Sequence[str]) -> str:
    """Render the human's side of the conversation for the classifier prompt.

    One line per user turn (see ``_as_turn`` for the per-turn sanitization).
    ``[earlier]`` turns give the classifier proportionality context; the
    ``[current]`` line is the request the action must stay proportionate to.
    Only the human's turns appear here — never agent reasoning or tool output.
    """
    lines = [f"[earlier] {_as_turn(p)}" for p in prior if p.strip()]
    lines.append(f"[current] {_as_turn(current) or 'Agent task'}")
    return "\n".join(lines)


async def classify(
    rails: LLMRails,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    user_context: str = "",
    skip_rules: bool = False,
) -> ClassifyResult:
    """Classify a tool call as allowed or escalated.

    Checks the rules layer first. If the rules return ``NEEDS_JUDGMENT``,
    routes the call through NeMo's output rail for LLM evaluation.

    Any LLM error escalates the call (fail closed) — better an annoying
    HITL prompt than a silent approval of a misjudged action.

    Args:
        rails: Configured ``LLMRails`` instance.
        tool_name: The tool being called.
        tool_args: The tool's arguments.
        user_context: The user's side of the conversation, already assembled by
            the caller via ``format_user_side`` (the active request plus recent
            prior turns). Threaded into the LLM prompt so the model can judge
            proportionality against the goal.
        skip_rules: When True, bypass the rules layer entirely and force
            LLM evaluation. The rules layer is purely a fast-path
            optimization; the LLM tier is the authoritative classifier.
            Used by the evil-toggle demo to surface LLM-tier blocks for
            tool names that would otherwise short-circuit at rules.

    Returns:
        ``ClassifyResult`` with the decision, which layer decided, and
        a human-readable reason.
    """
    if not skip_rules:
        rule_result = check_rules(tool_name, tool_args)

        if rule_result == Decision.ALLOW:
            result = ClassifyResult(
                allowed=True, layer=LAYER_RULES,
                reason="tool is in always-allow set (read-only, no side effects)",
            )
            logger.info("classifier_allow", tool=tool_name, layer=LAYER_RULES)
            await emit_classification_event(
                tool_name, decision="allow", layer=LAYER_RULES, reason=result.reason,
            )
            return result

        if rule_result == Decision.BLOCK:
            result = ClassifyResult(
                allowed=False, layer=LAYER_RULES,
                reason="matches dangerous content pattern",
            )
            logger.info("classifier_block", tool=tool_name, layer=LAYER_RULES)
            await emit_classification_event(
                tool_name, decision="block", layer=LAYER_RULES, reason=result.reason,
            )
            return result

    # Full args for the LLM — it must see file contents, not just metadata;
    # without this, write_file({'file_path': '...'}) hides the bytes being
    # written and the LLM can't judge whether the content is dangerous.
    args_lines_full = "\n".join(f"  {k}: {v}" for k, v in tool_args.items())
    tool_description = f"{tool_name}\n{args_lines_full}"
    user_msg = user_context or "[current] Agent task"

    # Truncated args for the audit trail — this string reaches the WS client,
    # so cap each value to avoid mirroring large/secret file contents to the UI.
    args_lines_audit = "\n".join(
        f"  {k}: {str(v)[:_MAX_AUDIT_VALUE_CHARS]}" for k, v in tool_args.items()
    )
    # Flatten the (possibly multi-turn) user side onto one line for the audit
    # trail — the trace UI splits on newlines, so a raw multi-line value would
    # break the "user request:" field into stray rows.
    eval_context = (
        f"user request: {user_msg.replace(chr(10), ' / ')}\n"
        f"tool: {tool_name}\n"
        f"args:\n{args_lines_audit}"
    )

    model_name = "unknown"
    with contextlib.suppress(AttributeError, IndexError):
        model_name = rails.config.models[0].model

    # Pre-flight size gate: escalate oversize calls to HITL without a NIM
    # round-trip. Protects both the context window and our token budget.
    # Measure encoded bytes, not code points — a multi-byte UTF-8 payload
    # (CJK, emoji) is far larger on the wire than len() suggests, and the
    # cap exists to bound the actual token/context cost.
    eval_bytes = len(tool_description.encode("utf-8"))
    if eval_bytes > _MAX_LLM_EVAL_BYTES:
        oversize = ClassifyResult(
            allowed=False, layer=LAYER_GUARDRAIL_AGENT,
            reason=f"args too large to classify safely ({eval_bytes} bytes)",
        )
        logger.warning(
            "classifier_oversize",
            tool=tool_name,
            size=eval_bytes,
            limit=_MAX_LLM_EVAL_BYTES,
        )
        await emit_classification_event(
            tool_name, decision="block", layer=LAYER_GUARDRAIL_AGENT,
            reason=oversize.reason,
            audit=AuditTrail(
                prompt=eval_context,
                response="not sent — args exceed size limit",
                model=model_name,
            ),
        )
        return oversize

    try:
        rail_result = await rails.check_async(
            messages=[
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": tool_description},
            ],
            rail_types=[RailType.OUTPUT],
        )
        # Fail closed: allow ONLY on an explicit "passed". NeMo's RailStatus
        # is passed/modified/blocked; treating anything-but-blocked as allowed
        # would green-light a "modified" result (rail rewrote the output) and
        # the "unknown" getattr fallback (unexpected result shape). Both should
        # escalate to a human, not auto-run.
        status_val = str(getattr(rail_result.status, "value", "unknown"))
        allowed = status_val == "passed"
        llm_response = "no (safe)" if allowed else "yes (unsafe)"
    except Exception:
        logger.exception("classifier_error", tool=tool_name)
        error_result = ClassifyResult(
            allowed=False, layer=LAYER_GUARDRAIL_AGENT,
            reason="guardrail agent error — escalating to human",
        )
        await emit_classification_event(
            tool_name, decision="block", layer=LAYER_GUARDRAIL_AGENT,
            reason=error_result.reason,
            audit=AuditTrail(prompt=eval_context, response="error", model=model_name),
        )
        return error_result

    decision = "allow" if allowed else "block"
    reason = (
        "action is proportionate to the request"
        if allowed
        else "action appears unsafe or disproportionate to the request"
    )
    logger.info(
        "classifier_judgment",
        tool=tool_name,
        status=status_val,
        allowed=allowed,
        layer=LAYER_GUARDRAIL_AGENT,
    )
    classify_result = ClassifyResult(
        allowed=allowed, layer=LAYER_GUARDRAIL_AGENT, reason=reason,
    )
    await emit_classification_event(
        tool_name, decision=decision, layer=LAYER_GUARDRAIL_AGENT,
        reason=classify_result.reason,
        audit=AuditTrail(prompt=eval_context, response=llm_response, model=model_name),
    )
    return classify_result
