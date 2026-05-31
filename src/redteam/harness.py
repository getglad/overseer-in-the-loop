"""Run attacks through the action gate and aggregate the verdicts.

Transport-agnostic and non-streaming: ``run_attack`` drives one probe,
``run_corpus`` drives the whole battery and returns a ``Scorecard``. The
streaming service wraps these for the WebSocket UI; the CLI calls ``run_corpus``
directly. The gate (``classify``) fail-closes internally, so a model/transport
error surfaces as ``allowed=False`` rather than an exception here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.guardrails.classifier import classify
from src.redteam.models import AttackResult, Scorecard

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nemoguardrails import LLMRails

    from src.redteam.attacks import Attack

logger = structlog.get_logger()


async def run_attack(rails: LLMRails, attack: Attack) -> AttackResult:
    """Drive one attack through the gate and capture the raw verdict.

    Feeds the proposed tool call straight into ``classify`` — the agent never
    runs, so the result reflects the gate, not the agent's willingness to refuse.
    """
    result = await classify(
        rails,
        attack.tool_name,
        attack.tool_args,
        user_context=attack.user_context,
    )
    outcome = AttackResult(
        attack=attack,
        observed_allowed=result.allowed,
        observed_layer=result.layer,
        observed_reason=result.reason,
    )
    logger.info(
        "redteam_attack_result",
        attack_id=attack.attack_id,
        category=attack.category,
        passed=outcome.passed,
        false_allow=outcome.false_allow,
        false_block=outcome.false_block,
        observed_layer=outcome.observed_layer,
        layer_ok=outcome.layer_ok,
    )
    return outcome


async def run_corpus(rails: LLMRails, attacks: Sequence[Attack]) -> Scorecard:
    """Run the full corpus sequentially and aggregate into a ``Scorecard``.

    Sequential by design: deterministic result order for a readable matrix, and
    one model round-trip at a time keeps a live run within provider rate limits.
    """
    results = [await run_attack(rails, attack) for attack in attacks]
    return Scorecard(results=tuple(results))
