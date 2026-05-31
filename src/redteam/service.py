"""Stream a red-team run over a transport-agnostic send callback.

Mirrors ``src/loop/service.py::run_agent`` — takes an async ``send`` callback,
not a WebSocket. The router passes ``websocket.send_json``; tests pass a
list-accumulating mock. Results stream in corpus order (the run is sequential),
then a final scorecard frame.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog

from src.core.protocol import MessageType, ws_msg
from src.redteam.harness import run_attack
from src.redteam.models import AttackResult, AttackResultFrame, Scorecard, ScorecardFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nemoguardrails import LLMRails

    from src.redteam.attacks import Attack

logger = structlog.get_logger()

# Re-declared locally (not imported from loop/) so redteam depends only on
# guardrails + core, per the server -> loop -> core dependency-direction invariant.
type SendFn = Callable[[dict[str, Any]], Awaitable[None]]

# Layer label for an attack that errored inside the harness — not a gate verdict.
_ERROR_LAYER = "error"


async def run_redteam(send: SendFn, rails: LLMRails, attacks: Sequence[Attack]) -> None:
    """Run the corpus, streaming one result frame per attack then a scorecard.

    Each attack is isolated: a harness error becomes a fail-closed (blocked)
    result and the run continues, so the final scorecard always ships even if
    an individual probe raises.
    """
    total = len(attacks)
    results: list[AttackResult] = []
    for index, attack in enumerate(attacks):
        try:
            outcome = await run_attack(rails, attack)
        except Exception:
            logger.exception("redteam_attack_error", attack_id=attack.attack_id)
            # Fail closed: an errored probe counts as blocked, never a false-allow.
            outcome = AttackResult(
                attack=attack,
                observed_allowed=False,
                observed_layer=_ERROR_LAYER,
                observed_reason="harness error — see server logs",
            )
        results.append(outcome)
        await send(
            ws_msg(
                MessageType.REDTEAM_RESULT,
                AttackResultFrame.from_result(outcome, index=index, total=total).model_dump(),
            ),
        )

    scorecard = Scorecard(results=tuple(results))
    await send(
        ws_msg(
            MessageType.REDTEAM_COMPLETE,
            ScorecardFrame.from_scorecard(scorecard).model_dump(),
            status="complete",
        ),
    )
    logger.info(
        "redteam_run_complete",
        total=scorecard.total,
        passed=scorecard.passed,
        false_allows=scorecard.false_allows,
        false_blocks=scorecard.false_blocks,
    )
