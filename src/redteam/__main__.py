"""CLI entry: run the live red-team battery and print the scorecard.

Builds the real action gate from env (``LLM_API_KEY``, optional ``LLM_BASE_URL``
/ ``LLM_MODEL``) and drives the full corpus through it — this makes real model
calls. Exits non-zero when any block-expected attack was allowed (a false-allow),
so it doubles as a scheduled regression gate. NOT part of ``mise run check``
(that path is offline and deterministic).

Run: ``mise run redteam``  (or ``uv run python -m src.redteam``)
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

import structlog

from src.guardrails.classifier import (
    DEFAULT_EVAL_MODEL,
    NVIDIA_NIM_BASE_URL,
    build_rails,
)
from src.redteam.attacks import CORPUS
from src.redteam.harness import run_corpus

if TYPE_CHECKING:
    from src.redteam.models import AttackResult, Scorecard

logger = structlog.get_logger()

_EXIT_FALSE_ALLOW = 1
_EXIT_NO_KEY = 2


def _result_line(result: AttackResult) -> str:
    """Format one attack result as a scorecard row."""
    mark = "PASS" if result.passed else "FAIL"
    if result.false_allow:
        flag = "  <-- FALSE-ALLOW (dangerous action permitted)"
    elif result.false_block:
        flag = "  <-- false-block (over-paranoid)"
    else:
        flag = ""
    return (
        f"  [{mark}] {result.attack.attack_id:<38} "
        f"{result.attack.category:<16} {result.observed_layer:<15}{flag}"
    )


def _render(scorecard: Scorecard) -> str:
    """Render the scorecard as a plain-text report for stdout."""
    lines = ["", "=== RED-TEAM SCORECARD ==="]
    lines.extend(_result_line(r) for r in scorecard.results)
    lines.append("")
    lines.extend(
        f"  {b.category:<16} {b.passed}/{b.total} pass"
        f"   false_allows={b.false_allows} false_blocks={b.false_blocks}"
        for b in scorecard.by_category()
    )
    lines.append("")
    lines.append(
        f"  TOTAL {scorecard.passed}/{scorecard.total} passed ({scorecard.pass_rate:.0%})"
        f"   FALSE-ALLOWS={scorecard.false_allows}   false_blocks={scorecard.false_blocks}",
    )
    lines.append("")
    return "\n".join(lines)


async def _main() -> int:
    """Build live rails, run the corpus, print the scorecard, return an exit code."""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        logger.error("redteam_no_api_key", reason="LLM_API_KEY is required for the live battery")
        return _EXIT_NO_KEY
    base_url = os.environ.get("LLM_BASE_URL") or NVIDIA_NIM_BASE_URL
    model = os.environ.get("LLM_MODEL") or DEFAULT_EVAL_MODEL
    rails = build_rails(api_key=api_key, base_url=base_url, model_name=model)

    scorecard = await run_corpus(rails, CORPUS)
    sys.stdout.write(_render(scorecard))
    # Non-zero exit if any dangerous action slipped through — the regression gate.
    return _EXIT_FALSE_ALLOW if scorecard.false_allows else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
