"""Live red-team battery — the real gate must catch every dangerous action.

Opt-in only: skipped unless ``REDTEAM_LIVE=1`` (and ``LLM_API_KEY``) are set, so
the default ``mise run check`` never fires real model calls even though the mise
env carries the key. The offline suites cover the deterministic rules tier and
the harness logic; this is the authoritative "does the gate hold under attack?"
check. ``mise run redteam`` runs the same corpus as a CLI scorecard.
"""

from __future__ import annotations

import os

import pytest

from src.guardrails.classifier import (
    DEFAULT_EVAL_MODEL,
    NVIDIA_NIM_BASE_URL,
    build_rails,
)
from src.redteam.attacks import CORPUS
from src.redteam.harness import run_corpus

pytestmark = pytest.mark.skipif(
    os.environ.get("REDTEAM_LIVE") != "1" or not os.environ.get("LLM_API_KEY"),
    reason="set REDTEAM_LIVE=1 (with LLM_API_KEY) to run the live battery — makes real model calls",
)


class TestLiveBattery:
    """The full corpus, driven through the real two-tier gate."""

    async def test_no_false_allows(self) -> None:
        """No block-expected attack may slip through — excluding known-soft probes.

        A false-allow on a non-soft attack is a real regression (the gate let a
        dangerous action through). Known-soft probes (heavy obfuscation) are run
        and reported but not asserted, so a documented limitation never reads as
        a regression.
        """
        rails = build_rails(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or NVIDIA_NIM_BASE_URL,
            model_name=os.environ.get("LLM_MODEL") or DEFAULT_EVAL_MODEL,
        )
        scorecard = await run_corpus(rails, CORPUS)
        offenders = [
            r.attack.attack_id
            for r in scorecard.results
            if r.false_allow and not r.attack.known_soft
        ]
        assert offenders == [], f"gate allowed dangerous actions: {offenders}"

    async def test_no_false_blocks(self) -> None:
        """Benign controls must pass — a gate that blocks them is over-paranoid."""
        rails = build_rails(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or NVIDIA_NIM_BASE_URL,
            model_name=os.environ.get("LLM_MODEL") or DEFAULT_EVAL_MODEL,
        )
        scorecard = await run_corpus(rails, CORPUS)
        over_blocked = [r.attack.attack_id for r in scorecard.results if r.false_block]
        assert over_blocked == [], f"gate over-blocked benign actions: {over_blocked}"
