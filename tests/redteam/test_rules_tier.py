"""The deterministic gate: rules-tier attacks resolve without a model call.

This is the offline heart of the red-team — every attack the corpus labels
``rules`` is driven through the real ``classify()`` and must produce its expected
verdict from the deterministic tier alone, never consulting the (stubbed) model.
A rules regression flips a verdict or escalates, and the assertions catch both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.guardrails.classifier import classify
from src.redteam.attacks import CORPUS, LAYER_RULES, Expected

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.redteam.attacks import Attack

    from .conftest import StubRails

_RULES_TIER = [a for a in CORPUS if a.expected_layer == LAYER_RULES]


class TestRulesTierDeterministic:
    """Each rules-tier attack gets its expected verdict from rules alone."""

    @pytest.mark.parametrize("attack", _RULES_TIER, ids=lambda a: a.attack_id)
    async def test_rules_verdict_matches_and_skips_model(
        self,
        attack: Attack,
        make_rails: Callable[..., StubRails],
    ) -> None:
        """The rules tier decides the verdict and never reaches the model."""
        # status="passed" would ALLOW if the model were consulted — so a BLOCK
        # verdict here can only have come from the deterministic rules tier.
        rails = make_rails(status="passed")
        result = await classify(
            rails,
            attack.tool_name,
            attack.tool_args,
            user_context=attack.user_context,
        )
        expected_blocked = attack.expected == Expected.BLOCK
        assert (not result.allowed) == expected_blocked
        assert result.layer == LAYER_RULES
        assert rails.called is False

    def test_rules_tier_is_substantial(self) -> None:
        """A meaningful share of the battery is deterministically gated."""
        assert len(_RULES_TIER) >= 15
