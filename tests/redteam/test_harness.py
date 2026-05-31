"""Harness comparison + scorecard aggregation — offline, deterministic.

Outcomes are forced through rules-tier actions (decided without the model) so
the stub's verdict never matters: a block-expected ``read_file README`` is a
guaranteed false-allow, an allow-expected ``rm -rf`` a guaranteed false-block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.redteam.attacks import (
    LAYER_GUARDRAIL_AGENT,
    LAYER_RULES,
    Attack,
    Expected,
)
from src.redteam.harness import run_attack, run_corpus
from src.redteam.models import AttackResultFrame, Scorecard, ScorecardFrame

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conftest import StubRails

_BLOCKED_ACTION = ("bash", {"command": "rm -rf ."})
_ALLOWED_ACTION = ("read_file", {"file_path": "README.md"})


def _attack(
    attack_id: str,
    action: tuple[str, dict],
    expected: Expected,
    *,
    category: str = "test",
    expected_layer: str | None = None,
) -> Attack:
    """Build a minimal rules-tier Attack from a (tool_name, tool_args) action."""
    tool_name, tool_args = action
    return Attack(
        attack_id=attack_id,
        category=category,
        description="probe",
        tool_name=tool_name,
        tool_args=tool_args,
        user_context="[current] do a thing",
        expected=expected,
        expected_layer=expected_layer,
    )


class TestRunAttack:
    """run_attack compares the observed verdict to the corpus expectation."""

    async def test_correct_block_passes(self, make_rails: Callable[..., StubRails]) -> None:
        """A block-expected attack the gate blocks passes with no false flags."""
        result = await run_attack(make_rails(), _attack("blk", _BLOCKED_ACTION, Expected.BLOCK))
        assert result.passed
        assert result.observed_blocked
        assert not result.false_allow
        assert not result.false_block

    async def test_false_allow_flagged(self, make_rails: Callable[..., StubRails]) -> None:
        """A block-expected attack the gate allows is a false-allow — the security failure."""
        result = await run_attack(make_rails(), _attack("fa", _ALLOWED_ACTION, Expected.BLOCK))
        assert not result.passed
        assert result.false_allow
        assert not result.false_block

    async def test_false_block_flagged(self, make_rails: Callable[..., StubRails]) -> None:
        """An allow-expected control the gate blocks is a false-block — over-paranoia."""
        result = await run_attack(make_rails(), _attack("fb", _BLOCKED_ACTION, Expected.ALLOW))
        assert not result.passed
        assert result.false_block
        assert not result.false_allow

    async def test_layer_mismatch_still_passes(self, make_rails: Callable[..., StubRails]) -> None:
        """Verdict-only passing: a correct block at the wrong tier passes but flags layer_ok."""
        rails = make_rails()
        ok_attack = _attack("ok", _BLOCKED_ACTION, Expected.BLOCK, expected_layer=LAYER_RULES)
        ok = await run_attack(rails, ok_attack)
        assert ok.observed_layer == LAYER_RULES
        assert ok.layer_ok

        mismatch = await run_attack(
            rails,
            _attack("mm", _BLOCKED_ACTION, Expected.BLOCK, expected_layer=LAYER_GUARDRAIL_AGENT),
        )
        assert mismatch.passed
        assert not mismatch.layer_ok


class TestRunCorpus:
    """run_corpus aggregates results into a Scorecard."""

    async def test_aggregates_mixed_outcomes(self, make_rails: Callable[..., StubRails]) -> None:
        """The scorecard tallies passes, false-allows, and false-blocks by category."""
        corpus = [
            _attack("ok-block", _BLOCKED_ACTION, Expected.BLOCK, category="A1"),
            _attack("ok-allow", _ALLOWED_ACTION, Expected.ALLOW, category="benign-control"),
            _attack("fa", _ALLOWED_ACTION, Expected.BLOCK, category="A6"),
            _attack("fb", _BLOCKED_ACTION, Expected.ALLOW, category="benign-control"),
        ]
        scorecard = await run_corpus(make_rails(), corpus)
        assert scorecard.total == 4
        assert scorecard.passed == 2
        assert scorecard.false_allows == 1
        assert scorecard.false_blocks == 1
        assert scorecard.pass_rate == 0.5

        by_cat = {b.category: b for b in scorecard.by_category()}
        assert by_cat["A6"].false_allows == 1
        assert by_cat["benign-control"].total == 2
        assert by_cat["benign-control"].false_blocks == 1

    async def test_results_preserve_corpus_order(
        self,
        make_rails: Callable[..., StubRails],
    ) -> None:
        """Results stay in corpus order so the streamed matrix is deterministic."""
        corpus = [_attack(f"a{i}", _ALLOWED_ACTION, Expected.ALLOW) for i in range(5)]
        scorecard = await run_corpus(make_rails(), corpus)
        assert [r.attack.attack_id for r in scorecard.results] == ["a0", "a1", "a2", "a3", "a4"]


class TestFrames:
    """Wire frames mirror the in-process results faithfully."""

    async def test_attack_result_frame_shape(self, make_rails: Callable[..., StubRails]) -> None:
        """The result frame carries verdict, flags, and stream position."""
        attack = _attack(
            "blk", _BLOCKED_ACTION, Expected.BLOCK, category="A1", expected_layer=LAYER_RULES,
        )
        result = await run_attack(make_rails(), attack)
        frame = AttackResultFrame.from_result(result, index=3, total=10)
        assert frame.attack_id == "blk"
        assert frame.expected_blocked
        assert frame.observed_blocked
        assert frame.passed
        assert frame.layer == LAYER_RULES
        assert frame.index == 3
        assert frame.total == 10
        assert not frame.false_allow
        assert not frame.false_block

    def test_scorecard_frame_empty_run(self) -> None:
        """An empty scorecard renders a 0.0 pass rate without dividing by zero."""
        frame = ScorecardFrame.from_scorecard(Scorecard(results=()))
        assert frame.total == 0
        assert frame.passed == 0
        assert frame.pass_rate == 0.0
        assert frame.by_category == []
