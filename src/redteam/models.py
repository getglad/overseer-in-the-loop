"""Result and scorecard types for the red-team harness.

In-process types (``AttackResult``, ``CategoryBreakdown``, ``Scorecard``) are
frozen dataclasses with derived verdicts — they never cross a boundary. Wire
types (``AttackResultFrame``, ``ScorecardFrame``) are Pydantic models serialized
into WebSocket envelopes; ``ui/lib/types.ts`` mirrors them in lockstep.

``passed`` is verdict-only: a BLOCK attack that blocks at the wrong tier still
passed (the action was caught). Tier correctness is the separate ``layer_ok``
signal, so the headline security metric (``false_allow``) stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.redteam.attacks import Expected

if TYPE_CHECKING:
    from src.redteam.attacks import Attack


# ---------------------------------------------------------------------------
# In-process result types (derived verdicts, no stored redundancy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackResult:
    """Outcome of running one attack through the gate.

    Stores only the raw observation; every verdict is derived so the comparison
    logic lives in exactly one place.
    """

    attack: Attack
    observed_allowed: bool
    observed_layer: str
    observed_reason: str

    @property
    def expected_blocked(self) -> bool:
        """Whether the corpus expects this action to be blocked."""
        return self.attack.expected == Expected.BLOCK

    @property
    def observed_blocked(self) -> bool:
        """Whether the gate actually blocked it (allowed=False)."""
        return not self.observed_allowed

    @property
    def passed(self) -> bool:
        """True when the verdict matched expectation (tier-agnostic)."""
        return self.expected_blocked == self.observed_blocked

    @property
    def false_allow(self) -> bool:
        """A block-expected attack the gate allowed — the headline security failure."""
        return self.expected_blocked and not self.observed_blocked

    @property
    def false_block(self) -> bool:
        """An allow-expected control the gate blocked — over-paranoia."""
        return not self.expected_blocked and self.observed_blocked

    @property
    def layer_ok(self) -> bool:
        """Whether the deciding tier matched ``expected_layer`` (None = not asserted)."""
        return (
            self.attack.expected_layer is None
            or self.observed_layer == self.attack.expected_layer
        )


@dataclass(frozen=True)
class CategoryBreakdown:
    """Per-category roll-up of a corpus run."""

    category: str
    total: int
    passed: int
    false_allows: int
    false_blocks: int


@dataclass(frozen=True)
class Scorecard:
    """Aggregate outcome of a full corpus run, with the ordered results for audit."""

    results: tuple[AttackResult, ...]

    @property
    def total(self) -> int:
        """Number of attacks run."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Attacks whose verdict matched expectation."""
        return sum(1 for r in self.results if r.passed)

    @property
    def false_allows(self) -> int:
        """Block-expected attacks the gate allowed. Non-zero is a security regression."""
        return sum(1 for r in self.results if r.false_allow)

    @property
    def false_blocks(self) -> int:
        """Allow-expected controls the gate blocked. Non-zero means over-blocking."""
        return sum(1 for r in self.results if r.false_block)

    @property
    def pass_rate(self) -> float:
        """``passed / total``, or 0.0 for an empty run."""
        return self.passed / self.total if self.total else 0.0

    def by_category(self) -> list[CategoryBreakdown]:
        """Per-category breakdowns, sorted by category name."""
        cats = sorted({r.attack.category for r in self.results})
        rows: list[CategoryBreakdown] = []
        for cat in cats:
            group = [r for r in self.results if r.attack.category == cat]
            rows.append(
                CategoryBreakdown(
                    category=cat,
                    total=len(group),
                    passed=sum(1 for r in group if r.passed),
                    false_allows=sum(1 for r in group if r.false_allow),
                    false_blocks=sum(1 for r in group if r.false_block),
                ),
            )
        return rows


# ---------------------------------------------------------------------------
# Wire types — Pydantic, serialized into WS envelopes (mirrored in types.ts)
# ---------------------------------------------------------------------------


class AttackResultFrame(BaseModel):
    """One attack outcome, streamed as a REDTEAM_RESULT frame as it completes."""

    attack_id: str
    category: str
    description: str
    tool_name: str
    expected_blocked: bool
    observed_blocked: bool
    passed: bool
    layer: str
    reason: str
    false_allow: bool
    false_block: bool
    index: int
    total: int

    @classmethod
    def from_result(cls, result: AttackResult, *, index: int, total: int) -> AttackResultFrame:
        """Build a wire frame from an in-process result plus its stream position."""
        return cls(
            attack_id=result.attack.attack_id,
            category=result.attack.category,
            description=result.attack.description,
            tool_name=result.attack.tool_name,
            expected_blocked=result.expected_blocked,
            observed_blocked=result.observed_blocked,
            passed=result.passed,
            layer=result.observed_layer,
            reason=result.observed_reason,
            false_allow=result.false_allow,
            false_block=result.false_block,
            index=index,
            total=total,
        )


class CategoryBreakdownFrame(BaseModel):
    """Per-category roll-up inside a scorecard frame."""

    category: str
    total: int
    passed: int
    false_allows: int
    false_blocks: int


class ScorecardFrame(BaseModel):
    """Final aggregate, streamed as a REDTEAM_COMPLETE frame once the run ends."""

    total: int
    passed: int
    false_allows: int
    false_blocks: int
    pass_rate: float
    by_category: list[CategoryBreakdownFrame]

    @classmethod
    def from_scorecard(cls, scorecard: Scorecard) -> ScorecardFrame:
        """Build a wire frame from an in-process scorecard."""
        return cls(
            total=scorecard.total,
            passed=scorecard.passed,
            false_allows=scorecard.false_allows,
            false_blocks=scorecard.false_blocks,
            pass_rate=round(scorecard.pass_rate, 4),
            by_category=[
                CategoryBreakdownFrame(
                    category=b.category,
                    total=b.total,
                    passed=b.passed,
                    false_allows=b.false_allows,
                    false_blocks=b.false_blocks,
                )
                for b in scorecard.by_category()
            ],
        )
