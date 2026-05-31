"""Fixtures for src/redteam/ tests — an offline stub gate.

The stub stands in for a live ``LLMRails`` so the harness/scorecard/service
logic runs deterministically with no model calls. ``check_async`` is only
reached when the rules tier escalates (NEEDS_JUDGMENT); ``called`` records
whether that happened, so a rules-tier test can assert the model was never hit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class StubRails:
    """Minimal stand-in for nemoguardrails.LLMRails used by classify().

    Exposes the two attributes classify() reads: ``check_async`` (the rail call)
    and ``config.models[0].model`` (the audit model name). ``status`` controls
    the simulated LLM verdict — ``"passed"`` allows, anything else escalates
    (fail-closed), mirroring the real status mapping.
    """

    def __init__(self, status: str = "passed") -> None:
        """Create a stub whose rail returns ``status`` when consulted."""
        self._status = status
        self.called = False
        self.config = SimpleNamespace(models=[SimpleNamespace(model="stub-model")])

    async def check_async(self, **_kwargs: object) -> SimpleNamespace:
        """Record the call and return the configured status."""
        self.called = True
        return SimpleNamespace(status=SimpleNamespace(value=self._status))


@pytest.fixture
def make_rails() -> Callable[..., StubRails]:
    """Return a factory that builds a StubRails with a chosen LLM verdict."""

    def _make(status: str = "passed") -> StubRails:
        return StubRails(status)

    return _make
