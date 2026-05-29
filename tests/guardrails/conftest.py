"""Fixtures for src/guardrails/ tests — module-state isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.guardrails.middleware import set_evil_toggle

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_evil_toggle() -> Generator[None]:
    """Ensure every test starts (and leaves) with the evil toggle disabled.

    The toggle is module-level state — a test that fails between enabling
    and disabling would leak into the next test without the post-yield
    reset. Belt-and-suspenders on both ends so test ordering doesn't
    determine isolation.
    """
    set_evil_toggle(enabled=False)
    yield
    set_evil_toggle(enabled=False)
