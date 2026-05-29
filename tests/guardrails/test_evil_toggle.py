"""Tests for the evil-toggle module state.

The toggle is a module-level boolean — acceptable for a single-user blog
demo but a known limitation if two sessions ran concurrently. These tests
pin the setter/getter contract and the default-disabled invariant.

Isolation between tests is handled by the autouse fixture in conftest.py.
"""

from __future__ import annotations

from src.guardrails.middleware import is_evil_toggle_enabled, set_evil_toggle


def test_toggle_defaults_to_disabled() -> None:
    """A fresh process must have the evil toggle off."""
    assert is_evil_toggle_enabled() is False


def test_set_evil_toggle_enables_the_payload_swap() -> None:
    """``set_evil_toggle(enabled=True)`` flips the flag the middleware reads."""
    set_evil_toggle(enabled=True)
    assert is_evil_toggle_enabled() is True


def test_set_evil_toggle_can_disable_again() -> None:
    """The toggle round-trips — disable returns to the default state."""
    set_evil_toggle(enabled=True)
    set_evil_toggle(enabled=False)
    assert is_evil_toggle_enabled() is False
