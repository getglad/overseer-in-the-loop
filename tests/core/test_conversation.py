"""Tests for the per-conversation prompt-history ContextVar (core.conversation).

The window is what the Layer-2 classifier reads to judge an action against the
conversation's goal. The load-bearing property is task isolation: two concurrent
conversations must never see each other's prompts (the reason this is a
ContextVar, not module-global state).
"""

from __future__ import annotations

import asyncio

from src.core.conversation import (
    MAX_PROMPT_CHARS,
    MAX_PROMPT_HISTORY,
    get_recent_user_prompts,
    reset_recent_user_prompts,
    set_recent_user_prompts,
)


class TestRecentUserPrompts:
    """The window binds task-locally, stays bounded, and restores on reset."""

    def test_default_is_empty(self) -> None:
        """With nothing bound, the classifier sees no prior turns."""
        assert get_recent_user_prompts() == ()

    def test_set_get_reset_roundtrip(self) -> None:
        """A bound window is visible, then restored to empty on reset."""
        token = set_recent_user_prompts(["read config", "deploy it"])
        assert get_recent_user_prompts() == ("read config", "deploy it")
        reset_recent_user_prompts(token)
        assert get_recent_user_prompts() == ()

    def test_keeps_only_last_n(self) -> None:
        """Only the most recent MAX_PROMPT_HISTORY turns survive (oldest dropped)."""
        prompts = [f"p{i}" for i in range(MAX_PROMPT_HISTORY + 3)]
        token = set_recent_user_prompts(prompts)
        kept = get_recent_user_prompts()
        reset_recent_user_prompts(token)
        assert kept == tuple(prompts[-MAX_PROMPT_HISTORY:])

    def test_caps_prompt_length(self) -> None:
        """An oversized prompt is truncated so it can't blow the classifier's budget."""
        token = set_recent_user_prompts(["x" * (MAX_PROMPT_CHARS + 50)])
        kept = get_recent_user_prompts()
        reset_recent_user_prompts(token)
        assert len(kept[0]) == MAX_PROMPT_CHARS

    def test_drops_empty_prompts(self) -> None:
        """Empty strings are dropped — they carry no goal context."""
        token = set_recent_user_prompts(["a", "", "b"])
        kept = get_recent_user_prompts()
        reset_recent_user_prompts(token)
        assert kept == ("a", "b")

    async def test_isolated_across_concurrent_tasks(self) -> None:
        """Concurrent conversations each see only their own window — no cross-bleed.

        This is the property module-global state could not provide: each agent
        run executes in its own asyncio task, which gets a private copy of the
        context, so one user's prompts never leak into another's guard decision.
        """
        seen: dict[str, tuple[str, ...]] = {}

        async def worker(name: str, prompts: list[str]) -> None:
            token = set_recent_user_prompts(prompts)
            await asyncio.sleep(0)  # yield so the two workers interleave
            seen[name] = get_recent_user_prompts()
            reset_recent_user_prompts(token)

        await asyncio.gather(worker("a", ["alpha"]), worker("b", ["beta"]))
        assert seen == {"a": ("alpha",), "b": ("beta",)}
