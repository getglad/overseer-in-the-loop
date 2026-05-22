"""Per-conversation context shared with the guardrail classifier.

The Layer-2 classifier (``src/guardrails/classifier.py``) judges each tool call
against the user's GOAL, which often spans several turns ("read the config",
then "now deploy it"). To judge proportionality it needs the recent user turns
that led to the current action — but ONLY the human's side. Agent reasoning and
tool output are deliberately excluded: they can carry attacker-controlled
content (a poisoned file the agent read), and feeding that to the judge is the
classic indirect-prompt-injection vector.

History is carried as a task-local ``ContextVar`` — the same mechanism NAT uses
for ``input_message`` — set per run by the service layer from the conversation's
rolling window. Because each agent run executes in its own asyncio task, two
concurrent conversations never see each other's history (unlike module-global
state, which would silently leak one user's prompts into another's guard
decisions). This module stays NAT-free so it can live in ``core/``.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Most recent USER prompts (excluding the current one) the classifier may see.
MAX_PROMPT_HISTORY = 5
# Per-prompt character cap — a pasted blob shouldn't blow the classifier's
# context window or token budget, and only the gist is needed for proportionality.
MAX_PROMPT_CHARS = 500

_recent_user_prompts: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "recent_user_prompts",
    default=(),
)


def set_recent_user_prompts(prompts: Sequence[str]) -> contextvars.Token[tuple[str, ...]]:
    """Bind the recent user-prompt window for the current run.

    Drops empties, caps each prompt at ``MAX_PROMPT_CHARS`` and keeps only the
    last ``MAX_PROMPT_HISTORY`` (oldest first). Returns a token the caller
    passes to ``reset_recent_user_prompts`` when the run ends.
    """
    trimmed = tuple(p[:MAX_PROMPT_CHARS] for p in prompts if p)[-MAX_PROMPT_HISTORY:]
    return _recent_user_prompts.set(trimmed)


def reset_recent_user_prompts(token: contextvars.Token[tuple[str, ...]]) -> None:
    """Restore the prior window after a run (pairs with ``set_recent_user_prompts``)."""
    _recent_user_prompts.reset(token)


def get_recent_user_prompts() -> tuple[str, ...]:
    """The recent USER prompts (oldest first) bound for the current run, or ``()``."""
    return _recent_user_prompts.get()
