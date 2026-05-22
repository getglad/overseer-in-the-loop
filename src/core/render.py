"""Jinja prompt rendering engine.

Compiles + renders prompt template strings to plain strings. The templates
themselves live with their domain (e.g. `src/loop/prompts.py`); this module
is just the engine they all funnel through.

`StrictUndefined` ensures missing variables raise at render time instead of
silently rendering empty. `trim_blocks` and `lstrip_blocks` keep multi-line
templates clean without manual whitespace management.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from jinja2 import Environment, StrictUndefined, Template

# S701: autoescape=False is intentional — we render plain text prompts, not HTML
_env = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701
)


@lru_cache(maxsize=64)
def _compile(source: str) -> Template:
    """Parse and cache a Jinja template string."""
    return _env.from_string(source)


def render(template: str, **kwargs: Any) -> str:  # noqa: ANN401 — Jinja renders any type; typed wrappers provide safety
    """Render an inline Jinja template with strict variable checking.

    Templates are compiled once and cached by source string.

    Args:
        template: Jinja template string (e.g., "Tool: {{ tool_name }}").
        **kwargs: Template variables. Missing variables raise `UndefinedError`.

    Returns:
        Rendered string.
    """
    return _compile(template).render(**kwargs)
