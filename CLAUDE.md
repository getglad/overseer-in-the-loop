# Project conventions

Python project building a code-first, layered permission model for agent actions.

## Tooling

- **Dependencies**: `uv`. `uv.lock` is committed; `uv sync` is deterministic across the series.
- **Lint + format**: `ruff` with `select = ["ALL"]`. Global ignores are `COM812` and `ISC001` (both conflict with the formatter). Test files relax `S101`, `PLR2004`, `ANN`. Add per-file ignores only when justified, with a comment explaining why.
- **Types**: `mypy` in strict mode (`strict = true`, `warn_return_any = true`, `warn_unused_configs = true`).
- **Tests**: `pytest` with `asyncio_mode = "auto"`. Async test functions need no decorator.
- **Tasks**: `mise` (`.mise.toml`). One-liners stay in `.mise.toml`; longer scripts go in `.mise/tasks/` when introduced.

## Conventions

- **TDD**: write a failing test first, then implement. Run `mise run check` before declaring work done.
- **Structured logging**: `structlog` only — no bare `print()` for application output.
- **Pydantic at boundaries**: every data crossing a module or process boundary is a Pydantic model. Validators belong on the model.
- **Code-first**: Python is the source of truth. YAML, when produced, is a serialization artifact emitted by code — never hand-authored.
- **No suppression as a shortcut**: if a lint rule fires, fix the code or document the per-file ignore with a comment. Don't add a blanket `# noqa`.

## Working in the repo

- New module: create `src/<domain>/` with `__init__.py`; mirror under `tests/<domain>/`.
- Run `mise run check` before considering work done; it covers lint, typecheck, and tests in that order.
