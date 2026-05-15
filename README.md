# Overseer-in-the-loop — Companion Code

Companion code for the **Overseer-in-the-loop** blog series — practically implementing Auto Mode for agent loops, one layer at a time: the agent loop, the action classifier, and the red-team results. Each post ships a runnable system; this repo's history mirrors that progression, one commit + tag at a time.

## What this is

`post-01` is the pre-implementation baseline. Post 1 is the framing piece — it argues for a layered agent permission model and previews the stack (NeMo Agent Toolkit, NeMo Guardrails, OpenCode, OpenShell) — and ships no runnable code. The repo state here is *exactly* what the later posts build from: tooling, lint/type/test config, and empty source/test trees.

## Post tags

Each blog post has a matching tagged commit in this repo's history:

```bash
git checkout post-01   # project scaffold (tooling only)
git checkout post-02   # NAT agent loop + FastAPI WS gateway + Next.js UI
git checkout post-03   # OpenCode-inspired file-system tools
git checkout post-04   # NeMo Guardrails action classifier (auto mode)
git checkout post-05   # red-teaming the gate
```

List existing tags with `git tag --list`.

## Setup

Prereqs: [`mise`](https://mise.jdx.dev/) on your `PATH`.

```bash
mise install         # resolves python + uv
uv sync              # installs the dev dependency group from uv.lock
mise run check       # lint + typecheck + test (parallel)
```

## Mise tasks

| Task                | Purpose                                  |
|---------------------|------------------------------------------|
| `mise run lint`     | `uv run ruff check src/ tests/`        |
| `mise run format`   | `uv run ruff format src/ tests/`       |
| `mise run typecheck`| `uv run mypy src/`                     |
| `mise run test`     | `uv run pytest tests/`                 |
| `mise run check`    | lint + typecheck + test (parallel)     |

More tasks (`mise run serve`, `mise run dev`, …) appear as later posts introduce app surfaces.

## Project layout

```
src/      empty at post-01; per-domain modules land as posts ship code
tests/    empty at post-01 apart from a smoke test; mirrors src/ as it grows
```

## Working in the repo

See `CLAUDE.md` for tooling, conventions, and how to run checks.
