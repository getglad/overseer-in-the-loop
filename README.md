# Overseer-in-the-loop — Companion Code

Companion code for the **Overseer-in-the-loop** blog series — practically implementing Auto Mode for agent loops, one layer at a time: the agent loop, the action classifier, and the red-team results. Each post ships a runnable system; this repo's history mirrors that progression, one commit + tag at a time.

> **Local demo only.** This runs on your machine for the posts — it's loopback-bound with no auth, and is not a deployment target.

## What this is

`git checkout post-N` lands on any post's state. At `post-05`, the runnable system is a code-first agent that acts at arm's length:

- a **NeMo Agent Toolkit (NAT)** ReAct loop, embedded as a library behind a **FastAPI** WebSocket gateway — never `nat serve`
- a **Next.js** UI that streams the agent's reasoning and tool calls in real time
- **auto mode**: a two-tier action classifier (deterministic rules + an LLM judge) gates every tool call — safe actions run automatically, dangerous ones are blocked, and only the ambiguous fall back to human approval
- a **red-team battery** that drives an attack corpus straight at the classifier and scores whether the gate holds

The agent is configured entirely in Python — no YAML — and OpenTelemetry tracing is wired (optional) so reasoning steps surface in any OTLP-compatible backend.

## Post tags

```bash
git checkout post-01   # project scaffold (tooling only)
git checkout post-02   # NAT agent loop + FastAPI WS gateway + Next.js UI
git checkout post-03   # OpenCode-inspired file-system tools
git checkout post-04   # NeMo Guardrails action classifier (auto mode)
git checkout post-05   # red-teaming the gate
```

List existing tags with `git tag --list`.

## Setup

Prereqs: [`mise`](https://mise.jdx.dev/) on your `PATH`. The `mise install` step pulls Python 3.14, Node 26, uv, and hivemind.

```bash
mise install                                                # python + node + uv + hivemind
mise run sync                                               # uv sync (+ NAT extras) + npm install in ui/
cp mise.local.toml.example mise.local.toml && $EDITOR $_    # set your inference key — see "Inference endpoint" below
mise run check                                              # lint + typecheck + tests, all parallel
```

`mise.local.toml` is gitignored. Its `[env]` block is auto-loaded into every `mise run ...` task — no `.env` file, no `source`, no `op run --` wrapper needed for local dev.

## Inference endpoint

The agent talks to any OpenAI-compatible inference endpoint. Set these in `mise.local.toml` (`[env]`) — they're auto-loaded into every `mise run …` task:

| Variable | What it sets | Default |
|---|---|---|
| `LLM_API_KEY`  | API key — **required**     | —                                                  |
| `LLM_BASE_URL` | the inference **endpoint** | NVIDIA NIM (`https://integrate.api.nvidia.com/v1`)  |
| `LLM_MODEL`    | the model name             | `z-ai/glm-5.1`                                      |

Out of the box it points at **NVIDIA NIM** — grab a free key at [build.nvidia.com](https://build.nvidia.com), set `LLM_API_KEY`, and you're done. To use another provider (Azure OpenAI, a local vLLM, OpenAI, …), also set `LLM_BASE_URL` and `LLM_MODEL`; `mise.local.toml.example` lists copy-paste endpoints for each.

## Run the agent

```bash
mise run dev                              # hivemind starts FastAPI + Next.js together
```

One command brings up the whole stack and streams both processes' logs in your terminal. The first run builds the UI (~1 min); after that the build is cached, so it starts immediately. Then open `http://localhost:3000` and send a query like *"what time is it?"* — the UI streams reasoning steps in real time, and the classifier auto-approves safe tool calls, prompting you only when it blocks something.

| Surface | Port | Purpose |
|---|---|---|
| FastAPI gateway | 8000 | `/health`, `/status`, `/ws` (WebSocket agent loop) |
| Next.js UI | 3000 | Agent loop + auto-mode approvals at `/`, red-team matrix at `/redteam` |

## Run the red-team battery

```bash
mise run redteam                          # drives the attack corpus through the live classifier
```

Prints a scorecard and exits non-zero on any **false-allow** (a block-expected attack the gate let through), so it doubles as a regression gate. Makes real model calls, so it's kept out of `mise run check` (which stays offline and deterministic). The same streamed attack → verdict matrix and scorecard are also available in the UI at `/redteam`.

## Mise tasks

| Task                | Purpose                                  |
|---------------------|------------------------------------------|
| `mise run lint`     | `uv run ruff check src/ tests/`        |
| `mise run format`   | `uv run ruff format src/ tests/`       |
| `mise run typecheck`| `uv run mypy src/`                     |
| `mise run test`     | `uv run pytest tests/`                 |
| `mise run check`    | lint + typecheck + test (parallel)     |
| `mise run sync`     | `uv sync --extra nat` + `npm install` in `ui/` |
| `mise run serve`    | uvicorn FastAPI server on :8000        |
| `mise run build:ui` | `next build` — production UI bundle    |
| `mise run serve:ui` | `next start` — serve the built UI on :3000 |
| `mise run dev`      | hivemind → server + ui together        |
| `mise run redteam`  | live red-team battery → scorecard (real model calls) |

## Working in the repo

See `CLAUDE.md` for project layout, architecture invariants (NAT-as-library, the action classifier, tool-level HITL, transport-agnostic service layer), and tooling conventions.
