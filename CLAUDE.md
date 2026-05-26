# Project conventions

Python + Next.js project building a code-first, layered permission model for agent actions. The agent itself is a NeMo Agent Toolkit (NAT) ReAct loop configured entirely in Python (no YAML); a FastAPI WebSocket gateway delivers reasoning steps to a Next.js UI and brokers HITL approvals.

## Project layout

```
src/
├── core/           shared primitives (no NAT, no FastAPI)
│   ├── protocol.py    WS message shape: MessageType, ws_msg, extract_query
│   └── render.py      Jinja render engine
├── server/         FastAPI WebSocket gateway
│   ├── app.py         FastAPI app + lifespan
│   ├── router.py      REST (/health, /status) + WS (/ws)
│   └── hitl_bridge.py WS ↔ asyncio.Future plumbing
├── loop/           NAT agent loop
│   ├── agent.py       WorkflowBuilder + HITL-wrapped tool
│   ├── service.py     transport-agnostic run_agent()
│   ├── hitl.py        prompt_binary_approval() + APPROVE/REJECT options
│   ├── prompts.py     Jinja templates + AGENT_SYSTEM_PROMPT
│   └── react_steps.py custom ReAct register — native LLM IntermediateSteps, no monkey-patch
└── tools/          File-system tool suite (LangChain wrappers + custom edit_file)
    ├── edit.py            OpenCode-inspired 9-strategy targeted string replacement
    └── tool_registry.py   FunctionGroup wiring + HITL approval middleware

ui/                 Next.js 16 + Tailwind 4 + shadcn/ui
├── app/               app-router shell
├── components/
│   ├── agent-loop/    AgentLoopPanel + step rendering
│   └── ui/            shadcn primitives
└── lib/
    ├── types.ts       TS mirror of src/core/protocol.py
    └── ws-client.ts   WebSocket client manager

tests/
├── loop/              fixtures + HITL/OTel tests
└── server/            REST endpoint tests
```

New post-3+ domain folders (`src/tools/`, `src/guardrails/`, etc.) sit alongside `loop/`, not under it.

## Architecture invariants

These hold across every post commit. Touching them requires explicit discussion.

1. **NAT is a library, never a server.** The FastAPI gateway in `src/server/app.py` embeds NAT via `WorkflowBuilder`. We never use `nat serve`. This is what lets later posts insert guardrails (post-4), policy validators (post-5), and red-team harnesses (post-7) as in-process layers around the agent.
2. **HITL gates every tool call.** Two patterns coexist: (a) per-tool wrapping inside the function body — used by `hitl_current_datetime` in `src/loop/agent.py`. (b) `FunctionGroup` middleware — `HITLApprovalMiddleware` in `src/tools/tool_registry.py` gates every tool in the group at once. NAT's ReAct agent has no pre-tool-call hook, so both patterns intercept INSIDE the tool. Middleware must be name-registered (`@register_middleware` + `add_middleware`); constructor injection gets clobbered by NAT's builder.
3. **Service layer is transport-agnostic.** `src/loop/service.py::run_agent` takes a `SendFn` callback, not a WebSocket. The router passes `websocket.send_json`; tests pass a list-accumulating mock. Every future post-domain follows this pattern.
4. **Three-layer dependency direction: `server → loop → core`.** No inverse runtime imports. `core/` has zero NAT or FastAPI coupling. Future post-3+ domains (`tools/`, `guardrails/`, `policy/`) are siblings to `loop/`, also depending only on `core/`. A type-only `TYPE_CHECKING` reference from `loop/` to `server/` (for the `WebSocketHITLBridge` annotation in `service.py`) is tolerated; nothing else.
5. **Code-first.** Python is the source of truth. YAML, when produced (e.g. OpenShell policies in post-5), is a serialization artifact emitted by code — never hand-authored.

## Tooling

### Python (`src/`, `tests/`)
- **Dependencies**: `uv`. `uv.lock` is committed; `uv sync` is deterministic across the series.
- **Lint + format**: `ruff` with `select = ["ALL"]`. Global ignores: `COM812`, `ISC001` (conflict with formatter). Tests relax `S101`, `PLR2004`, `ANN`. Per-file ignores must carry a comment explaining why.
- **Types**: `mypy` in strict mode (`strict = true`, `warn_return_any = true`, `warn_unused_configs = true`).
- **Tests**: `pytest` with `asyncio_mode = "auto"`. Async test functions need no decorator. Subdir conftests (`tests/loop/conftest.py`, `tests/server/conftest.py`) own their fixtures.
- **NAT runtime**: optional extras installed via `mise run sync` (`uv sync --extra nat`). Lint and typecheck pass without NAT installed (mypy override handles the missing imports); tests and the running server need it.

### Frontend (`ui/`)
- **Framework**: Next.js 16 (App Router) + React 19 + TypeScript 6.
- **Styling**: Tailwind 4 + shadcn/ui primitives.
- **Lint + format**: Biome (replaces ESLint + Prettier). Strict rules enabled — complexity, correctness, style, suspicious, a11y. Run via `npm run lint`.
- **Dependencies**: `npm` (driven by `mise run sync` alongside `uv sync`). `ui/package-lock.json` is tracked for deterministic installs across the series.

### Tasks
- `mise` (`.mise.toml`). One-liners stay in `.mise.toml`; longer scripts go in `.mise/tasks/` when introduced.

## Conventions

- **TDD**: failing test first, then implement. Run `mise run check` before declaring work done.
- **Structured logging**: `structlog` (Python) only — no bare `print()`. The frontend uses `console.error` / `console.log` sparingly during dev.
- **Pydantic at boundaries**: data crossing a module or process boundary is a Pydantic model on the Python side; matching TS types in `ui/lib/types.ts` on the frontend.
- **No suppression as a shortcut**: if a lint rule fires, fix the code or document the per-file ignore with a comment. Don't add a blanket `# noqa` / `biome-ignore`.

## Working in the repo

- **New backend module**: create `src/<domain>/` with `__init__.py`. Domain folders sit alongside `loop/`, not under it.
- **New frontend component**: under `ui/components/<domain>/`. Mirror the Python domain name.
- **New tests**: `tests/<domain>/test_<module>.py`. Each subdir owns its own `conftest.py` for fixtures that don't generalize.
- **Before declaring done**: `mise run check` (Python) and `(cd ui && npm run lint && npm run build)` (frontend) both green.
- Use `scratch/` (gitignored) for investigation artifacts you don't intend to commit.
