# UI conventions

See root `CLAUDE.md` for project-wide conventions, language choices, and architecture invariants. This file covers UI-specific patterns the Next.js dashboard needs to stay coherent.

## Layout

```
ui/
├── app/                  Next.js app-router shell (layout.tsx, page.tsx, globals.css)
├── components/
│   ├── ui/               shadcn primitives — DO NOT hand-edit; managed by `npx shadcn add`
│   └── <domain>/         composed components per backend domain (agent-loop today; guardrails, planner, etc. later)
└── lib/
    ├── types.ts          TS mirror of src/core/protocol.py — lockstep with backend MessageType
    ├── ws-client.ts      WSClient class — connection lifecycle, auto-reconnect with backoff
    └── utils.ts          shadcn cn() helper
```

`public/` doesn't exist yet — add it when you have static assets to serve.

New dashboard panels go under `components/<domain>/`, matching the backend domain folder under `src/`.

## Path aliases

`@/components`, `@/lib`, `@/components/ui`, `@/hooks` — defined in `tsconfig.json` (`paths`) and `components.json` (shadcn). Always use the alias, never relative `../../` paths.

## Client/server boundary

Default in App Router is **server component**. Any component that uses hooks, state, refs, or browser APIs (e.g. `WebSocket`) needs `"use client"` at the top — see `AgentLoopPanel.tsx`. Keep the boundary as low in the tree as possible; the parent `page.tsx` stays a server component and just mounts the client panel.

## WebSocket protocol contract

`lib/types.ts` mirrors `src/core/protocol.py::MessageType` exactly. Adding a message type requires updating BOTH files in the same commit. The TS discriminator is the literal string ("user_message", "system_intermediate_message", etc.) — keep it identical to the Python enum value.

## WSClient pattern

`AgentLoopPanel` owns a single `WSClient` instance via `useRef<WSClient | null>` and mounts it in `useEffect` with cleanup. Pattern for any future panel that needs the WS:

```tsx
const clientRef = useRef<WSClient | null>(null);

useEffect(() => {
  const client = new WSClient({ url: WS_URL, onMessage: ... });
  clientRef.current = client;
  client.connect();
  return () => client.disconnect();
}, []);
```

`WSClient` handles reconnection with exponential backoff (up to 30s, with jitter) — don't reimplement.

## ENV vars (Next.js)

- `NEXT_PUBLIC_*` vars are baked into the client bundle at build/dev-start. Restart `mise run dev` after changing them in `mise.local.toml`.
- Non-`NEXT_PUBLIC_*` vars stay server-only — invisible to the browser. There's no server runtime at post-02 (page.tsx is fully static), so this rarely matters yet.
- `NEXT_PUBLIC_WS_URL` is the only public var today; defaults to `ws://localhost:8000/ws` if unset (see `AgentLoopPanel.tsx`).

## shadcn primitives

- Add primitives with `npx shadcn add <component>` — never hand-write.
- Config lives in `components.json` (`style: base-nova`, `baseColor: neutral`, `iconLibrary: lucide`).
- The underlying primitive library is `@base-ui/react`; don't add headless-ui or radix-ui alongside it.
- If a primitive needs project-specific styling, wrap it in a domain component under `components/<domain>/` rather than editing the shadcn file.

## Tailwind 4

- **No `tailwind.config.ts`** — Tailwind 4 reads everything from PostCSS + CSS variables in `app/globals.css`.
- Theming via CSS variables (`--background`, `--foreground`, etc.) in `globals.css`. Use the existing tokens; don't introduce new ones without coordinating.
- Animation utilities come from `tw-animate-css` — import the class names directly.

## Bounded buffers

Long-running agent traces can produce hundreds of intermediate steps. `AgentLoopPanel` caps stored steps at `MAX_STEPS = 500` and drops the oldest. Any future panel that subscribes to a long-running stream follows the same pattern — don't let React state grow unbounded.

## Lint + format

Biome handles both via `npm run lint` (or `mise run lint:ui` if a task is added later). The ruleset in `biome.json` enables strict checks including `noExcessiveCognitiveComplexity`, `noNestedTernary`, `noForEach`, and a11y rules. Per-file overrides need a `biome-ignore` comment with a reason.

## Before declaring done

- `npm run lint` (biome — fast, ~150ms)
- `npm run build` (next build — also runs `tsc`, ~3s)

Both green = safe to commit.
