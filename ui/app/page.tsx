import Link from "next/link";

import { AgentLoopPanel } from "@/components/agent-loop/AgentLoopPanel";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col gap-6 p-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            Overseer-in-the-loop
          </h1>
          <p className="text-sm text-muted-foreground">
            Code-first agent dashboard — HITL approval, observability, policy
            control
          </p>
        </div>
        <Link
          href="/redteam"
          className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          Red-team →
        </Link>
      </header>

      <AgentLoopPanel />
    </main>
  );
}
