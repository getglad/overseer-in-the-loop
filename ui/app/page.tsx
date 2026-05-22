import { AgentLoopPanel } from "@/components/agent-loop/AgentLoopPanel";

export default function Home() {
  return (
    <main className="flex flex-1 flex-col gap-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          Agent Auto Mode
        </h1>
        <p className="text-sm text-muted-foreground">
          Code-first agent dashboard — HITL approval, observability, policy
          control
        </p>
      </header>

      <AgentLoopPanel />
    </main>
  );
}
