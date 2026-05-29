"use client";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

import { StepCards } from "../agent-loop/agent-loop-parts";
import type { Step } from "../agent-loop/step-utils";

interface ToolPanelProps {
  steps: Step[];
  /** ISO timestamp when the current query was sent; null before first send. */
  queryStartedAt: string | null;
}

export function ToolPanel({ steps, queryStartedAt }: ToolPanelProps) {
  if (steps.length === 0) {
    return (
      <div className="flex flex-col gap-2 p-3">
        <h2 className="text-sm font-medium text-foreground">Agent trace</h2>
        <p className="py-4 text-center text-xs text-muted-foreground">
          Send a query to start the agent loop.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="px-3 pt-3">
        <h2 className="text-sm font-medium text-foreground">
          Agent trace
          <Badge variant="secondary" className="ml-2 font-mono text-xs">
            {steps.length}
          </Badge>
        </h2>
      </div>
      <ScrollArea className="flex-1 min-h-0 px-3 pb-3">
        <div className="flex flex-col gap-2">
          <StepCards
            steps={steps}
            cardClassName="bg-card p-2.5"
            queryStartedAt={queryStartedAt}
          />
        </div>
      </ScrollArea>
    </div>
  );
}
