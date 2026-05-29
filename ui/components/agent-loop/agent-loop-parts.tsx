"use client";

import { Loader2Icon } from "lucide-react";
import { memo, type RefObject } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

import { AgentResponseMarkdown } from "./AgentResponseMarkdown";
import {
  type AgentThought,
  elapsedMs,
  formatElapsed,
  getStepCategory,
  type HITLPrompt,
  parsePayloadFields,
  type Step,
} from "./step-utils";

/** Render payload as labeled fields when parseable, raw text otherwise. */
function StepPayload({ payload }: { payload: string }) {
  const fields = parsePayloadFields(payload);

  if (!fields) {
    return (
      <pre className="mt-1.5 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 font-mono text-[11px] text-muted-foreground">
        {payload}
      </pre>
    );
  }

  return (
    <div className="mt-1.5 flex flex-col gap-1 rounded bg-muted/50 p-2 text-[11px]">
      {fields.map((field) => (
        <div key={field.id} className="flex gap-1.5">
          {field.key ? (
            <span className="shrink-0 font-medium text-muted-foreground">
              {field.key}:
            </span>
          ) : null}
          <span className="text-foreground break-all font-mono">
            {field.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Single card in the step trace. Memoized so each card re-renders only when
 *  its own step changes — without this, a stream of 100 steps re-runs
 *  `parsePayloadFields` on all 100 cards every time one new step lands. */
const StepCard = memo(function StepCard({
  step,
  cardClassName,
  cumulativeMs,
  deltaMs,
}: {
  step: Step;
  cardClassName: string;
  cumulativeMs: number | null;
  deltaMs: number | null;
}) {
  const category = getStepCategory(step.name);
  return (
    <div className={`rounded-md border border-border ${cardClassName}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Badge
            variant="outline"
            className={`shrink-0 text-[10px] ${category.color}`}
          >
            {category.label}
          </Badge>
          <span className="truncate font-mono text-xs text-foreground">
            {step.name}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {cumulativeMs === null ? null : (
            <span
              className="font-mono text-[10px] text-muted-foreground"
              title={
                deltaMs === null
                  ? "first step in this run"
                  : `+${formatElapsed(deltaMs)} since previous step`
              }
            >
              T+{formatElapsed(cumulativeMs)}
              {deltaMs === null ? null : (
                <span className="ml-1 text-foreground/60">
                  · +{formatElapsed(deltaMs)}
                </span>
              )}
            </span>
          )}
          <Badge
            variant={step.status === "complete" ? "secondary" : "default"}
            className="shrink-0 text-[10px]"
          >
            {step.status === "complete" ? "done" : "running"}
          </Badge>
        </div>
      </div>
      {step.payload ? <StepPayload payload={step.payload} /> : null}
    </div>
  );
});

export function StepCards({
  steps,
  cardClassName = "bg-muted/50 p-3",
  queryStartedAt,
}: {
  steps: Step[];
  cardClassName?: string;
  /** ISO timestamp of when the user's query was sent. Null until first send. */
  queryStartedAt: string | null;
}) {
  return (
    <>
      {steps.map((step, idx) => {
        const cumulativeMs = queryStartedAt
          ? elapsedMs(queryStartedAt, step.timestamp)
          : null;
        const prev = idx > 0 ? steps[idx - 1] : null;
        const deltaMs = prev ? elapsedMs(prev.timestamp, step.timestamp) : null;
        return (
          <StepCard
            key={step.id}
            step={step}
            cardClassName={cardClassName}
            cumulativeMs={cumulativeMs}
            deltaMs={deltaMs}
          />
        );
      })}
    </>
  );
}

function ProcessingIndicator() {
  return (
    <div
      className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground"
      aria-live="polite"
      aria-busy="true"
    >
      <Loader2Icon
        className="size-4 shrink-0 animate-spin text-foreground"
        aria-hidden
      />
      <span>Processing…</span>
    </div>
  );
}

export function TraceStepsBlock({
  steps,
  endRef,
  queryStartedAt,
}: {
  steps: Step[];
  endRef: RefObject<HTMLDivElement | null>;
  queryStartedAt: string | null;
}) {
  return (
    <div className="flex flex-col gap-3 pr-1">
      {steps.length === 0 ? (
        <p className="py-6 text-center text-xs text-muted-foreground">
          No steps yet.
        </p>
      ) : (
        <StepCards steps={steps} queryStartedAt={queryStartedAt} />
      )}
      <div ref={endRef} />
    </div>
  );
}

export function PrimaryChatColumn({
  primaryEmpty,
  lastUserText,
  isProcessing,
  hitlPrompt,
  finalResponse,
  agentThoughts,
  primaryEndRef,
  onHITLApprove,
  onHITLReject,
}: {
  primaryEmpty: boolean;
  lastUserText: string | null;
  isProcessing: boolean;
  hitlPrompt: HITLPrompt | null;
  finalResponse: string | null;
  agentThoughts: AgentThought[];
  primaryEndRef: RefObject<HTMLDivElement | null>;
  onHITLApprove: () => void;
  onHITLReject: () => void;
}) {
  return (
    <ScrollArea className="min-h-0 flex-1 px-6 py-4">
      {primaryEmpty ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          Send a query to start the agent loop.
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          {lastUserText ? (
            <div className="rounded-md border border-border bg-muted/30 p-3">
              <span className="mb-1 block font-mono text-xs font-medium text-muted-foreground">
                You
              </span>
              <p className="text-sm text-foreground">{lastUserText}</p>
            </div>
          ) : null}

          {agentThoughts.map((thought) => (
            <div
              key={thought.id}
              className="rounded-md border border-dashed border-border bg-background p-3"
            >
              <span className="mb-1 block font-mono text-xs font-medium text-muted-foreground">
                Agent
              </span>
              <p className="whitespace-pre-wrap text-sm text-foreground">
                {thought.text}
              </p>
            </div>
          ))}

          {isProcessing ? <ProcessingIndicator /> : null}

          {hitlPrompt ? (
            <div className="rounded-md border-2 border-primary bg-primary/5 p-4">
              <p className="mb-3 whitespace-pre-wrap text-sm font-medium text-foreground">
                {hitlPrompt.text}
              </p>
              <div className="flex gap-2">
                <Button size="sm" onClick={onHITLApprove}>
                  Approve
                </Button>
                <Button size="sm" variant="destructive" onClick={onHITLReject}>
                  Reject
                </Button>
              </div>
            </div>
          ) : null}

          {finalResponse ? (
            <div className="rounded-md border border-border bg-card p-4">
              <span className="mb-2 block font-mono text-xs font-medium text-muted-foreground">
                Response
              </span>
              <AgentResponseMarkdown content={finalResponse} />
            </div>
          ) : null}

          <div ref={primaryEndRef} />
        </div>
      )}
    </ScrollArea>
  );
}

export function MobileTraceOverlay({
  steps,
  traceEndRef,
  onClose,
  queryStartedAt,
}: {
  steps: Step[];
  traceEndRef: RefObject<HTMLDivElement | null>;
  onClose: () => void;
  queryStartedAt: string | null;
}) {
  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-50 bg-background/80 md:hidden"
        aria-label="Close trace panel"
        onClick={onClose}
      />
      <div
        className="fixed inset-y-0 right-0 z-50 flex w-[min(85vw,20rem)] flex-col border-l border-border bg-card shadow-lg md:hidden"
        role="dialog"
        aria-modal="true"
        aria-labelledby="trace-panel-title"
      >
        <div className="flex items-center justify-between gap-2 border-b border-border p-3">
          <div className="min-w-0">
            <h2
              id="trace-panel-title"
              className="text-sm font-medium text-foreground"
            >
              Agent trace
            </h2>
            <p className="text-xs text-muted-foreground">
              Reasoning and tool steps
            </p>
          </div>
          <Button type="button" variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <ScrollArea className="min-h-0 flex-1 p-3">
          <TraceStepsBlock
            steps={steps}
            endRef={traceEndRef}
            queryStartedAt={queryStartedAt}
          />
        </ScrollArea>
      </div>
    </>
  );
}
