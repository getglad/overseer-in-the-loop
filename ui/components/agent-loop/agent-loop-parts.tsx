"use client";

import { Loader2Icon } from "lucide-react";
import type { RefObject } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { BinaryOption, WebSocketMessageStatus } from "@/lib/types";

export interface Step {
  id: string;
  name: string;
  payload: string;
  status: WebSocketMessageStatus;
  timestamp: string;
}

export interface HITLPrompt {
  messageId: string;
  text: string;
  options: BinaryOption[];
}

/** One agent reasoning bubble. Same id across stream chunks → one bubble that grows. */
export interface AgentThought {
  id: string;
  text: string;
}

/** Step categories for visual grouping — tools get colored badges, others get neutral. */
const SEARCH = { label: "search", color: "bg-green-500/10 text-green-500" };
const SHELL = { label: "shell", color: "bg-red-500/10 text-red-500" };
const TOOL = { label: "tool", color: "bg-orange-500/10 text-orange-500" };
const LLM = { label: "llm", color: "bg-indigo-500/10 text-indigo-500" };
const FN = { label: "fn", color: "bg-slate-500/10 text-slate-500" };

const STEP_CATEGORIES: Record<string, { label: string; color: string }> = {
  read_file: { label: "read", color: "bg-blue-500/10 text-blue-500" },
  write_file: { label: "write", color: "bg-amber-500/10 text-amber-500" },
  edit_file: { label: "edit", color: "bg-purple-500/10 text-purple-500" },
  grep_search: SEARCH,
  glob_search: SEARCH,
  list_directory: { label: "list", color: "bg-cyan-500/10 text-cyan-500" },
  bash: SHELL,
  shell: SHELL,
  TOOL_START: TOOL,
  TOOL_END: TOOL,
  LLM_START: LLM,
  LLM_END: LLM,
  FUNCTION_START: FN,
  FUNCTION_END: FN,
};

function getStepCategory(name: string) {
  if (name in STEP_CATEGORIES) {
    return STEP_CATEGORIES[name];
  }
  for (const [key, cat] of Object.entries(STEP_CATEGORIES)) {
    if (name.includes(key)) {
      return cat;
    }
  }
  return { label: "step", color: "bg-muted text-muted-foreground" };
}

interface PayloadField {
  key: string;
  value: string;
}

/** Parse "key: value" lines from a multi-line payload. */
function parseColonFields(payload: string): PayloadField[] {
  const fields: PayloadField[] = [];
  for (const line of payload.split("\n")) {
    const idx = line.indexOf(": ");
    if (idx > 0) {
      fields.push({
        key: line.slice(0, idx).trim(),
        value: line.slice(idx + 2).trim(),
      });
    } else if (line.trim()) {
      fields.push({ key: "", value: line.trim() });
    }
  }
  return fields;
}

/** Try to parse a payload string into labeled fields. Returns null if not structured. */
function parsePayloadFields(payload: string): PayloadField[] | null {
  if (!payload.includes(": ")) return null;
  const fields = parseColonFields(payload);
  return fields.length > 0 ? fields : null;
}

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
        <div key={field.key || field.value} className="flex gap-1.5">
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

export function StepCards({ steps }: { steps: Step[] }) {
  return (
    <>
      {steps.map((step) => {
        const category = getStepCategory(step.name);
        return (
          <div
            key={step.id}
            className="rounded-md border border-border bg-muted/50 p-3"
          >
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
              <Badge
                variant={step.status === "complete" ? "secondary" : "default"}
                className="shrink-0 text-[10px]"
              >
                {step.status === "complete" ? "done" : "running"}
              </Badge>
            </div>
            {step.payload ? <StepPayload payload={step.payload} /> : null}
          </div>
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
}: {
  steps: Step[];
  endRef: RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="flex flex-col gap-3 pr-1">
      {steps.length === 0 ? (
        <p className="py-6 text-center text-xs text-muted-foreground">
          No steps yet.
        </p>
      ) : (
        <StepCards steps={steps} />
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
              <p className="mb-3 text-sm font-medium text-foreground">
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
              <span className="mb-1 block font-mono text-xs font-medium text-muted-foreground">
                Response
              </span>
              <p className="text-sm text-foreground">{finalResponse}</p>
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
}: {
  steps: Step[];
  traceEndRef: RefObject<HTMLDivElement | null>;
  onClose: () => void;
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
          <TraceStepsBlock steps={steps} endRef={traceEndRef} />
        </ScrollArea>
      </div>
    </>
  );
}
