"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { ServerMessage } from "@/lib/types";
import { WSClient } from "@/lib/ws-client";

import { ToolPanel } from "../tool-panel/ToolPanel";
import { MobileTraceOverlay, PrimaryChatColumn } from "./agent-loop-parts";
import type { AgentThought, HITLPrompt, Step } from "./step-utils";

type ConnectionStatus = "disconnected" | "connecting" | "connected";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const MAX_STEPS = 500;
const MAX_THOUGHTS = 200;

/** Bounded-buffer rule: cap step history so React state doesn't grow unboundedly. */
function capSteps(steps: Step[]): Step[] {
  return steps.length > MAX_STEPS ? steps.slice(-MAX_STEPS) : steps;
}

/** Same bounded-buffer rule for the agent's reasoning bubbles. */
function capThoughts(thoughts: AgentThought[]): AgentThought[] {
  return thoughts.length > MAX_THOUGHTS
    ? thoughts.slice(-MAX_THOUGHTS)
    : thoughts;
}

export function AgentLoopPanel() {
  const [steps, setSteps] = useState<Step[]>([]);
  const [finalResponse, setFinalResponse] = useState<string | null>(null);
  const [hitlPrompt, setHitlPrompt] = useState<HITLPrompt | null>(null);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("disconnected");
  const [query, setQuery] = useState("");
  const [lastUserText, setLastUserText] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentThoughts, setAgentThoughts] = useState<AgentThought[]>([]);

  const clientRef = useRef<WSClient | null>(null);
  const primaryEndRef = useRef<HTMLDivElement>(null);
  const traceEndMobileRef = useRef<HTMLDivElement>(null);

  // Depend on `.length` not the array — every stream chunk produces a new
  // array reference, which would otherwise fire scrollIntoView per token.
  const thoughtCount = agentThoughts.length;
  useEffect(() => {
    if (
      lastUserText !== null ||
      hitlPrompt !== null ||
      finalResponse !== null ||
      isProcessing ||
      thoughtCount > 0
    ) {
      primaryEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [lastUserText, hitlPrompt, finalResponse, isProcessing, thoughtCount]);

  useEffect(() => {
    // Skip when the mobile trace overlay isn't visible — otherwise every
    // step update during a long agent run kicks off a smooth-scroll
    // animation against an offscreen element, which backlogs on mobile.
    if (!traceOpen || steps.length === 0) return;
    traceEndMobileRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps, traceOpen]);

  const handleMessage = useCallback((message: ServerMessage) => {
    switch (message.type) {
      case "system_intermediate_message": {
        // Route agent reasoning to the chat column, not the trace panel —
        // the trace stays focused on tool calls and downstream domain
        // events. Merge by message id: every stream chunk for one LLM call
        // shares the same id, so partial updates accumulate into ONE
        // growing bubble instead of N bubbles per LLM call.
        if (message.content.name === "agent:thinking") {
          // Payload is a DELTA (new text since the last chunk) — append it to
          // the bubble with this id, or start a new bubble. Pure updater, so
          // React strict-mode double-invocation stays correct (both start from
          // the same prev; last result wins).
          setAgentThoughts((prev) => {
            const idx = prev.findIndex((t) => t.id === message.id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = {
                id: message.id,
                text: prev[idx].text + message.content.payload,
              };
              return updated;
            }
            return capThoughts([
              ...prev,
              { id: message.id, text: message.content.payload },
            ]);
          });
          break;
        }
        setSteps((prev) => {
          const existing = prev.findIndex((s) => s.id === message.id);
          const step: Step = {
            id: message.id,
            name: message.content.name,
            payload: message.content.payload,
            status: message.status,
            timestamp: message.timestamp,
          };
          if (existing >= 0) {
            const old = prev[existing];
            if (
              old.name === step.name &&
              old.payload === step.payload &&
              old.status === step.status
            ) {
              return prev;
            }
            const updated = [...prev];
            updated[existing] = step;
            return updated;
          }
          const next = [...prev, step];
          return capSteps(next);
        });
        break;
      }

      case "system_interaction_message": {
        // Surface BOTH binary and text prompts. The backend resolves either via
        // the same yes/no bridge, so a text prompt rendered with approve/reject
        // still unblocks the run — silently dropping non-binary prompts would
        // hang the agent forever waiting on a reply the user can't give.
        setIsProcessing(false);
        setHitlPrompt({
          messageId: message.id,
          text: message.content.text,
          options:
            message.content.input_type === "binary_choice"
              ? message.content.options
              : [],
        });
        break;
      }

      case "system_response_message": {
        if (message.status === "complete") {
          setIsProcessing(false);
          setFinalResponse(message.content);
        }
        break;
      }

      case "error_message":
        setIsProcessing(false);
        // The server's error path cancels pending HITL futures, so clear any
        // stale prompt — otherwise the approve/reject card lingers with nothing
        // listening behind it.
        setHitlPrompt(null);
        setSteps((prev) => {
          const next = [
            ...prev,
            {
              // crypto.randomUUID() not Date.now() — two errors in the same ms
              // would collide as React keys and break reconciliation.
              id: `error-${crypto.randomUUID()}`,
              name: "Error",
              payload: message.content,
              status: "complete" as const,
              timestamp: message.timestamp,
            },
          ];
          return capSteps(next);
        });
        break;
    }
  }, []);

  useEffect(() => {
    const client = new WSClient({
      url: WS_URL,
      onMessage: handleMessage,
      onOpen: () => setConnectionStatus("connected"),
      onClose: () => {
        setIsProcessing(false);
        setConnectionStatus("disconnected");
      },
      onError: () => {
        setIsProcessing(false);
        setConnectionStatus("disconnected");
      },
    });

    clientRef.current = client;
    setConnectionStatus("connecting");
    client.connect();

    return () => {
      client.disconnect();
    };
  }, [handleMessage]);

  const sendQuery = () => {
    if (!query.trim() || !clientRef.current?.isConnected) return;

    const text = query.trim();
    setSteps([]);
    setFinalResponse(null);
    setHitlPrompt(null);
    setAgentThoughts([]);
    setLastUserText(text);

    clientRef.current.send({
      type: "user_message",
      id: crypto.randomUUID(),
      content: {
        messages: [{ role: "user", content: [{ type: "text", text }] }],
      },
    });
    setQuery("");
    setIsProcessing(true);
  };

  const handleHITLResponse = (approved: boolean) => {
    if (!hitlPrompt || !clientRef.current?.isConnected) return;

    clientRef.current.send({
      type: "user_interaction_message",
      id: crypto.randomUUID(),
      thread_id: hitlPrompt.messageId,
      parent_id: hitlPrompt.messageId,
      content: {
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: approved ? "yes" : "no" }],
          },
        ],
      },
    });
    setHitlPrompt(null);
    setIsProcessing(true);
  };

  const primaryEmpty =
    lastUserText === null && hitlPrompt === null && finalResponse === null;

  return (
    <Card className="flex flex-1 flex-col">
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <div className="min-w-0">
          <CardTitle className="text-lg">Agent Loop</CardTitle>
          <CardDescription>
            Chat and HITL here; intermediate steps in the trace panel
          </CardDescription>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="md:hidden"
            onClick={() => setTraceOpen(true)}
          >
            Trace
            {steps.length > 0 ? (
              <Badge variant="secondary" className="ml-1.5 font-mono text-xs">
                {steps.length}
              </Badge>
            ) : null}
          </Button>
          <Badge
            variant={
              connectionStatus === "connected" ? "default" : "destructive"
            }
            className="font-mono text-xs"
          >
            {connectionStatus}
          </Badge>
        </div>
      </CardHeader>

      <Separator />

      <CardContent className="flex min-h-0 flex-1 flex-col overflow-hidden p-0">
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <PrimaryChatColumn
              primaryEmpty={primaryEmpty}
              lastUserText={lastUserText}
              isProcessing={isProcessing}
              hitlPrompt={hitlPrompt}
              finalResponse={finalResponse}
              agentThoughts={agentThoughts}
              primaryEndRef={primaryEndRef}
              onHITLApprove={() => handleHITLResponse(true)}
              onHITLReject={() => handleHITLResponse(false)}
            />
          </div>

          <Separator orientation="vertical" className="hidden md:block" />

          <div className="hidden min-h-0 w-80 flex-col md:flex">
            <ToolPanel steps={steps} />
          </div>
        </div>

        {traceOpen ? (
          <MobileTraceOverlay
            steps={steps}
            traceEndRef={traceEndMobileRef}
            onClose={() => setTraceOpen(false)}
          />
        ) : null}

        <div className="border-t border-border p-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendQuery();
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask the agent..."
              className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              disabled={connectionStatus !== "connected" || isProcessing}
            />
            <Button
              type="submit"
              disabled={
                !query.trim() ||
                connectionStatus !== "connected" ||
                isProcessing
              }
            >
              Send
            </Button>
          </form>
        </div>
      </CardContent>
    </Card>
  );
}
