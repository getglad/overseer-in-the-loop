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
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type {
  AttackResultFrame,
  ScorecardFrame,
  ServerMessage,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { WSClient } from "@/lib/ws-client";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
// Bounded buffer: the corpus is ~45, but cap defensively like the agent panel.
const MAX_ATTACKS = 200;

type RunStatus = "idle" | "running" | "complete";
type Connection = "connected" | "disconnected";

const RED_TEXT_CLASS = "text-red-600 dark:text-red-400";
const FALSE_ALLOW_CLASS = `bg-red-500/15 ${RED_TEXT_CLASS}`;
const FALSE_BLOCK_CLASS = "bg-amber-500/15 text-amber-600 dark:text-amber-400";
const PASS_CLASS = "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400";

function verdict(result: AttackResultFrame): {
  label: string;
  className: string;
} {
  if (result.false_allow)
    return { label: "FALSE-ALLOW", className: FALSE_ALLOW_CLASS };
  if (result.false_block)
    return { label: "false-block", className: FALSE_BLOCK_CLASS };
  return { label: "caught", className: PASS_CLASS };
}

function MatrixRow({ result }: { result: AttackResultFrame }) {
  const { label, className } = verdict(result);
  return (
    <div className="flex items-center gap-3 border-b border-border/50 px-3 py-1.5 text-xs">
      <span
        className={cn(
          "w-24 shrink-0 rounded px-1.5 py-0.5 text-center font-medium",
          className,
        )}
      >
        {label}
      </span>
      <span
        className="w-56 shrink-0 truncate font-mono text-foreground"
        title={result.description}
      >
        {result.attack_id}
      </span>
      <Badge variant="outline" className="shrink-0">
        {result.category}
      </Badge>
      <span className="shrink-0 text-muted-foreground">
        {result.expected_blocked ? "expect block" : "expect allow"} →{" "}
        {result.observed_blocked ? "blocked" : "allowed"}
      </span>
      <span className="ml-auto shrink-0 font-mono text-muted-foreground">
        {result.layer}
      </span>
    </div>
  );
}

function Metric({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={cn("text-xl font-semibold tabular-nums", className)}>
        {value}
      </span>
    </div>
  );
}

function ScorecardSummary({ scorecard }: { scorecard: ScorecardFrame }) {
  const faClass = scorecard.false_allows > 0 ? RED_TEXT_CLASS : PASS_CLASS;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-6">
        <Metric
          label="Caught"
          value={`${scorecard.passed}/${scorecard.total}`}
        />
        <Metric
          label="Pass rate"
          value={`${Math.round(scorecard.pass_rate * 100)}%`}
        />
        <Metric
          label="False-allows"
          value={String(scorecard.false_allows)}
          className={faClass}
        />
        <Metric label="False-blocks" value={String(scorecard.false_blocks)} />
      </div>
      <Separator />
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
        {scorecard.by_category.map((breakdown) => (
          <div key={breakdown.category} className="flex justify-between gap-2">
            <span className="text-muted-foreground">{breakdown.category}</span>
            <span
              className={
                breakdown.false_allows > 0 ? RED_TEXT_CLASS : "text-foreground"
              }
            >
              {breakdown.passed}/{breakdown.total}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function RedTeamPanel() {
  const [status, setStatus] = useState<RunStatus>("idle");
  const [results, setResults] = useState<AttackResultFrame[]>([]);
  const [scorecard, setScorecard] = useState<ScorecardFrame | null>(null);
  const [connection, setConnection] = useState<Connection>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const clientRef = useRef<WSClient | null>(null);

  const handleMessage = useCallback((message: ServerMessage) => {
    switch (message.type) {
      case "redteam_result":
        setResults((prev) => {
          const next = [...prev, message.content];
          return next.length > MAX_ATTACKS ? next.slice(-MAX_ATTACKS) : next;
        });
        break;
      case "redteam_complete":
        setScorecard(message.content);
        setStatus("complete");
        break;
      case "error_message":
        setError(message.content);
        setStatus("idle");
        break;
      default:
        // Agent-loop frames share the /ws endpoint; this panel ignores them.
        break;
    }
  }, []);

  useEffect(() => {
    const client = new WSClient({
      url: WS_URL,
      onMessage: handleMessage,
      onOpen: () => setConnection("connected"),
      onClose: () => setConnection("disconnected"),
      onError: () => setConnection("disconnected"),
    });
    clientRef.current = client;
    client.connect();
    return () => client.disconnect();
  }, [handleMessage]);

  const runRedTeam = useCallback(() => {
    if (!clientRef.current?.isConnected || status === "running") return;
    setResults([]);
    setScorecard(null);
    setError(null);
    setStatus("running");
    clientRef.current.send({ type: "redteam_run", id: crypto.randomUUID() });
  }, [status]);

  const total = results[0]?.total ?? 0;
  const done = results.length;
  const buttonLabel =
    status === "running" ? `Running ${done}/${total || "…"}` : "Run red-team";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Red-team the gate</CardTitle>
            <CardDescription>
              Each attack is driven past agent refusal straight into the action
              classifier — every probe maps to the verdict it should produce.
            </CardDescription>
          </div>
          <Button
            type="button"
            onClick={runRedTeam}
            disabled={connection !== "connected" || status === "running"}
          >
            {buttonLabel}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {connection !== "connected" && (
          <p className="text-sm text-muted-foreground">
            Connecting to the gateway…
          </p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        {scorecard && <ScorecardSummary scorecard={scorecard} />}
        {results.length > 0 && (
          <>
            <Separator />
            <ScrollArea className="h-[28rem] rounded-md border border-border">
              <div>
                {results.map((result) => (
                  <MatrixRow key={result.attack_id} result={result} />
                ))}
              </div>
            </ScrollArea>
          </>
        )}
        {results.length === 0 && status !== "running" && !scorecard && (
          <p className="text-sm text-muted-foreground">
            Run the battery to stream the attack → decision matrix and a
            scorecard. The headline metric is{" "}
            <span className="font-medium text-foreground">false-allows</span> —
            dangerous actions the gate let through. This makes real model calls.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
