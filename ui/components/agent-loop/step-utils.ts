/** Shared utilities and types for rendering agent trace steps.
 *
 * Lives here (not in agent-loop-parts) because both the agent-loop column
 * and the tool-panel column render steps and need the same shapes.
 */

import type { BinaryOption, WebSocketMessageStatus } from "@/lib/types";

/** A single intermediate step from the agent trace stream. */
export interface Step {
  id: string;
  name: string;
  payload: string;
  status: WebSocketMessageStatus;
  timestamp: string;
}

/** A pending HITL approval prompt awaiting user input. */
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

/** Step category for visual grouping — tools get colored badges. */
export interface StepCategory {
  label: string;
  color: string;
}

const SEARCH: StepCategory = {
  label: "search",
  color: "bg-green-500/10 text-green-500",
};
const SHELL: StepCategory = {
  label: "shell",
  color: "bg-red-500/10 text-red-500",
};
const TOOL: StepCategory = {
  label: "tool",
  color: "bg-orange-500/10 text-orange-500",
};
const LLM: StepCategory = {
  label: "llm",
  color: "bg-indigo-500/10 text-indigo-500",
};
const FN: StepCategory = {
  label: "fn",
  color: "bg-slate-500/10 text-slate-500",
};

const STEP_CATEGORIES: Record<string, StepCategory> = {
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

const DEFAULT_CATEGORY: StepCategory = {
  label: "step",
  color: "bg-muted text-muted-foreground",
};

// NAT emits qualified step names like `getglad_tools__read_file`; the
// substring-match fallback recovers the bare tool name. Hoisted to a
// module-level constant so we don't reallocate the entries array on every
// step render.
const STEP_CATEGORY_ENTRIES: ReadonlyArray<[string, StepCategory]> =
  Object.entries(STEP_CATEGORIES);

/** Get the visual category for a step by name (exact or substring match). */
export function getStepCategory(name: string): StepCategory {
  // Object.hasOwn (not `in`) so a step literally named "constructor"/"toString"
  // doesn't match an inherited Object.prototype key and return a garbage category.
  if (Object.hasOwn(STEP_CATEGORIES, name)) {
    return STEP_CATEGORIES[name];
  }
  for (const [key, cat] of STEP_CATEGORY_ENTRIES) {
    if (name.includes(key)) {
      return cat;
    }
  }
  return DEFAULT_CATEGORY;
}

export interface PayloadField {
  /** Stable identity for React keys — same content lines repeat (e.g. duplicate
   *  grep matches), so the parse-order index is the only reliable identity. */
  id: string;
  key: string;
  value: string;
}

/** Parse "key: value" lines from a multi-line payload. */
function parseColonFields(payload: string): PayloadField[] {
  const fields: PayloadField[] = [];
  payload.split("\n").forEach((line, index) => {
    const idx = line.indexOf(": ");
    // Only treat as a field when the key is a single token — otherwise prose
    // with a mid-sentence ": " (e.g. "I think: maybe") renders its prefix as a
    // bold field label. Structured payloads use single-word keys (input/output).
    if (idx > 0 && !line.slice(0, idx).includes(" ")) {
      fields.push({
        id: `${index}`,
        key: line.slice(0, idx).trim(),
        value: line.slice(idx + 2).trim(),
      });
    } else if (line.trim()) {
      fields.push({ id: `${index}`, key: "", value: line.trim() });
    }
  });
  return fields;
}

/** Try to parse a payload string into labeled fields. Returns null if not structured. */
export function parsePayloadFields(payload: string): PayloadField[] | null {
  if (!payload.includes(": ")) return null;
  const fields = parseColonFields(payload);
  return fields.length > 0 ? fields : null;
}
