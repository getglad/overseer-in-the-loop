/**
 * WebSocket message types matching the server's MessageType enum.
 *
 * Protocol:
 * - Client → Server: user_message, user_interaction_message
 * - Server → Client: system_response_message, system_intermediate_message,
 *                     system_interaction_message, error_message
 */

// ---------------------------------------------------------------------------
// Message type discriminator
// ---------------------------------------------------------------------------

export type WebSocketMessageType =
  | "user_message"
  | "user_interaction_message"
  | "system_response_message"
  | "system_intermediate_message"
  | "system_interaction_message"
  | "error_message"
  | "redteam_run"
  | "redteam_result"
  | "redteam_complete";

export type WebSocketMessageStatus = "in_progress" | "complete";

// ---------------------------------------------------------------------------
// Client → Server
// ---------------------------------------------------------------------------

/** Start a new agent run with a user query. */
export interface UserMessage {
  type: "user_message";
  id: string;
  content: {
    messages: Array<{
      role: "user";
      content: Array<{ type: "text"; text: string }>;
    }>;
    /**
     * Evil toggle: when true, the server swaps real tool args with a
     * hardcoded exfiltration payload for classification only. Demo
     * affordance for guardrail testing.
     */
    evil_toggle?: boolean;
  };
  timestamp?: string;
}

/** HITL response — approval or rejection of a pending action. */
export interface UserInteractionMessage {
  type: "user_interaction_message";
  id: string;
  thread_id: string;
  parent_id: string;
  conversation_id?: string | null;
  content: {
    messages: Array<{
      role: "user";
      content: Array<{ type: "text"; text: string }>;
    }>;
  };
  timestamp?: string;
}

/** Trigger a red-team run — drive the attack corpus through the gate. No body. */
export interface RedTeamRunMessage {
  type: "redteam_run";
  id: string;
  timestamp?: string;
}

// ---------------------------------------------------------------------------
// Server → Client
// ---------------------------------------------------------------------------

/** Intermediate step — reasoning trace, tool calls, function execution. */
export interface SystemIntermediateMessage {
  type: "system_intermediate_message";
  id: string;
  parent_id: string;
  conversation_id?: string | null;
  content: {
    name: string;
    payload: string;
  };
  status: WebSocketMessageStatus;
  timestamp: string;
}

/** HITL prompt — binary choice (approve/reject) or text input. */
export interface SystemInteractionMessage {
  type: "system_interaction_message";
  id: string;
  parent_id: string;
  conversation_id?: string | null;
  content: HumanPrompt;
  status: "in_progress";
  timestamp: string;
}

/** Final agent response or streaming token. */
export interface SystemResponseMessage {
  type: "system_response_message";
  id: string;
  parent_id?: string;
  conversation_id?: string | null;
  content: string;
  status: WebSocketMessageStatus;
  timestamp: string;
}

/** Error from the server. */
export interface ErrorMessage {
  type: "error_message";
  id?: string;
  content: string;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Red-team frames (mirror src/redteam/models.py — keep in lockstep)
// ---------------------------------------------------------------------------

/** One attack's outcome, streamed as it completes. */
export interface AttackResultFrame {
  attack_id: string;
  category: string;
  description: string;
  tool_name: string;
  expected_blocked: boolean;
  observed_blocked: boolean;
  passed: boolean;
  layer: string;
  reason: string;
  /** Block-expected attack the gate allowed — the headline security failure. */
  false_allow: boolean;
  /** Allow-expected control the gate blocked — over-paranoia. */
  false_block: boolean;
  index: number;
  total: number;
}

/** Per-category roll-up inside the scorecard. */
export interface CategoryBreakdownFrame {
  category: string;
  total: number;
  passed: number;
  false_allows: number;
  false_blocks: number;
}

/** Aggregate scorecard, sent once the run completes. */
export interface ScorecardFrame {
  total: number;
  passed: number;
  false_allows: number;
  false_blocks: number;
  pass_rate: number;
  by_category: CategoryBreakdownFrame[];
}

/** A single attack result during a red-team run. */
export interface RedTeamResultMessage {
  type: "redteam_result";
  id: string;
  content: AttackResultFrame;
  timestamp: string;
}

/** The final scorecard once every attack has run. */
export interface RedTeamCompleteMessage {
  type: "redteam_complete";
  id: string;
  content: ScorecardFrame;
  status: "complete";
  timestamp: string;
}

// ---------------------------------------------------------------------------
// HITL prompt types (discriminated union on input_type)
// ---------------------------------------------------------------------------

export type HumanPrompt = HumanPromptBinary | HumanPromptText;

export interface HumanPromptBinary {
  input_type: "binary_choice";
  text: string;
  options: BinaryOption[];
  timeout?: number | null;
}

export interface HumanPromptText {
  input_type: "text";
  text: string;
  timeout?: number | null;
}

export interface BinaryOption {
  id: string;
  label: string;
  value: string;
}

// ---------------------------------------------------------------------------
// Union of all message types
// ---------------------------------------------------------------------------

export type ClientMessage =
  | UserMessage
  | UserInteractionMessage
  | RedTeamRunMessage;

export type ServerMessage =
  | SystemIntermediateMessage
  | SystemInteractionMessage
  | SystemResponseMessage
  | ErrorMessage
  | RedTeamResultMessage
  | RedTeamCompleteMessage;

export type WebSocketMessage = ClientMessage | ServerMessage;
