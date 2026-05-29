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
  | "error_message";

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

export type ClientMessage = UserMessage | UserInteractionMessage;

export type ServerMessage =
  | SystemIntermediateMessage
  | SystemInteractionMessage
  | SystemResponseMessage
  | ErrorMessage;

export type WebSocketMessage = ClientMessage | ServerMessage;
