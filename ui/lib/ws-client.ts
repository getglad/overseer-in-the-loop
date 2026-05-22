/**
 * WebSocket client abstraction for communicating with the agent gateway.
 *
 * Handles connection lifecycle, reconnection, and typed message dispatching.
 * Shared by all dashboard panels — AgentLoopPanel, guardrails, planner, etc.
 */

import type {
  ClientMessage,
  ServerMessage,
  WebSocketMessageType,
} from "./types";

type MessageHandler = (message: ServerMessage) => void;

export interface WSClientOptions {
  /** WebSocket URL (e.g., "ws://localhost:8000/ws") */
  url: string;
  /** Called when a server message arrives */
  onMessage: MessageHandler;
  /** Called when the connection opens */
  onOpen?: () => void;
  /** Called when the connection closes */
  onClose?: (event: CloseEvent) => void;
  /** Called on connection error */
  onError?: (event: Event) => void;
  /** Auto-reconnect on disconnect. Default: true */
  reconnect?: boolean;
  /** Reconnect base delay in ms. Default: 2000 */
  reconnectDelay?: number;
  /** Max reconnect attempts. Default: 5 */
  maxReconnectAttempts?: number;
}

const MAX_BACKOFF_MS = 30_000;
const JITTER_MAX_MS = 1_000;

export class WSClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionalClose = false;
  private readonly options: Required<WSClientOptions>;

  constructor(options: WSClientOptions) {
    this.options = {
      onOpen: () => {},
      onClose: () => {},
      onError: () => {},
      reconnect: true,
      reconnectDelay: 2000,
      maxReconnectAttempts: 5,
      ...options,
    };
  }

  /** Open the WebSocket connection. Resets the reconnect counter (fresh intent). */
  connect(): void {
    this.intentionalClose = false;
    this.reconnectAttempts = 0;

    // Clear any pending reconnect timer to avoid double connections
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.openSocket();
  }

  /**
   * Open the underlying socket WITHOUT resetting the reconnect counter.
   * The reconnect timer calls this (not connect()) so the maxReconnectAttempts
   * cap is actually reached and exponential backoff grows — otherwise a
   * permanently-down server would be hammered at the base delay forever.
   */
  private openSocket(): void {
    this.ws = new WebSocket(this.options.url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.options.onOpen();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") {
        return; // ignore binary frames
      }
      try {
        const data = JSON.parse(event.data) as ServerMessage;
        this.options.onMessage(data);
      } catch {
        this.options.onError(
          new ErrorEvent("parse", {
            message: `Failed to parse WebSocket message: ${event.data.slice(0, 200)}`,
          }),
        );
      }
    };

    this.ws.onclose = (event: CloseEvent) => {
      this.options.onClose(event);
      if (!this.intentionalClose && this.options.reconnect) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (event: Event) => {
      this.options.onError(event);
    };
  }

  /** Send a typed message to the server. Returns true if sent. */
  send(message: ClientMessage): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  /** Close the connection without auto-reconnecting. */
  disconnect(): void {
    this.intentionalClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  /** Current connection state. */
  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  /** Whether the connection is open. */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      return;
    }
    // Exponential backoff with jitter
    const delay = Math.min(
      this.options.reconnectDelay * 2 ** this.reconnectAttempts +
        Math.random() * JITTER_MAX_MS,
      MAX_BACKOFF_MS,
    );
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => {
      this.openSocket();
    }, delay);
  }
}

/**
 * Create a type-safe message filter for a specific server message type.
 *
 * Usage:
 *   const isIntermediate = messageFilter("system_intermediate_message");
 *   if (isIntermediate(msg)) { msg.content.name; } // typed
 */
export function messageFilter<T extends WebSocketMessageType>(type: T) {
  return (msg: ServerMessage): msg is Extract<ServerMessage, { type: T }> =>
    msg.type === type;
}
