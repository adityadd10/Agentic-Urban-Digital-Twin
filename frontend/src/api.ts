/** Thin REST/WebSocket client for module M10b's FastAPI service (dev
 * doc §11.2). No auth (dev doc §11.3's own words: "No auth beyond a
 * static token; this is a demo surface, not a public product") — not
 * even the static token in this build, matching `api/main.py`'s own
 * disclosure. */

import type {
  CreateEpisodeResponse,
  DecisionRecord,
  DependencyGraph,
  StepResponse,
  TwinState,
} from "./types";

// Empty by default -> same-origin requests, proxied to the real
// backend by `vite.config.ts` during development. Set
// `VITE_API_BASE_URL` for a production build talking to a
// differently-hosted backend.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${options?.method ?? "GET"} ${path} -> ${response.status}: ${body}`);
  }
  return (await response.json()) as T;
}

export function createEpisode(seed: number): Promise<CreateEpisodeResponse> {
  return request("/episodes", { method: "POST", body: JSON.stringify({ seed }) });
}

export function stepEpisode(episodeId: string, n: number): Promise<StepResponse> {
  return request(`/episodes/${episodeId}/step?n=${n}`, { method: "POST" });
}

export function getTwinState(episodeId: string): Promise<TwinState> {
  return request(`/twin/state?episode_id=${encodeURIComponent(episodeId)}`);
}

export function getTwinGraph(): Promise<DependencyGraph> {
  return request("/twin/graph");
}

export function getDecisions(episodeId: string): Promise<DecisionRecord[]> {
  return request(`/decisions?episode_id=${encodeURIComponent(episodeId)}`);
}

export function injectIncident(episodeId: string, severity: number): Promise<{ incident_id: string }> {
  return request("/incidents", {
    method: "POST",
    body: JSON.stringify({ episode_id: episodeId, severity }),
  });
}

export function overrideApproval(
  decisionId: string,
  response: "approve" | "reject" | "modify",
): Promise<DecisionRecord> {
  return request(`/approvals/${decisionId}`, { method: "POST", body: JSON.stringify({ response }) });
}

export interface LiveMessage {
  type: "step" | "incident_injected";
  [key: string]: unknown;
}

/** dev doc §11.2's `WS /ws/live` — one connection per subscribed
 * episode, pushed a message every time `POST /episodes/{id}/step` or
 * `POST /incidents` runs against it (`api/ws.py`'s `ConnectionManager`).
 * The caller re-fetches whatever state it needs on each message rather
 * than this client trying to keep a full local mirror in sync. */
export function connectLiveSocket(episodeId: string, onMessage: (msg: LiveMessage) => void): WebSocket {
  const base = API_BASE || window.location.origin;
  const wsUrl = new URL(base);
  wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
  wsUrl.pathname = "/ws/live";
  wsUrl.search = `episode_id=${encodeURIComponent(episodeId)}`;
  const ws = new WebSocket(wsUrl.toString());
  ws.onmessage = (event: MessageEvent<string>) => onMessage(JSON.parse(event.data) as LiveMessage);
  return ws;
}
