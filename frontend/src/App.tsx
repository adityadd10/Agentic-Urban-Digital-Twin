import { useEffect, useState } from "react";
import * as api from "./api";
import { ApprovalModal } from "./components/ApprovalModal";
import { ControlBar } from "./components/ControlBar";
import { MapPane } from "./components/MapPane";
import { MetricsStrip } from "./components/MetricsStrip";
import { Timeline } from "./components/Timeline";
import type { DecisionRecord, DependencyGraph, TwinState } from "./types";

/** dev doc §11.3's four-pane thesis demo, tied together over module
 * M10b's real FastAPI service. No auth (disclosed in `api.ts`); a
 * fresh page load always starts with no episode selected — episode
 * state lives only in this component's memory and the backend's own
 * in-memory `EpisodeManager` (module M10b's own disclosed scope, no
 * Postgres yet), so a reload loses the UI's place even though the
 * decision log itself is still real and on disk. */
export default function App() {
  const [episodeId, setEpisodeId] = useState<string | null>(null);
  const [graph, setGraph] = useState<DependencyGraph | null>(null);
  const [twinState, setTwinState] = useState<TwinState | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [ended, setEnded] = useState(false);
  const [showEdges, setShowEdges] = useState(false);
  const [selectedDecision, setSelectedDecision] = useState<DecisionRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getTwinGraph().then(setGraph).catch((e: unknown) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!episodeId) return undefined;
    const ws = api.connectLiveSocket(episodeId, (msg) => {
      if (msg.type === "step") {
        api.getTwinState(episodeId).then(setTwinState).catch(() => undefined);
        api.getDecisions(episodeId).then(setDecisions).catch(() => undefined);
      }
    });
    return () => ws.close();
  }, [episodeId]);

  async function handleCreateEpisode(seed: number): Promise<void> {
    setError(null);
    try {
      const response = await api.createEpisode(seed);
      setEpisodeId(response.episode_id);
      setTwinState(null);
      setDecisions([]);
      setEnded(false);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleStep(n: number): Promise<void> {
    if (!episodeId) return;
    setError(null);
    try {
      const response = await api.stepEpisode(episodeId, n);
      const latest = response.snapshots[response.snapshots.length - 1];
      if (latest) setTwinState(latest);
      setDecisions((prev) => [...prev, ...response.decisions]);
      setEnded(response.episode_ended);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleInjectIncident(severity: number): Promise<void> {
    if (!episodeId) return;
    setError(null);
    try {
      await api.injectIncident(episodeId, severity);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleApprovalResponse(
    decisionId: string,
    response: "approve" | "reject" | "modify",
  ): Promise<void> {
    setError(null);
    try {
      const updated = await api.overrideApproval(decisionId, response);
      setDecisions((prev) => prev.map((d) => (d.decision_id === updated.decision_id ? updated : d)));
      setSelectedDecision(updated);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Urban Digital Twin — Demo Console</h1>
        <p className="subtitle">dev doc §11.3 — thesis demo surface, not a public product.</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <ControlBar
        episodeId={episodeId}
        ended={ended}
        onCreateEpisode={handleCreateEpisode}
        onStep={handleStep}
        onInjectIncident={handleInjectIncident}
      />

      <label className="edge-toggle">
        <input
          type="checkbox"
          checked={showEdges}
          onChange={(e) => setShowEdges(e.target.checked)}
        />
        show dependency edges
      </label>

      <MetricsStrip twinState={twinState} decisions={decisions} />

      <div className="main-panes">
        <MapPane
          assets={twinState?.assets ?? graph?.assets ?? []}
          edges={graph?.edges ?? []}
          showEdges={showEdges}
        />
        <Timeline decisions={decisions} onSelect={setSelectedDecision} />
      </div>

      <ApprovalModal
        decision={selectedDecision}
        onClose={() => setSelectedDecision(null)}
        onRespond={handleApprovalResponse}
      />
    </div>
  );
}
