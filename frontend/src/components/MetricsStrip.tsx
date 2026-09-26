import type { DecisionRecord, TwinState } from "../types";

interface Props {
  twinState: TwinState | null;
  decisions: DecisionRecord[];
}

/** dev doc §11.3's Metrics strip: "live unmet demand, cascades, %
 * autonomous". `patient_queue` is read directly off each hospital's
 * live `attributes` (the same field `logging/metrics.py`'s own
 * `unmet_patient_hours` is built from). */
export function MetricsStrip({ twinState, decisions }: Props) {
  const patientsQueued = twinState
    ? twinState.assets.reduce((sum, a) => sum + (Number(a.attributes.patient_queue) || 0), 0)
    : 0;
  const cascades = twinState?.cascading_failure_count ?? 0;
  const autonomousPct = decisions.length
    ? (100 * decisions.filter((d) => d.risk.tier === "AUTONOMOUS").length) / decisions.length
    : 0;

  return (
    <div className="metrics-strip">
      <div className="metric">
        <span className="metric-value">{patientsQueued}</span>
        <span className="metric-label">patients queued</span>
      </div>
      <div className="metric">
        <span className="metric-value">{cascades}</span>
        <span className="metric-label">cascading failures</span>
      </div>
      <div className="metric">
        <span className="metric-value">{autonomousPct.toFixed(0)}%</span>
        <span className="metric-label">autonomous decisions</span>
      </div>
      <div className="metric">
        <span className="metric-value">{decisions.length}</span>
        <span className="metric-label">decisions logged</span>
      </div>
    </div>
  );
}
