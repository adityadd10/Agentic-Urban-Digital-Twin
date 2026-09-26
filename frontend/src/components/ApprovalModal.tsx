import type { DecisionRecord } from "../types";

interface Props {
  decision: DecisionRecord | null;
  onClose: () => void;
  onRespond: (decisionId: string, response: "approve" | "reject" | "modify") => void;
}

/** dev doc §11.3's Approval modal: "plan vs. simulated outcomes vs.
 * risk — approve/reject/modify". **Disclosed:** every decision this
 * build produces has already been resolved headlessly by `risk/
 * human_model.py`'s `HumanModel` before it's ever logged (module M10a)
 * — approve/reject here calls `POST /approvals/{decision_id}`, which
 * *annotates* the recorded response for audit purposes; it does not
 * rewind the simulation or re-execute a different action (`api/
 * schemas.py`'s `ApprovalOverrideRequest` docstring has the full
 * reasoning). "Modify" is disabled for the same reason — there is no
 * server-side path yet that takes a `modified_action` and re-runs
 * anything with it. */
export function ApprovalModal({ decision, onClose, onRespond }: Props) {
  if (!decision) return null;
  const selectedPlan = decision.plans.find((p) => p.plan.plan_id === decision.selected_plan_id);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Decision — tick {decision.tick}</h2>
        <p className="modal-subtitle">
          {decision.routing.path} path · support score {decision.routing.support_score.toFixed(3)}
        </p>

        {selectedPlan ? (
          <section>
            <h3>Selected plan</h3>
            <p>{selectedPlan.plan.objective}</p>
            <p className="rationale">{selectedPlan.plan.rationale}</p>
            <p>
              Simulated outcome: P(failure)={selectedPlan.simulation_summary.p_failure.toFixed(2)},
              mean outcome={selectedPlan.simulation_summary.mean_outcome.toFixed(1)}
            </p>
          </section>
        ) : (
          <section>
            <h3>Proposed action (OOD path)</h3>
            <pre>{JSON.stringify(decision.executed_action, null, 2)}</pre>
          </section>
        )}

        <section>
          <h3>Risk</h3>
          <p>
            P(failure)={decision.risk.p_failure.toFixed(2)} · consequence=
            {decision.risk.consequence.toFixed(1)} · risk={decision.risk.risk.toFixed(2)} ·
            confidence={decision.risk.confidence.toFixed(2)}
          </p>
          <p>
            tier: <strong>{decision.risk.tier}</strong>
          </p>
        </section>

        <section>
          <h3>Constraint check</h3>
          <p>
            {decision.constraint_report.passed
              ? "passed — no violations"
              : `${decision.constraint_report.violations.length} violation(s) caught and repaired`}
          </p>
        </section>

        <section>
          <h3>Approval</h3>
          <p>
            required: {decision.approval.required ? "yes" : "no"} — recorded response:{" "}
            {decision.approval.response ?? "n/a"}
          </p>
          {decision.approval.required && (
            <div className="modal-actions">
              <button onClick={() => onRespond(decision.decision_id, "approve")}>Approve</button>
              <button onClick={() => onRespond(decision.decision_id, "reject")}>Reject</button>
              <button
                disabled
                title="Not wired to re-execute in this build — see api/schemas.py's ApprovalOverrideRequest"
              >
                Modify
              </button>
            </div>
          )}
        </section>

        <button className="modal-close" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
