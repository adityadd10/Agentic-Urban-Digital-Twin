import type { DecisionRecord } from "../types";

function tierClass(tier: DecisionRecord["risk"]["tier"]): string {
  if (tier === "AUTONOMOUS") return "tier-autonomous";
  if (tier === "EXECUTE_AND_FLAG") return "tier-flag";
  return "tier-approval";
}

interface Props {
  decisions: DecisionRecord[];
  onSelect: (decision: DecisionRecord) => void;
}

/** dev doc §11.3's Timeline pane: "decision cards: what/why/risk tier",
 * newest first. Clicking a card opens the full detail in `ApprovalModal`
 * (used for every decision, not only ones needing approval — the name
 * follows dev doc's own §11.3 naming, the modal itself shows any
 * decision's full plan/risk/constraint detail regardless). */
export function Timeline({ decisions, onSelect }: Props) {
  return (
    <div className="timeline">
      <h2>Timeline</h2>
      {decisions.length === 0 && <p className="empty-hint">No decisions yet — step the episode.</p>}
      <ul>
        {[...decisions].reverse().map((d) => (
          <li
            key={d.decision_id}
            className={`decision-card ${tierClass(d.risk.tier)}`}
            onClick={() => onSelect(d)}
          >
            <div className="decision-card-header">
              <span>tick {d.tick}</span>
              <span className="tier-badge">{d.risk.tier.replace("_", " ")}</span>
            </div>
            <div className="decision-card-body">
              <div>
                <strong>{d.routing.path}</strong> path{d.llm.fallback ? " · LLM fallback" : ""}
              </div>
              <div>
                P(failure)={d.risk.p_failure.toFixed(2)} · confidence={d.risk.confidence.toFixed(2)}
              </div>
              {d.approval.required && <div>approval: {d.approval.response ?? "pending"}</div>}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
