/** TypeScript mirrors of the backend's real Pydantic models (dev doc
 * §12.2's own rule extended across the wire: the frontend's types
 * follow the API's real response shapes exactly, not an independently
 * invented schema). See `src/udt/common/models.py` and
 * `src/udt/logging/decision_log.py` for the source of truth. */

export type AssetType = "hospital" | "substation" | "water" | "road" | "ambulance" | "ecc";

export interface Geometry {
  type: string;
  coordinates: unknown;
}

export interface Asset {
  asset_id: string;
  asset_type: AssetType;
  geometry: Geometry;
  intrinsic_level: number;
  functional_level: number;
  attributes: Record<string, unknown>;
}

export interface DependencyEdge {
  edge_id: string;
  supplier: string;
  consumer: string;
  kind: "power" | "water" | "access";
  demand: number;
  criticality: number;
  buffer_hours: number;
  floor: number;
}

export interface DependencyGraph {
  version: string;
  assets: Asset[];
  edges: DependencyEdge[];
}

export interface TwinState {
  tick: number;
  assets: Asset[];
  cascading_failure_count: number;
  patient_deaths_cumulative: number;
  ambulance_response_times_this_tick: number[];
  pending_requests_count: number;
  patients_transferred_cumulative: number;
  safety_violations_attempted_cumulative: number;
}

export interface AgentAction {
  repair_target: string | null;
  ambulance_assignment: Record<string, string> | null;
  patient_transfer: [string, string, number] | null;
  shed_tier: Record<string, number> | null;
}

export interface RoutingDecision {
  path: "ID" | "OOD";
  registry_hit: boolean;
  support_score: number;
  reasons: string[];
}

export interface GoalWeights {
  g_health: number;
  g_power: number;
  g_transport: number;
  g_cost: number;
}

export interface Directive {
  kind: string;
  details: Record<string, string>;
}

export interface Plan {
  plan_id: string;
  objective: string;
  goal_weights: GoalWeights;
  priority_assets: string[];
  directives: Directive[];
  rationale: string;
  assumptions: string[];
  expected_outcome: string;
}

export interface RolloutOutcome {
  min_hospital_functional_level: number;
  min_critical_functional_level: number;
  unmet_patient_hours: number;
  new_cascading_failures: number;
}

export interface SimulationResult {
  rollouts: RolloutOutcome[];
  p_failure: number;
  mean_outcome: number;
  std_outcome: number;
}

export interface PlanSummary {
  plan: Plan;
  simulation_summary: SimulationResult;
}

export interface ConstraintViolation {
  rule_id: string;
  scope: "health" | "power" | "transport";
  on_violation: "clip" | "drop" | "reject";
  detail: string;
}

export interface ConstraintReport {
  passed: boolean;
  violations: ConstraintViolation[];
  repaired_action: AgentAction;
}

export type AutonomyTier = "AUTONOMOUS" | "EXECUTE_AND_FLAG" | "HUMAN_APPROVAL";

export interface RiskSummary {
  p_failure: number;
  consequence: number;
  risk: number;
  confidence: number;
  tier: AutonomyTier;
}

export interface ApprovalSummary {
  required: boolean;
  response: "approve" | "reject" | "modify" | null;
  latency: number;
  timeout_triggered: boolean;
}

export interface LLMCallSummary {
  prompt_version: string | null;
  model: string | null;
  tokens: number;
  cache_hit: boolean;
  fallback: boolean;
}

export interface MarlActionSummary {
  policy_id: string;
  action: Record<string, unknown>;
  masked_actions_count: number;
}

export interface DecisionRecord {
  decision_id: string;
  episode_id: string;
  tick: number;
  twin_state_digest: string;
  incident_id: string;
  routing: RoutingDecision;
  llm: LLMCallSummary;
  plans: PlanSummary[];
  selected_plan_id: string | null;
  marl: MarlActionSummary;
  constraint_report: ConstraintReport;
  risk: RiskSummary;
  approval: ApprovalSummary;
  executed_action: AgentAction;
  realized_outcome_next_interval: Record<string, number> | null;
}

export interface CreateEpisodeResponse {
  episode_id: string;
  incident_id: string;
  severity: number;
}

export interface StepResponse {
  tick: number;
  snapshots: TwinState[];
  decisions: DecisionRecord[];
  episode_ended: boolean;
}
