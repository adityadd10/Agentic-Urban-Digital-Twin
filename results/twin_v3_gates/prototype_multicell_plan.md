# Feasibility test 3: two-cell rainfall + acuity (prototype only)

**Written 2026-10-01, before the prototype is implemented or run.** It follows test 2 (acuity, no
headroom) per that plan's next step, at the user's request. Prototype only: no change to `src/`,
the protocol or any gate.

**Mechanics:**
- Everything from test 2 (acuity: triage, time to care, ICU power dependence).
- **A second storm cell** per scenario, drawn deterministically from the scenario seed with the
  generator's own ranges:
  - its centre is uniform inside the ward;
  - σ U(800, 2500) m; severity U(0.2, 1.0);
  - growth / hold / recede U(1, 3) / U(4, 10) / U(4, 8) h;
  - **onset 2–4 h after the first cell**.

  Flood depth everywhere is the sum of both cells (facilities, dependency-graph roads and the
  routing network). Facility fragility (critical depths) is unchanged. Emergency calls from the
  second cell are generated at the same severity-scaled rate once it starts.

  Disclosed simplification: the substation load surge follows the first cell only.

**Conditions:** the same 10 moderate/severe train scenarios, seed k = 0, all inside the
prototype. RB-S; the simple acuity-aware rule (test 2); and the oracle over the same 45
strategies (perfect foresight, re-chosen hourly, 6 h scoring).

**Reading (fixed now):** if the oracle is ≥ 15% below the better of RB-S and the rule in mean
deaths, there is headroom. Only then are two-cell and acuity separated, to attribute it.
Otherwise no headroom in this form.
