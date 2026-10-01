# Feasibility tests 4A and 4C: "hard questions" (prototype only)

**Written 2026-10-02, before either prototype is implemented or run.** User-requested. Prototype
only: no change to `src/`, the protocol or any gate. The same 10 moderate/severe train scenarios,
seed k = 0.

## 4A: Changing priorities (frozen twin-v3, no acuity)

RB-S is written for one objective (patients). The question is whether a policy that knows the
commander's current objective has room to do better.

**Objectives.** Each term is normalised by RB-S's train mean (margins.json detail): D0 = 65.42
deaths, U0 = 375.42 unmet p-h, C0 = 1 − 0.6438 = 0.3562 critical-function loss, T0 = 2.2506 h
casualty time to admission.
- **O2 "restore infrastructure":** J = 0.2·deaths/D0 + 0.2·unmet/U0 + 1.0·(1 − critical
  function)/C0.
- **O3 "fast emergency response":** J = 0.2·deaths/D0 + 0.2·unmet/U0 + 1.0·time-to-admission/T0
  (0 if there are no casualties).

O1 (patients) was already tested: oracle +1.6%.

**Conditions per objective:**
- RB-S (fixed);
- an oracle re-chosen hourly among the 30 strategies of the oracle diagnostic, with perfect
  foresight, 6 h scoring of J (the strategy for 1 h, then RB-S).

**Reading:** headroom for an objective if the oracle's mean J is ≥ 15% below RB-S's mean J.

## 4C: Evacuation under uncertainty (twin-v3 + acuity mechanics of test 2)

**Lever.** Evacuating a hospital = divert on, plus an urgent transfer request of 5 patients each
decision while patients remain. A diverting hospital's transfers take ICU patients first. This is
the MIOT pattern: critical patients moved before power is lost.

**Fixed evacuation rules** (RB-S otherwise), by trigger:
- E0: never (= RB-S);
- E1: power buffer being drawn with < 2 h left;
- E2: flood depth at the hospital ≥ 0.2 m;
- E3: flood depth ≥ 0.4 m;
- E4: aggressive (buffer < 6 h or depth ≥ 0.1 m).

**Conditions:**
1. Each fixed rule E0–E4 for the whole episode. "Best fixed" = the lowest mean deaths among them,
   chosen in hindsight on these scenarios, which is generous to rules.
2. **Fair planner (no foresight):** every hour it chooses among E0–E4 by mean (new deaths + unmet
   p-h) over 4 rollouts of 6 h. Each rollout redraws facility fragility (conditional on depth
   already survived) and the random stream, so the planner does not know which facilities will
   fail or which patients will arrive.
3. **Perfect-foresight oracle** over E0–E4, for reference only.

**Reading:** headroom for a learner if the fair planner's mean deaths are ≥ 15% below the best
fixed rule's. The perfect-foresight gap is reported, but it does not count as headroom.
