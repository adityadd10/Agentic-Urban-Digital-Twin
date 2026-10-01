# Pre-registered: SOP ("current practice") baseline for twin-v3

**Written 2026-10-02, before the SOP policy is implemented or run.** Committed to git so the
timestamp is verifiable.

**Context.** Gates round 0 and eight feasibility tests showed that the tuned RB-S is near the best
achievable policy in twin-v3 (`results/twin_v3_gates/`). The same tests showed that the *choice*
of rule matters a great deal (evacuation rules: 70.6–105.4 deaths). Real responders follow
written procedures, not a rule tuned on 60 simulated floods. This adds an **untuned baseline coded
from published guidance**, to measure the gap between current practice and the best achievable.

**Comparators.** RB-S stays in every comparison. Any MARL claim reports both gaps (vs SOP and vs
RB-S).

## 1. Sources

- **NDMA (2016), *National Disaster Management Guidelines: Hospital Safety*** (NIDM PDF, read in
  full text):
  - triage (§4.9, "the sickest is seen first"; Category I zero delay, II within 10 min);
  - surge (§4.10: shift non-critical patients, designate overflow areas, inter-facility transfer);
  - area networking (§4.14: when surge capacity is exceeded, transfer "to the nearest equipped
    hospital for treatment without any delay");
  - hospital affected: "partial or complete evacuation and transfer of critical patients to
    networked hospitals".
- **NDMA (2010), *Management of Urban Flooding*:** post-flood restoration of power "will get top
  priority"; hospitals face "loss of essential services, like water supply, electricity".
- **108 emergency response service (GVK EMRI and state NHM descriptions):** the call goes to "the
  ambulance located nearest to the site"; the patient goes to "the nearest suitable medical
  facility".
- **Mumbai utility practice (news reports):** supply maintained to "vital installations" during
  disturbances.
- **Not yet read:** the CEA *Disaster Management Plan for Power Sector* (2022), which may refine
  the restoration order; disclosed.

## 2. SOP policy in the v3 action interface (no tuned thresholds; the most natural reading of each rule)

**Health (per hospital):**
- **Overflow / surge (§4.10):** surge on when the hospital has no effective free beds and a queue
  > 0 (budget permitting).
- **Area networking (§4.14):** when it is full (no effective free beds) and patients are queued,
  request a transfer of 2 patients, urgent ("without any delay"), unless transfer jobs from it are
  already pending.
- **Hospital affected (evacuation):** if the hospital is physically damaged (intrinsic < 1) or a
  supply has failed (a power or water buffer is exhausted), evacuate:
  - divert on;
  - request an urgent transfer of 5 patients each decision while patients remain.
- Divert is otherwise off. The guidance gives no emergency-department diversion rule other than
  evacuation.

**Power (crew):**
- Restoration priority for vital installations: (1) damaged substations that directly supply a
  hospital, (2) damaged pumps that supply a hospital, (3) any other damaged facility.
- Within a class, the most damaged first; reachable sites only. Keep the current target while it
  is still damaged.

**Transport (108 practice):**
- Every visible job is served, in the env's fixed order, by the nearest available ambulance (the
  env assigns it).
- The destination is the "nearest suitable facility": the nearest reachable hospital that is
  functioning (functional level ≥ 0.5), ignoring bed counts and divert status. A transfer never
  goes to its own source. If none qualifies, the nearest reachable.

## 3. Headroom check (train only; no MARL, no oracle)

- **Runs:** SOP vs RB-S on the 20 train scenarios × seeds k = 0, 1, 2. RB-S's episodes are the
  frozen reference (`results/twin_v3_gates/rbs_v3_train_reference.json`).
- **Statistics:** per-scenario means, paired difference, 95% bootstrap CI, Wilcoxon.
- **Reading (fixed now):** the SOP leaves meaningful room for a learner if its mean deaths exceed
  RB-S's by ≥ 15% of the SOP mean, with the CI excluding 0. RB-S is a proxy for the best
  achievable: the oracle was within +1.6%. Otherwise current practice is already near-best and
  this framing fails as well.
- **Discipline:** the SOP is never tuned after this run; its definition here is binding.
