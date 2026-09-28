# Pre-registered: twin-v3 success criteria and design gates

**Written 2026-09-29, before any twin-v3 mechanic is coded.** Committed to git so the timestamp is
verifiable. Nothing here may change after the first gate run; deviations are reported as
deviations. Design context: dev doc §3.9; decision record: working notes §2.18.

## 0. Purpose

Twin-v2 gave the agents one consequential decision (repair) and nothing to coordinate; learned
policies tied the rule-based baseline (`results/full_budget_2M/`, `results/robustness/`,
`results/reward_ablations/`). Twin-v3 adds hospitals, substations and interdependent sector
decisions. **Before any policy is trained**, the gates below must show that each decision is
genuine: that it measurably changes system outcomes, and that coordination and foresight have
room to help. The gates use heuristics only, never learned policies.

## 1. Build order (mechanics)

Each mechanic is added behind the v3 switch, with tests. Twin-v2 must still reproduce
bit-identically after every step (`scripts/check_reproduction.py`).

0. **Death metric first:** metric-v2 (`COUNT_UNCOLLECTED_CASUALTY_DEATHS`, done in `76a7000`)
   is **on** in v3. Patients in transit on a transfer do not die in transit; they join the
   destination queue on arrival and are subject to its deadline (disclosed simplification).
1. **Destination choice:** casualties and transfers go to the hospital transport selects.
2. **Hospital status:** accepting / diverting per hospital; surge capacity (a temporary bed
   increase with a stated cost).
3. **Transfer requests:** health issues requests (from, count, urgency); transport executes them.
4. **Shared fleet:** street calls and transfers compete for the same 6 ambulances.
5. **Repair-crew movement:** the crew travels on the flooded road network; an unreachable site
   cannot be repaired.
6. **Observations:** finalised last, after the mechanics are stable.

## 2. Strong rule (RB-S)

"Rule-based v2" extended to the v3 decisions, reactive only (current state, no forecasting). It is
never weakened to create headroom (dev doc §3.9 headroom principle).
- **Repair:** most dependants first, then most physically damaged; reachable sites only.
- **Dispatch:** every idle ambulance goes to a distinct reachable job, oldest first,
  urgency-weighted between calls and transfers.
- **Destination:** the nearest reachable hospital that is accepting and has free beds.
- **Transfers:** requested when a hospital's power or water buffer will run out, or its queue
  exceeds its degraded capacity.
- **Divert/surge:** set by queue thresholds.

Thresholds are tuned on the **train** split only. RB-S is then **frozen (git tag `rbs-v3`)
before the noise calibration or any gate runs.**

## 3. Scenario suite `flood_suite_v3`

20 train / 10 validation / 10 test, with disjoint seed ranges, a leakage check and SHA-256
manifests, as for `flood_suite_v2`.

**Strata, all physical and computed before any policy runs:**
- **Flood-centre sector:** which hospital's Voronoi sector (within the ward) contains the
  footprint centre. 3 strata.
- **Severity band:** mild / moderate / severe.

**Trade-off features.** These are computed from the flood field and the fragility curves alone,
never from a policy's performance:
- **T1 (alternative-access closure):** the access roads of at least one hospital that is *not*
  expected to fail are blocked within 6 h of onset.
- **T2 (multi-facility threat):** at least 2 facilities (hospitals, substations, pumps) have peak
  failure probability > 0.5 from the fragility curves at their peak depth.
- **T3 (high call volume):** expected emergency calls above the suite median.

**Requirement:** each of T1–T3 must occur in **≥ 25% of train scenarios**. If the stratified
draw falls short, the generator oversamples the relevant *physical* region (e.g. footprint
centres between two hospitals) until it holds. Scenarios are never selected or filtered by any
policy outcome. The achieved frequencies of T1–T3 are reported per split.

## 4. Statistics (all gates)

- **Split:** **train** only. Validation is kept for model selection and test for the final result;
  neither is touched by gates or redesign.
- **Unit of analysis:** the **scenario** (n = 20).
- **Seeds:** each scenario is run with evaluation seeds k = 0, 1, 2 (`scenario.seed + k × 100000`)
  and the **mean over the 3 seeds** is the scenario's value. The 60 episodes are never treated as
  independent.
- **Common random numbers:** every condition in a gate uses the same scenarios and the same seeds.
- **Paired difference:** condition X minus condition Y per scenario.
- **Confidence interval:** 95% bootstrap CI over scenarios, 10,000 resamples, generator seed 0.
- **Multiplicity:** where a gate has two outcome metrics, a Holm correction is applied across the
  two (family-wise α = 0.05).

## 5. Margins: derived from measured noise, fixed before any gate runs

**Noise calibration** (after RB-S is frozen, before any gate):
- Run RB-S on train with two disjoint seed groups: G1 = k 0–2 and G2 = k 3–5.
- For each metric, form the per-scenario difference G1 − G2. This is the difference between two
  runs of the *same* policy, i.e. the null.
- Its bootstrap distribution of the mean gives `noise(metric)` = the 97.5th percentile of
  |mean difference|.
- This is conservative: the gates themselves use common random numbers, which have less noise.

**Margin per metric** = max(2 × noise(metric), practical minimum), where the practical minimum is:

| Metric | Practical minimum |
|---|---|
| Deaths (metric-v2) | 5% of RB-S mean |
| Unmet patient-hours | 5% of RB-S mean |
| Casualty time-to-admission (call → bed) | 5% of RB-S mean |
| Jobs completed (calls + transfers) | 5% of RB-S mean |
| Mean critical-facility functional level | 0.02 (absolute) |

The computed margins are committed to `results/twin_v3_gates/margins.json` before any gate runs.

## 6. Gates (all on train, heuristics only)

**Pass rule for A–E:**
1. The gate's **mechanism metric** passes: the lower bound of the 95% bootstrap CI of the paired
   difference (in the favourable direction) is **> its margin**, **and**
2. the gate's **outcome check** moves in the same direction with its 95% CI excluding 0
   (Holm-corrected where there are two outcome metrics; one passing is enough).

Condition pairs are **pre-named** below, never chosen after seeing results. For B–D the question
is "does this decision matter?", so the pair is two sensible alternatives and either direction
passes. The comparison is |difference|, and the CI must exclude ±margin.

| Gate | Condition pair | Mechanism metric | Outcome check |
|---|---|---|---|
| **A-health** | RB-S vs RB-S with health fixed (no transfer requests, always accepting, no surge) | Unmet patient-hours | Deaths |
| **A-power** | RB-S vs RB-S with power fixed (no repair, no shedding) | Critical-facility function | Deaths or unmet patient-hours |
| **A-transport** | RB-S vs RB-S with transport fixed (FIFO, oldest job first, nearest ambulance, casualty to home hospital, transfers to nearest other hospital) | Casualty time-to-admission | Deaths or unmet patient-hours |
| **B destination** | Adaptive destination (RB-S) vs fixed "nearest functioning hospital" | Casualty time-to-admission | Deaths |
| **C fleet** | Calls-first vs transfers-first allocation (rest = RB-S) | Jobs completed | Deaths or unmet patient-hours |
| **D repair order** | Most-dependants-first (RB-S) vs nearest-reachable-first, crew movement on | Critical-facility function | Unmet patient-hours |
| **E coordination** | Coordinated heuristic vs the same rules without sharing | Deaths | Unmet patient-hours or critical function |

**Gate E conditions:**
- **Coordinated heuristic:** transport sees divert status and free beds; power prioritises the
  supply chain of the hospital receiving the most diversions; health surges the hospital transport
  is sending to.
- **Same rules without sharing:** transport ignores beds and divert status; power ranks by
  dependants only; health doesn't know about incoming patients.
- **Direction is fixed:** coordinated must be *better*.

**Gate F: headroom adequacy threshold (a design criterion, not a universal constant).**
- **Ceiling:** a lookahead policy. At each decision it evaluates the candidate joint actions formed
  from the gate variants above (RB-S plus the B–E alternatives) with `simulate()` (5 rollouts,
  2 h horizon), and picks the one with the lowest deaths + unmet patient-hours.
- **Pass:**
  - the ceiling's mean deaths are **≥ 15% below RB-S's**, **and**
  - the CI lower bound of the improvement is > 0.
- **Caveat:** the ceiling uses the twin's own model, so it is an optimistic upper bound. The gap
  measures room to improve; it does not predict the learned gain.

The implementations of every heuristic variant above are committed (with tests) **before** the
first gate run. Their definitions here are binding.

## 7. Decisions and stopping rule

- **All of A–F pass:**
  - proceed to reward normalisers (RB-S on train, metric-v2 on);
  - Experiment A on v3 (test);
  - trainer fixes;
  - a 300k screen on validation;
  - the pre-registered 2M run;
  - test once.
- **Any gate fails:** redesign the twin (not RB-S, and never by weakening a baseline). Document the
  change, re-freeze, and re-run the calibration and **all** gates. **At most 2 redesign rounds.**
- **Still failing after 2 rounds:**
  - report which decisions are not genuine and why;
  - drop MARL claims for those sectors;
  - move the thesis emphasis to single-agent + counterfactual planning (RQ2).

  This is reported as a result, not hidden.

## 8. Reported regardless of outcome

For every gate: both conditions' means, the paired difference with its CI, margins, pass/fail,
and results per stratum (flood-centre sector × severity band). For the suite: T1–T3 frequencies
per split.
