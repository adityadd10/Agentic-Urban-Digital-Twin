# Pre-registered reward ablations: unanswered calls (C) and cascade term (D)

**Written 2026-09-28, before any run it governs; committed to git so the timestamp is
verifiable.** Nothing here may be changed after the first ablation run starts; deviations are
reported as deviations.

## 0. Why, and a disclosure

Two reward-design artefacts are suspected of steering learned policies away from good emergency
practice:

- **Dispatch artefact.** The ambulance term penalises response time only when a casualty is
  picked up; unanswered calls cost nothing, and a delivered casualty adds hospital-queue load.
  Not dispatching is reward-optimal. This was identified **after** seeing the 2M results on the
  **test** split (requests completed fell with more training: PPO 2.5 → 0.3, MAPPO 7.5 → 4.5).
- **Cascade artefact.** Repairing a flooded facility while its supplier is down is counted as a
  cascade (≈ 11 death-equivalents of reward). Recorded before the 2M run
  (`2026-09-28_cascade_reward_artefact.md`), after Experiment A's **test** results showed it.

Both hypotheses came from test-split observations. **Every decision below uses the validation
split only; the test split is used once per condition, afterwards, for reporting.**

## 1. Conditions (one change each; everything else identical)

| | A (reference) | C (unanswered calls cost) | D (no cascade term) |
|---|---|---|---|
| Ambulance term | Σ response times of calls picked up this tick | **waiting hours accrued this tick by every pending call** (`pending_requests_count × dt`) | as A |
| Ambulance normaliser | 0.1820 (fitted) | refitted for the new term by the §5.4 rule (rule-based, train split, mean per tick) | as A |
| Cascade coefficient | 5 | 5 | **0** |
| Everything else | `configs/reward.yaml` | identical | identical |

A's total waiting cost for an *answered* call is unchanged in spirit (it accrues while the call
waits instead of being booked at pickup); the change is that *unanswered* calls now also accrue
cost. The death proxy, all evaluation metrics and the twin are unchanged.

**Training (C and D):** MAPPO, seeds 0–4, 1,200 iterations × 256 steps (307,200 steps), same
network and hyperparameters, train split of `flood_suite_v2`, twin-v2, final checkpoint. **A is
not retrained:** the `baseline-300k` MAPPO models are condition A (the §6 reproduction checks of
the 2026-09-27 protocol passed, and the code changes since then are verified training-identical
for A).

## 2. Evaluation

Validation split, 10 scenarios × 3 evaluation seeds, default goal weights, deterministic
actions; per-scenario means over 5 training seeds × 3 evaluation seeds; paired against A by
scenario; 95% bootstrap CI (10,000 resamples) and two-sided Wilcoxon, α = 0.05. Every condition is
scored with **A's reward definition** for `episode_reward`. Primary metrics: death proxy, unmet
patient-hours.

Mechanism metrics:
- **C:** requests completed (and requests pending at episode end).
- **D:** *restoration*: the fraction of facilities that were flood-failed (intrinsic at its
  failed residual) at some tick and later reached intrinsic ≥ 0.5 by the end of the episode.

## 3. Predictions (mechanism, not performance)

- **C:** dispatch recovers: requests completed higher than A. Effect on patient outcomes not
  predicted (more dispatches also add hospital-queue load).
- **D:** more flood-failed facilities are restored while their supplier is down. Effect on
  patient outcomes not predicted.

## 4. Decision rules (fixed now, validation only)

A condition X ∈ {C, D} is **adopted** for future training iff, on the validation split:
- (a) its mechanism metric (C: requests completed; D: restoration) is higher than A's with
  Wilcoxon p < 0.05; **and**
- (b) it is non-inferior to A on **both** primary metrics: the upper bound of the 95% CI of
  (X − A) is ≤ +10% of A's mean.

If (a) fails, the artefact is not what drives the behaviour. If (a) holds but (b) fails, the
artefact is real but removing it costs patient outcomes, and it is reported as such.

**If both C and D are adopted,** a combined condition C+D is trained and evaluated on validation
under the same rules before it is used for any further training (effects are not assumed to
add). Each adopted or rejected condition is then evaluated on **test once** and reported
whatever it shows.

## 5. Implementation constraints

The reward options are configuration fields whose defaults reproduce A exactly. Before any C/D
run, `scripts/check_reproduction.py` must show MAPPO seeds 0–4 still reproduce `baseline-300k`
bit-identically under the default configuration.
