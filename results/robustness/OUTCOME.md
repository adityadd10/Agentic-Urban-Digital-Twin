# Robustness study — outcome (2026-09-28)

Governed by `results/protocol/2026-09-28_robustness_study.md` (pass rules fixed before any 2M
result). That file is unchanged; this file records what happened. Full tables: `summary.md`,
`results.json`. Raw episodes: `runs/robustness_study/<condition>.json`.

**Run:** 26 conditions (nominal + 22 all-policy perturbations from 12 families + 3 rule-only),
test split, 10 scenarios × 3 evaluation seeds, policies do-nothing, rule-based, PPO 2M (3 seeds),
MAPPO 2M (5 seeds), MAPPO 300k (5 seeds). Nominal condition reproduced the reported 2M and
Experiment A results bit-identically before the run (commit `ad286b1`). All 26 conditions exited
cleanly. No deviations from the protocol.

## 1. Pre-registered verdicts

| Algorithm / primary metric | Nominal category | Unchanged in | Robust (≥ 18/22) | Rule 2 flag |
|---|---|---|---|---|
| PPO 2M / death proxy | no difference | 22/22 | yes | n/a (no advantage) |
| PPO 2M / unmet patient-hours | no difference | 22/22 | yes | n/a |
| MAPPO 2M / death proxy | no difference | 22/22 | yes | n/a |
| MAPPO 2M / unmet patient-hours | no difference | 22/22 | yes | n/a |

Rule 3 (descriptive): Spearman ρ of the four-policy ordering by mean death proxy vs nominal,
median 0.80, min 0.80 (1.0 where nothing swaps). The only swaps are between rule-based and
MAPPO, whose nominal means differ by 0.14 deaths, and between rule-based/MAPPO and PPO when
they are close.

**Conclusion:** the 2M result — *no learned policy is significantly better or worse than
rule-based on the primary metrics* — is robust to every perturbed simulator assumption tested.
There is no learned advantage for Rule 2 to test. The absolute death-proxy level moves a lot
with the assumptions (rule-based 22.7–67.6 across conditions), so absolute numbers are
model-conditional, while the comparison between policies is not.

Descriptive observations (not decision rules): repair rate matters most for rule-based
(R10_low 67.6 vs R10_high 22.7), and at R10_low all learned policies have *lower* mean death
proxy than rule-based (MAPPO 55.5, PPO 60.6), not significant; R14 travel-time noise has a
genuine but tiny effect (22/30 rule-based episodes differ from nominal; means unchanged to one
decimal).

## 2. Post-hoc findings (found after seeing the results — diagnoses, not pre-registered)

Two rule-only conditions gave results that looked implausible and were traced to their cause
by replaying single episodes (`flood_test_00`, `flood_test_04`, seed k = 0).

**(a) R12 (second substation) makes rule-based *worse* (43.6 → 67.4) while do-nothing improves
(104.9 → 78.7).** Cause: the rule-based repair rule (`_pick_repair_target`) chooses the facility
with the lowest *functional* level, not the lowest *intrinsic* (physical) level. In
`flood_test_00` under R12, the new substation floods and fails, which starves water facility
`W_synth_0` to F = 0.00 while it is physically intact (intrinsic 1.00). That ties with or
undercuts S0 (F = 0.05), so the single crew is sent to `W_synth_0` 51 times, where a repair can
do nothing, and S0 is never restored; the whole ward stays dark to the end. In the nominal run
the same rule picks S0 only because W's F (0.06) happens to exceed S0's (0.05). **This is a
latent flaw of the rule-based baseline, not evidence about redundancy.** R12 is therefore
uninformative about a second substation until the rule is fixed.

**(b) R7 (1 vs 3 ambulances per hospital) barely changes rule-based outcomes.** Cause: the
dispatch rule sends the nearest idle ambulance to the *oldest* pending request only, and does
nothing if that request is unreachable. In `flood_test_04` (nominal), at **80 of 91** decision
points there were pending calls **and** idle ambulances but no dispatch; there was never a
decision point with no idle ambulance (identical under R7_high). The fleet is not the
bottleneck; head-of-line blocking is. This corrects an earlier note that called the ~21 pending
calls at 24 h "a real capacity result": it is a rule-design result.

**Consequences.**
- Experiment A and all comparisons stand as reported: they compare against the rule-based agent
  exactly as specified (dev doc §5.6) and frozen. But the baseline is weaker than it needs to
  be in two identifiable ways, which matters for how "learned ties rule-based" is read: a
  better heuristic would likely be a *harder* baseline.
- Both are Stage 3 (baseline) review items. Any change to the rule is a new, versioned
  baseline with Experiment A re-run; the current one is not edited in place.
