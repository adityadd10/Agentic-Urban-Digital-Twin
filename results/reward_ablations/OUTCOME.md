# Reward ablations C and D — outcome (2026-09-28)

Governed by `results/protocol/2026-09-28_reward_ablations.md` (commit `ec99136`, written before
any run). That file is unchanged; this file records what happened.

**Run:** MAPPO seeds 0–4 per condition, 1,200 iterations × 256 = 307,200 steps each, final
checkpoint, trained in parallel from a read-only worktree at tag `reward-ablation-CD-start`
(`ddf6041`; code identical to `ad286b1`, where the §5 reproduction check passed). All 10 runs
completed 1,200 iterations with no errors. C = `configs/reward_C_pending_calls.yaml`,
D = `configs/reward_D_no_cascade.yaml`. A = the `baseline-300k` MAPPO models. Models and
checksums: `manifest.json`, `models/`. **No deviations from the protocol.**

## 1. Decisions (validation split, protocol §4)

| | Mechanism metric: A → X (p) | (a) met? | Deaths upper CI vs margin | Unmet p-h upper CI vs margin | (b) met? | **Decision** |
|---|---|---|---|---|---|---|
| **C** unanswered calls cost | requests completed 8.59 → **7.49** (p = 0.035, *lower*) | no | +4.63 ≤ +6.15 | +5.08 ≤ +32.86 | yes | **REJECT C** |
| **D** no cascade term | restoration 0.35 → 0.41 (p = 0.44) | no | +6.79 > +6.15 | +15.27 ≤ +32.86 | no | **REJECT D** |

Other validation effects: C sheds much more load (39.4 → 69.3 MWh, p = 0.002); D slightly lowers
hospital function (0.67 → 0.66, p = 0.039) and completes fewer requests (8.59 → 7.05,
p = 0.004). Rule-based on validation: deaths 57.03, requests completed 13.73. Full tables:
`val_C_vs_A_summary.md`, `val_D_vs_A_summary.md`.

## 2. Test split (reported once, after the decision, as protocol §4 requires)

| Test | Rule-based | A (`baseline-300k`) | C | D |
|---|---|---|---|---|
| Death proxy | 43.57 | 43.39 | 47.07 (n.s. vs rule) | 44.59 (n.s.) |
| Unmet patient-hours | 272.79 | 278.35 | 294.97 (n.s.) | 281.45 (n.s.) |
| Requests completed | 9.20 | 7.48 | 6.17 (worse, p = 0.002) | 5.85 (worse, p = 0.002) |
| Unserved energy (MWh) | 2.66 | 39.77 | 69.99 | 40.87 |
| Hospital functional level | 0.70 | 0.66 | 0.66 | 0.65 (worse, p = 0.027) |

The test split repeats the validation pattern: neither change improves its target behaviour, and
neither beats rule-based on anything. Full tables: `test_C_vs_rule_summary.md`,
`test_D_vs_rule_summary.md`.

## 3. Interpretation

- **C: the dispatch hypothesis is not supported.** It predicted that dispatch would recover once
  unanswered calls cost something. Dispatch fell instead (val 8.59 → 7.49; test 7.48 → 6.17).
  By protocol §4, "the artefact is not what drives the behaviour". The post-hoc explanation for
  the 2M dispatch collapse (unanswered calls are free) is therefore **withdrawn** as the main
  cause. Candidate explanations, **untested**: (i) the transport agent's observation lacks what
  dispatch needs (queue ages, call locations relative to flooding: §2.16's POMDP finding);
  (ii) the new pending-call cost accrues even when calls are unreachable (the same flooding that
  causes rule-based head-of-line blocking, robustness OUTCOME §2b), so it is a large, weakly
  controllable penalty; the policy reacted mainly by shedding more load; (iii) the 300k budget.
- **D: removing the cascade term did not measurably increase restoration** (+0.06, n.s.) and
  cost a little on patient outcomes. The cascade term is not shown to discourage repairs at this
  budget.
- **Consequence for the overall story.** Of the three suspected reward artefacts, only the
  energy floor is shown to drive a learned behaviour (load shedding; 2026-09-27 ablation). The
  earlier summary "the reward, not the budget, is the bottleneck" is **too strong** and is
  withdrawn. The supported statement: *learned policies tie rule-based on patient outcomes
  (robust to 22/22 simulator perturbations); their weak ambulance dispatch is not explained by
  the reward's treatment of unanswered calls or by the cascade term.*
- Reward A remains the reward for further work.
