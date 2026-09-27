# Pre-registered robustness study: is the comparison an artefact of the simulator?

**Written 2026-09-28, while the 2M training (tag `train-2M-start`) is running and before any of
its results exist.** Committed to git so the timestamp is verifiable. Nothing here may be changed
after the first robustness evaluation runs; deviations are reported as deviations.

Parameter sources and categories: `docs/parameter_provenance.md` (R-ids refer to it).

## 1. Question

Do the conclusions of the policy comparison (rule-based vs PPO vs MAPPO) survive plausible
changes to the simulator's assumed (D) and synthetic (E) parameters? Or do they depend on
particular values, i.e. is a learned policy exploiting a simulator artefact (examiner Q9)?

## 2. Design

- **Evaluation only.** No retraining. Each perturbation is applied to the twin at evaluation
  time; policies are the ones already trained on the nominal twin.
- **Policies:** do-nothing, rule-based, PPO (the 3 × 2M final models), MAPPO (the 5 × 2M final
  models). Secondary: the `baseline-300k` MAPPO models.
- **Scenarios:** the frozen `flood_suite_v2` **test** split, 10 scenarios × 3 evaluation seeds,
  exactly as in the main evaluation. Reporting only: no decision about training, reward or
  models is taken from these results.
- **One factor at a time**, low and high level, all else nominal:

| R-id | Family | Low | High |
|---|---|---|---|
| R1 | dependency floors (power / water / access→hospital) | ×0.5 (0.15 / 0.25 / 0.35) | ×1.5, capped at 0.95 (0.45 / 0.75 / 0.95) |
| R2 | buffers (generator / water reserve) | ×0.5 (4 h / 3 h) | ×1.5 (12 h / 9 h) |
| R3 | demand shares (hospital power & water; power→water) | ×0.5 | ×2 |
| R4 | patient arrival rate | ×0.5 | ×1.5 |
| R5 | mean length of stay | 24 h | 72 h |
| R6 | death-proxy deadline | 2 h | 6 h |
| R8 | fragility depths (all curves) | ×0.75 (fail at shallower water) | ×1.25 |
| R9 | road blockage depth scale | 0.27 m (DISruptionMap, B) | 0.9 m |
| R10 | repair rate | ×0.5 | ×2 |
| R11 | substation loading (surge factor; base load share) | 0.25; 50% | 1.0; 80% |
| R13 | dependency aggregation | min over dependencies instead of product | — (single alternative) |
| R14 | travel-time noise (dev doc §3.6, not in nominal twin) | ±15% uniform per route, seeded | — (single level) |

That's **22 conditions** evaluable by all policies. Rule-based only (they change the observation
size, so trained policies can't run; descriptive, not part of the pass rule): **R7** fleet size
1 or 3 per hospital; **R12** a synthetic second substation (redundancy).

## 3. Metrics and statistics

Primary: death proxy and unmet patient-hours (as protocol 2026-09-27 §1). Secondary as there.
Per condition, for each learned algorithm vs rule-based: per-scenario means, paired difference,
95% bootstrap CI, two-sided Wilcoxon, α = 0.05. Every outcome goes into one of three categories:
**better** (p < 0.05, favours learned), **no difference**, **worse** (p < 0.05, favours rule-based).

## 4. Pass rules (fixed now)

Let C0 be the category from the nominal (unperturbed) 2M test evaluation, for each
(algorithm, primary metric).

1. **Robust conclusion:** C0 is unchanged in **≥ 18 of the 22 conditions (≥ 80%)** for that
   (algorithm, primary metric). Every condition that flips it is reported by name.
2. **Advantage-exploitation flag:** if C0 = *better* for a learned algorithm, and it becomes
   *no difference* or *worse* in conditions from **≥ 2 different families**, the advantage is
   declared **not robust to simulator assumptions**, whatever rule 1 says.
3. **Ordering stability (descriptive):** Spearman ρ between the nominal ranking of the four
   policies' mean death proxy and each condition's ranking; median ρ reported.

## 5. Implementation constraints

Perturbations are implemented as evaluation-time overrides in the main repo, never in the frozen
training worktree. Before use, with **no perturbation active**, the override mechanism must
reproduce the nominal evaluation bit-identically (same episodes, same metrics). Runs start only
after the 2M training has finished (so they don't compete for CPU).
