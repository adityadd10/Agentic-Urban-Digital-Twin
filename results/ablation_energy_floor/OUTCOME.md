# Energy-floor ablation — outcome (2026-09-27)

Governed by `results/protocol/2026-09-27_energy_ablation_and_full_budget.md` (commit `4052ed9`,
written before the run). That file is unchanged; this file records what happened.

**Run:** condition B = MAPPO seeds 0–4, 307,200 steps each, `configs/reward_no_energy_floor.yaml`,
code `c775ec6`. Condition A = the `baseline-300k` MAPPO models (reused, because the §6
reproduction checks passed before and after the resume code). No deviations from the protocol.

## Decision (validation split, protocol §5): **USE A**

| Metric | A (floor) | B (no floor) | B − A [95% CI] | p | Rule-based |
|---|---|---|---|---|---|
| **deaths proxy** (primary) | 61.46 | 68.41 | +6.95 [+1.40, +14.03] | 0.055 | 57.03 |
| **unmet patient-hours** (primary) | 328.60 | 352.76 | +24.15 [−3.56, +58.81] | 0.426 | 310.26 |
| unserved energy (MWh) | 39.43 | 0.00 | −39.43 [−48.72, −28.39] | 0.002 | 1.67 |
| hospital functional level | 0.67 | 0.65 | −0.02 [−0.04, −0.00] | 0.055 | 0.67 |
| requests completed | 8.59 | 5.47 | −3.11 [−4.34, −1.93] | 0.004 | 13.73 |
| ambulance delay (h) | 7.48 | 6.76 | −0.72 [−1.31, −0.18] | 0.074 | 4.73 |
| episode reward (A's definition) | −4775 | −4875 | −100 [−514, +269] | 0.922 | −4719 |

- (a) energy reduced ≥ 50% with p < 0.05: **met** (100% reduction, p = 0.002).
- (b) non-inferior on both primary metrics (upper CI ≤ +10% of A): **not met.** Deaths: upper
  bound +14.03 > margin +6.15. Unmet patient-hours: +58.81 > +32.86.
- → Full-budget training uses **reward A** (protocol §5).

## Test split (reported once, after the decision, as protocol §5 requires)

B vs rule-based: deaths proxy 48.46 vs 43.57 (n.s.); unmet patient-hours 285.5 vs 272.8 (n.s.);
unserved energy 0.00 vs 2.66; hospital function 0.64 vs 0.70 (p = 0.027); requests completed 4.30
vs 9.20 (p = 0.002). Next to A on the same test set (`baseline-300k`: deaths 43.39, energy 39.77
MWh), this repeats the validation pattern: B sheds nothing but has more deaths.

## Interpretation

- **The predicted mechanism is confirmed.** Removing the floor removes load shedding completely,
  on val and on test. The energy floor is what made MAPPO shed.
- **The shedding was not purely unnecessary.** Without it, primary outcomes get worse (on val, the
  deaths CI excludes zero). Plausibly, some of A's shedding was protecting hospitals when the
  substation was degraded (after the dev doc §3.8 supply-allocation change, shedding frees supply for
  critical consumers). **Untested hypothesis:** the ideal is *targeted* shedding, which neither reward
  produced at 300k steps.
- Rule-based remains best on the primary metrics in both splits.
- B also completes fewer ambulance requests. Unexplained; not investigated.
