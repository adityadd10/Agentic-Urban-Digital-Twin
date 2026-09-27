# Pre-declared protocol: energy-floor ablation and full-budget training

**Written 2026-09-27, before any run it governs. Committed to git so the timestamp is
verifiable.** Nothing in this file may be changed after the first ablation run starts. Any
deviation is reported as a deviation, not edited in.

Reference result: `baseline-300k` (git tag → `52840c7`; models in
`results/baseline-300k/`). Twin `twin-v2`, env `udt_multi_env_v2`, suite `flood_suite_v2`.

## 1. Metrics

**Primary** (the ones decisions and claims rest on):
1. patient death proxy (`patient_deaths`: queue-wait > 4 h, not clinical mortality)
2. unmet patient-hours (`unmet_patient_hours`)

**Secondary** (reported, never used alone for a claim):
hospital functional level, critical-facility functional level, unserved energy (MWh),
ambulance response delay (h), requests completed, episode reward, cascade count. The cascade
count is always read next to functional levels (dev doc §3.5 caveat).

Statistics for every comparison: unit = scenario (per-scenario mean over training seeds ×
evaluation seeds), 95% bootstrap CI (10,000 resamples) of the paired mean difference,
two-sided Wilcoxon signed-rank, α = 0.05, no multiple-comparison correction. Secondary
p-values are descriptive.

## 2. Checkpoint rule

**Report the final checkpoint at the declared training budget.** No checkpoint selection, on
val or anywhere else. Intermediate checkpoints may be saved for learning curves only.

## 3. Test-set rule

The frozen test split (`flood_suite_v2/test`) is **never** used for checkpoint selection,
hyperparameter tuning, reward modification, or ablation decisions. It is not replaced or
extended because results look unfavourable. New scenarios may be generated for additional
experiments, but never substitute for it.

**Disclosure:** the energy-floor hypothesis below was formed after seeing the `baseline-300k`
results **on the test split** (MAPPO unserved energy 39.8 vs rule-based 2.7 MWh). From here on
every decision uses the validation split only.

## 4. Energy-floor ablation (runs at 300k, MAPPO only)

**Hypothesis being tested:** the energy-floor formulation makes load shedding artificially
cheap, which encourages MAPPO to shed load unnecessarily.

**Conditions.** Everything identical except one number:

| | A (current) | B (no floor) |
|---|---|---|
| `unserved_energy_mwh` normaliser | 0.21998209947402164 (floor) | **0.002575088038625174** (spec-literal §5.4: rule-based mean per tick on train, `raw_mean_abs_per_tick` in `configs/reward.yaml`) |
| other four normalisers | `configs/reward.yaml` (847297e) | identical |
| effect | — | shedding penalty ×85.4 |

Same for both: twin-v2, `flood_suite_v2` train split, MAPPO, seeds 0–4, 1,200 iterations ×
256-step rollouts (307,200 steps), same network and hyperparameters, final checkpoint.
**A is not retrained**: the `baseline-300k` MAPPO models are condition A, *provided* the
reproduction check (§6) passes. If it fails, A is retrained with the same code as B.

**MAPPO only:** by 300k, PPO had already stopped blanket shedding (0.76 MWh on test vs
rule-based 2.66), so the hypothesis concerns MAPPO.

**Prediction:** removing the energy floor will substantially reduce unnecessary load shedding,
because the reward then imposes the full shedding penalty. It may or may not improve patient
outcomes. (Prediction of mechanism, not of performance.)

**Evaluation:** validation split, 10 scenarios × 3 evaluation seeds
(`scenario.seed + k × 100000`, k = 0, 1, 2), default goal weights (1, 1, 1, 1), deterministic
actions. A vs B paired by scenario. **Both are scored with A's reward definition** (A's
normalisers) for the `episode_reward` metric; B's native reward may be shown separately and
labelled as such. Rule-based is also run on val as a reference.

## 5. Decision rule for full-budget training (fixed now)

Use **B** for full-budget training **if and only if**, on the validation split:
- (a) B's mean unserved energy is ≥ 50% lower than A's, **and** the paired difference is
  significant (Wilcoxon p < 0.05); **and**
- (b) B is non-inferior on **both** primary metrics: the upper bound of the 95% CI of (B − A)
  is ≤ +10% of A's mean, for the death proxy and for unmet patient-hours.

Otherwise use **A**. If (a) fails, the energy floor is not the primary cause of the shedding,
and the policy/action formulation is investigated next instead.

B is evaluated on the **test** split once, after this decision is made, and reported whatever
it shows.

## 6. Preconditions (checked before the ablation run)

1. **Reproduction check, current code.** With the code as of this commit, MAPPO seeds 0–4
   reproduce the first 20 iterations of the `baseline-300k` training runs **bit-identically**
   (every logged loss in `training_history.json`).
2. **Checkpoint/resume.** MAPPO checkpoints save actor and critic parameters, optimiser
   states, iteration and step counters, torch/numpy RNG states, env episode counter and
   in-progress environment state, and version metadata. **Acceptance:** N iterations
   uninterrupted must equal N/2 → checkpoint → resume → N/2, bit-identically (losses and final
   weights).
3. **Reproduction check, after the resume code** (same as 1). If it fails, the ablation does
   not use the baseline as A (see §4).
4. **Normalisation state:** none to save. Reward normalisers are a frozen config file, and
   neither algorithm uses running observation normalisation.

## 7. Full-budget training (after the §5 decision)

- **Budget: 2,000,000 steps per policy.** MAPPO: 7,813 iterations × 256 = 2,000,128 steps,
  seeds 0–4. PPO: `--total-timesteps 2000000` (SB3 rounds to whole 2,048-step rollouts), seeds
  0–2.
- Configuration identical to `baseline-300k` except the budget, and the reward if §5 selects B.
- CPU only (the simulator is 97.6% of the step time; measured 2026-09-27).
- Final checkpoint reported; evaluated on test once, against rule-based and against
  `baseline-300k`, with this file's metrics and statistics.
- PPO resume is not bit-exact (SB3 does not save its rollout buffer or env state). Any
  interrupted PPO run is disclosed.
