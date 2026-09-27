# Reported results (tracked in git; raw run folders under runs/ are not)

All on twin-v2 / `udt_multi_env_v2`, frozen suite `flood_suite_v2`, **test split**
(10 scenarios × 3 evaluation seeds, per-scenario means, 95% bootstrap CIs, two-sided
Wilcoxon signed-rank, α = 0.05, no multiple-comparison correction).

| Folder | What | Code commit |
|---|---|---|
| `experiment_a_flood_suite_v2/` | Experiment A: rule-based vs do-nothing | 847297e |
| `INTERIM_experiment_bc_half_trained/` | **Interim, half-trained**: PPO (3 seeds @ 150k steps), MAPPO (5 seeds @ iteration 600 ≈ 154k steps) vs rule-based | 847297e (training), f6c207d (eval) |
| `experiment_bc_flood_suite_v2/` | **Final** Experiments B/C: PPO (3 seeds, 300k steps), MAPPO (5 seeds, 1,200 iterations ≈ 307k steps) vs rule-based | 847297e (training), a55a3b9 (eval) |
| `baseline-300k/` | Frozen reference: manifest (SHA-256, true steps) and read-only copies of the 8 baseline models (tag `baseline-300k`) | 52840c7 |
| `protocol/` | Pre-declared protocols, committed before the runs they govern | 4052ed9 |
| `ablation_energy_floor/` | Energy-floor ablation (MAPPO, 300k): A vs B on val → **decision USE A**; B on test once. See `OUTCOME.md` | c775ec6 |

Read with these caveats:
- Training budget is ≈ 6–15% of dev doc §5.5's 2–5 M steps: these are under-trained policies.
- `patient_deaths` is the queue-wait proxy (dev doc §3.8 item 8), not clinical mortality.
- The cascade metric must be read with functional levels (dev doc §3.5 caveat).
- The energy-normaliser floor (dev doc §5.4) is a disclosed deviation that shapes how much
  learned policies shed load.
