# Reported results (tracked in git; raw run folders under runs/ are not)

| Folder | What | Twin / suite | Code commit |
|---|---|---|---|
| `experiment_a_flood_suite_v2/` | Experiment A: rule-based vs do-nothing, 10 test scenarios × 3 eval seeds, bootstrap CIs + Wilcoxon | twin-v2 / flood_suite_v2 (test split) | 847297e |

`patient_deaths` is the queue-wait proxy (dev doc §3.8 item 8), not clinical mortality.
The cascade metric must be read with functional levels (dev doc §3.5 caveat).
| `INTERIM_experiment_bc_half_trained/` | **Interim, half-trained**: PPO (3 seeds @150k steps) and MAPPO (5 seeds @ iter 600, ~154k steps) vs rule-based, same 10 test scenarios × 3 eval seeds | twin-v2 / flood_suite_v2 (test) | 847297e (training), f6c207d (eval) |
