# Reported results (tracked in git; raw run folders under runs/ are not)

| Folder | What | Twin / suite | Code commit |
|---|---|---|---|
| `experiment_a_flood_suite_v2/` | Experiment A: rule-based vs do-nothing, 10 test scenarios × 3 eval seeds, bootstrap CIs + Wilcoxon | twin-v2 / flood_suite_v2 (test split) | 847297e |

`patient_deaths` is the queue-wait proxy (dev doc §3.8 item 8), not clinical mortality.
The cascade metric must be read with functional levels (dev doc §3.5 caveat).
