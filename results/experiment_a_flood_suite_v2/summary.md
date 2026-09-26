# Experiment A — rule-based vs do-nothing (test split, flood_suite_v2)

n = 10 scenarios × 3 eval seeds; per-scenario means; 95% bootstrap CIs; two-sided Wilcoxon signed-rank, α = 0.05.

| Metric | Do-nothing | Rule-based | Difference (rule − nothing) | p | Rule better & significant |
|---|---|---|---|---|---|
| patient_deaths | 104.87 [76.70, 127.53] | 43.57 [29.20, 58.33] | -61.30 [-78.57, -42.33] | 0.004 | yes |
| unmet_patient_hours | 479.25 [353.89, 577.78] | 272.79 [195.27, 344.67] | -206.46 [-262.94, -142.92] | 0.004 | yes |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 [2.30, 4.00] | +0.20 [+0.00, +0.50] | 0.500 | no |
| mean_hospital_functional_level | 0.44 [0.35, 0.57] | 0.70 [0.63, 0.77] | +0.26 [+0.18, +0.33] | 0.004 | yes |
| mean_critical_functional_level | 0.32 [0.16, 0.53] | 0.59 [0.48, 0.72] | +0.27 [+0.18, +0.36] | 0.004 | yes |
| unserved_energy_mwh | 0.00 [0.00, 0.00] | 2.66 [0.00, 6.40] | +2.66 [+0.00, +6.40] | 0.500 | no |
| mean_ambulance_response_delay_hours | — | — | — | — | n/a |
| requests_completed | 0.00 [0.00, 0.00] | 9.20 [7.17, 11.37] | +9.20 [+7.17, +11.40] | 0.002 | yes |
| episode_reward | -7080.41 [-8645.11, -5113.62] | -4331.98 [-5426.99, -3115.35] | +2748.43 [+1793.56, +3610.38] | 0.004 | yes |
