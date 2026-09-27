# Experiment A — rule-based vs do-nothing (val split, flood_suite_v2)

n = 10 scenarios × 3 eval seeds; per-scenario means; 95% bootstrap CIs; two-sided Wilcoxon signed-rank, α = 0.05.

| Metric | Do-nothing | Rule-based | Difference (rule − nothing) | p | Rule better & significant |
|---|---|---|---|---|---|
| patient_deaths | 87.03 [54.83, 114.83] | 57.03 [30.57, 81.97] | -30.00 [-44.40, -16.63] | 0.008 | yes |
| unmet_patient_hours | 409.98 [263.96, 535.16] | 310.26 [191.52, 426.03] | -99.72 [-156.85, -49.34] | 0.010 | yes |
| cascading_failure_count | 2.00 [1.00, 3.00] | 2.50 [1.30, 3.60] | +0.50 [+0.10, +0.90] | 0.125 | no |
| mean_hospital_functional_level | 0.51 [0.35, 0.69] | 0.67 [0.54, 0.80] | +0.15 [+0.09, +0.21] | 0.008 | yes |
| mean_critical_functional_level | 0.42 [0.21, 0.65] | 0.59 [0.42, 0.76] | +0.16 [+0.09, +0.23] | 0.008 | yes |
| unserved_energy_mwh | 0.00 [0.00, 0.00] | 1.67 [0.00, 5.00] | +1.67 [+0.00, +5.00] | 1.000 | no |
| mean_ambulance_response_delay_hours | — | — | — | — | n/a |
| requests_completed | 0.00 [0.00, 0.00] | 13.73 [9.87, 17.83] | +13.73 [+9.90, +17.80] | 0.002 | yes |
| episode_reward | -5582.04 [-7446.09, -3597.77] | -4718.97 [-6631.82, -2692.68] | +863.06 [+5.43, +1819.01] | 0.193 | no |
