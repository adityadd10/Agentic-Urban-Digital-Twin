## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 47.07 [30.42, 65.74] | 43.57 | +3.50 [-7.74, +18.35] | 0.625 | no |
| unmet_patient_hours | 294.97 [215.36, 372.14] | 272.79 | +22.18 [-22.16, +75.59] | 0.846 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.66 [0.61, 0.72] | 0.70 | -0.04 [-0.08, -0.01] | 0.105 | no |
| mean_critical_functional_level | 0.55 [0.44, 0.67] | 0.59 | -0.04 [-0.08, -0.01] | 0.049 | no |
| unserved_energy_mwh | 69.99 [60.57, 78.52] | 2.66 | +67.33 [+58.15, +76.44] | 0.002 | no |
| mean_ambulance_response_delay_hours | 6.00 [4.46, 7.46] | 4.30 | +1.70 [+0.73, +2.65] | 0.014 | no |
| requests_completed | 6.17 [4.23, 8.17] | 9.20 | -3.03 [-3.95, -2.20] | 0.002 | no |
| episode_reward | -4701.20 [-5881.61, -3424.81] | -4331.98 | -369.22 [-1021.78, +146.97] | 0.557 | no |
