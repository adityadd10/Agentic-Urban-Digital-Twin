## PPO vs rule-based (test, 3 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 49.11 [31.01, 68.47] | 43.57 | +5.54 [-5.90, +19.80] | 0.770 | no |
| unmet_patient_hours | 283.62 [199.44, 369.97] | 272.79 | +10.83 [-37.50, +66.56] | 0.922 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.65 [0.60, 0.71] | 0.70 | -0.06 [-0.09, -0.02] | 0.027 | no |
| mean_critical_functional_level | 0.54 [0.43, 0.67] | 0.59 | -0.05 [-0.10, -0.01] | 0.105 | no |
| unserved_energy_mwh | 48.55 [41.53, 55.34] | 2.66 | +45.89 [+38.38, +53.32] | 0.002 | no |
| mean_ambulance_response_delay_hours | 5.85 [4.49, 7.04] | 4.30 | +1.55 [+0.68, +2.47] | 0.006 | no |
| requests_completed | 2.62 [1.97, 3.40] | 9.20 | -6.58 [-8.19, -5.04] | 0.002 | no |
| episode_reward | -4562.42 [-5750.84, -3316.93] | -4331.98 | -230.44 [-890.96, +313.51] | 0.922 | no |

## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 42.75 [26.85, 61.05] | 43.57 | -0.81 [-13.35, +14.31] | 0.432 | no |
| unmet_patient_hours | 269.93 [194.26, 344.84] | 272.79 | -2.87 [-52.46, +54.36] | 0.492 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.66 [0.61, 0.72] | 0.70 | -0.04 [-0.08, -0.01] | 0.084 | no |
| mean_critical_functional_level | 0.55 [0.44, 0.68] | 0.59 | -0.04 [-0.08, -0.00] | 0.105 | no |
| unserved_energy_mwh | 58.45 [50.89, 64.93] | 2.66 | +55.79 [+48.21, +62.69] | 0.002 | no |
| mean_ambulance_response_delay_hours | 6.34 [4.78, 7.71] | 4.30 | +2.04 [+1.18, +2.86] | 0.004 | no |
| requests_completed | 7.46 [5.13, 10.01] | 9.20 | -1.74 [-2.77, -0.83] | 0.006 | no |
| episode_reward | -4474.52 [-5613.13, -3250.66] | -4331.98 | -142.54 [-822.58, +431.35] | 1.000 | no |
