## PPO vs rule-based (test, 3 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 49.81 [31.97, 69.41] | 43.57 | +6.24 [-4.67, +20.02] | 0.846 | no |
| unmet_patient_hours | 285.71 [201.01, 373.45] | 272.79 | +12.92 [-32.08, +65.19] | 0.922 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.64 [0.59, 0.71] | 0.70 | -0.06 [-0.10, -0.03] | 0.027 | no |
| mean_critical_functional_level | 0.53 [0.42, 0.66] | 0.59 | -0.06 [-0.11, -0.02] | 0.084 | no |
| unserved_energy_mwh | 1.53 [0.00, 4.43] | 2.66 | -1.13 [-5.81, +3.35] | 0.875 | no |
| mean_ambulance_response_delay_hours | 7.86 [4.81, 10.92] | 4.69 | +3.18 [+1.25, +5.10] | 0.500 | no |
| requests_completed | 0.30 [0.00, 0.81] | 9.20 | -8.90 [-11.28, -6.68] | 0.002 | no |
| episode_reward | -4307.50 [-5519.31, -3000.60] | -4331.98 | +24.48 [-605.16, +549.83] | 0.557 | no |

## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 43.43 [27.83, 61.37] | 43.57 | -0.14 [-12.73, +15.18] | 0.432 | no |
| unmet_patient_hours | 271.74 [196.08, 349.59] | 272.79 | -1.06 [-55.78, +60.47] | 0.770 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.66 [0.61, 0.71] | 0.70 | -0.05 [-0.09, -0.00] | 0.064 | no |
| mean_critical_functional_level | 0.54 [0.43, 0.67] | 0.59 | -0.05 [-0.10, -0.00] | 0.160 | no |
| unserved_energy_mwh | 50.84 [43.59, 58.32] | 2.66 | +48.18 [+41.55, +55.16] | 0.002 | no |
| mean_ambulance_response_delay_hours | 6.06 [4.22, 7.75] | 4.30 | +1.76 [+0.71, +2.82] | 0.014 | no |
| requests_completed | 4.47 [3.13, 5.96] | 9.20 | -4.73 [-5.69, -3.70] | 0.002 | no |
| episode_reward | -4351.71 [-5502.77, -3159.63] | -4331.98 | -19.73 [-710.52, +567.63] | 0.695 | no |
