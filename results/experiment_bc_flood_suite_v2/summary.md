## PPO vs rule-based (test, 3 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 50.46 [32.74, 69.41] | 43.57 | +6.89 [-4.13, +20.70] | 0.846 | no |
| unmet_patient_hours | 290.19 [206.99, 375.07] | 272.79 | +17.40 [-27.77, +70.21] | 0.922 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.64 [0.59, 0.71] | 0.70 | -0.06 [-0.10, -0.02] | 0.027 | no |
| mean_critical_functional_level | 0.53 [0.42, 0.66] | 0.59 | -0.06 [-0.11, -0.02] | 0.105 | no |
| unserved_energy_mwh | 0.76 [0.00, 1.85] | 2.66 | -1.90 [-5.38, +0.66] | 0.500 | no |
| mean_ambulance_response_delay_hours | 6.37 [4.93, 7.61] | 4.30 | +2.07 [+1.19, +2.96] | 0.004 | no |
| requests_completed | 2.50 [1.82, 3.31] | 9.20 | -6.70 [-8.39, -5.08] | 0.002 | no |
| episode_reward | -4419.08 [-5578.20, -3187.66] | -4331.98 | -87.10 [-705.83, +423.75] | 0.557 | no |

## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 43.39 [27.81, 61.95] | 43.57 | -0.17 [-12.72, +15.55] | 0.432 | no |
| unmet_patient_hours | 278.35 [201.02, 357.71] | 272.79 | +5.55 [-45.66, +63.69] | 0.695 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.66 [0.61, 0.72] | 0.70 | -0.05 [-0.08, -0.01] | 0.105 | no |
| mean_critical_functional_level | 0.55 [0.44, 0.67] | 0.59 | -0.04 [-0.08, -0.01] | 0.064 | no |
| unserved_energy_mwh | 39.77 [33.92, 45.47] | 2.66 | +37.12 [+30.95, +43.73] | 0.002 | no |
| mean_ambulance_response_delay_hours | 6.82 [5.40, 8.12] | 4.30 | +2.52 [+1.60, +3.37] | 0.002 | no |
| requests_completed | 7.48 [5.17, 9.89] | 9.20 | -1.72 [-2.73, -0.84] | 0.004 | no |
| episode_reward | -4447.97 [-5587.35, -3245.15] | -4331.98 | -115.98 [-836.90, +478.75] | 0.770 | no |
