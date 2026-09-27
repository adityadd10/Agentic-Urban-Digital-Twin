## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 48.46 [32.42, 66.72] | 43.57 | +4.89 [-6.35, +19.47] | 0.625 | no |
| unmet_patient_hours | 285.54 [205.65, 369.66] | 272.79 | +12.75 [-34.66, +68.32] | 0.922 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.64 [0.59, 0.71] | 0.70 | -0.06 [-0.10, -0.03] | 0.027 | no |
| mean_critical_functional_level | 0.53 [0.42, 0.66] | 0.59 | -0.06 [-0.11, -0.02] | 0.105 | no |
| unserved_energy_mwh | 0.00 [0.00, 0.00] | 2.66 | -2.66 [-6.40, +0.00] | 0.500 | no |
| mean_ambulance_response_delay_hours | 5.83 [4.28, 7.27] | 4.30 | +1.53 [+0.61, +2.45] | 0.027 | no |
| requests_completed | 4.30 [2.99, 5.73] | 9.20 | -4.90 [-5.89, -3.84] | 0.002 | no |
| episode_reward | -4368.91 [-5495.20, -3179.17] | -4331.98 | -36.93 [-674.87, +469.76] | 0.492 | no |
