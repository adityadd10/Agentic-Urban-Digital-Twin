## MAPPO vs rule-based (test, 5 training seeds × 3 eval seeds)

| Metric | Policy | Rule-based | Difference (policy − rule) | p | Policy better & sig. |
|---|---|---|---|---|---|
| patient_deaths | 44.59 [28.41, 62.88] | 43.57 | +1.02 [-10.27, +15.48] | 0.557 | no |
| unmet_patient_hours | 281.45 [202.52, 359.29] | 272.79 | +8.66 [-35.89, +61.32] | 0.846 | no |
| cascading_failure_count | 3.10 [2.00, 4.00] | 3.30 | -0.20 [-0.50, +0.00] | 0.500 | no |
| mean_hospital_functional_level | 0.65 [0.60, 0.71] | 0.70 | -0.05 [-0.09, -0.02] | 0.027 | no |
| mean_critical_functional_level | 0.54 [0.43, 0.67] | 0.59 | -0.05 [-0.09, -0.01] | 0.027 | no |
| unserved_energy_mwh | 40.87 [32.71, 48.21] | 2.66 | +38.21 [+30.67, +45.13] | 0.002 | no |
| mean_ambulance_response_delay_hours | 6.53 [4.89, 8.06] | 4.30 | +2.23 [+1.08, +3.38] | 0.010 | no |
| requests_completed | 5.85 [4.11, 7.62] | 9.20 | -3.35 [-4.27, -2.48] | 0.002 | no |
| episode_reward | -4452.52 [-5622.34, -3193.50] | -4331.98 | -120.53 [-735.27, +395.15] | 1.000 | no |
