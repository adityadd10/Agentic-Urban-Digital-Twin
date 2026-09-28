# Full budget (2M) vs baseline-300k — test split (flood_suite_v2)

Per-scenario means; paired difference (2M − 300k), 95% bootstrap CI, two-sided Wilcoxon.

## PPO

| Metric | 300k | 2M | 2M − 300k [95% CI] | p |
|---|---|---|---|---|
| patient_deaths **(primary)** | 50.46 | 49.81 | -0.64 [-1.96, +0.78] | 0.148 |
| unmet_patient_hours **(primary)** | 290.19 | 285.71 | -4.48 [-9.63, +0.59] | 0.049 |
| mean_hospital_functional_level | 0.64 | 0.64 | -0.00 [-0.00, -0.00] | 0.125 |
| mean_critical_functional_level | 0.53 | 0.53 | -0.00 [-0.00, -0.00] | 0.125 |
| unserved_energy_mwh | 0.76 | 1.53 | +0.77 [-1.53, +4.10] | 1.000 |
| mean_ambulance_response_delay_hours | 7.55 | 7.86 | +0.31 [-2.36, +2.98] | 1.000 |
| requests_completed | 2.50 | 0.30 | -2.20 [-3.16, -1.37] | 0.004 |
| episode_reward | -4419.08 | -4307.50 | +111.57 [-8.36, +225.82] | 0.049 |
| cascading_failure_count | 3.10 | 3.10 | +0.00 [+0.00, +0.00] | 1.000 |

## MAPPO

| Metric | 300k | 2M | 2M − 300k [95% CI] | p |
|---|---|---|---|---|
| patient_deaths **(primary)** | 43.39 | 43.43 | +0.03 [-1.88, +1.93] | 0.943 |
| unmet_patient_hours **(primary)** | 278.35 | 271.74 | -6.61 [-13.33, -0.80] | 0.131 |
| mean_hospital_functional_level | 0.66 | 0.66 | -0.00 [-0.01, +0.01] | 0.844 |
| mean_critical_functional_level | 0.55 | 0.54 | -0.01 [-0.02, +0.01] | 0.461 |
| unserved_energy_mwh | 39.77 | 50.84 | +11.07 [+2.16, +19.90] | 0.049 |
| mean_ambulance_response_delay_hours | 6.82 | 6.06 | -0.76 [-1.49, -0.11] | 0.160 |
| requests_completed | 7.48 | 4.47 | -3.01 [-4.02, -2.01] | 0.002 |
| episode_reward | -4447.97 | -4351.71 | +96.25 [-2.26, +200.40] | 0.160 |
| cascading_failure_count | 3.10 | 3.10 | +0.00 [+0.00, +0.00] | 1.000 |

