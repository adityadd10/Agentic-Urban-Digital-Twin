# Energy-floor ablation — A (floor) vs B (no floor), val split

Scored with A's reward definition. Per-scenario means (5 training seeds × 3 eval seeds).

| Metric | A | B | B − A [95% CI] | p | Rule-based |
|---|---|---|---|---|---|
| patient_deaths | 61.46 | 68.41 | +6.95 [+1.40, +14.03] | 0.055 | 57.03 |
| unmet_patient_hours | 328.60 | 352.76 | +24.15 [-3.56, +58.81] | 0.426 | 310.26 |
| cascading_failure_count | 1.88 | 2.00 | +0.12 [+0.00, +0.36] | 1.000 | 2.50 |
| mean_hospital_functional_level | 0.67 | 0.65 | -0.02 [-0.04, -0.00] | 0.055 | 0.67 |
| mean_critical_functional_level | 0.60 | 0.58 | -0.02 [-0.04, +0.00] | 0.195 | 0.59 |
| unserved_energy_mwh | 39.43 | 0.00 | -39.43 [-48.72, -28.39] | 0.002 | 1.67 |
| mean_ambulance_response_delay_hours | 7.48 | 6.76 | -0.72 [-1.31, -0.18] | 0.074 | 4.73 |
| requests_completed | 8.59 | 5.47 | -3.11 [-4.34, -1.93] | 0.004 | 13.73 |
| episode_reward | -4775.48 | -4875.07 | -99.59 [-513.55, +269.46] | 0.922 | -4718.97 |

**Decision (protocol §5): USE A**

```
{
  "(a) energy: B <= 50% of A and p < 0.05": true,
  "(b) non-inferior (upper CI of B-A <= +10% of A)": {
    "patient_deaths": false,
    "unmet_patient_hours": false
  },
  "verdict": "USE A"
}
```
