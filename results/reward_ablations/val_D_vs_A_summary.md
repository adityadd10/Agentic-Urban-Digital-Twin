# Reward ablation D vs A — validation split

Scored with A's reward definition. Per-scenario means (5 training seeds × 3 eval seeds).

| Metric | A | D | D − A [95% CI] | p | Rule-based |
|---|---|---|---|---|---|
| patient_deaths | 61.46 | 64.04 | +2.58 [-0.65, +6.79] | 0.570 | 57.03 |
| unmet_patient_hours | 328.60 | 330.54 | +1.94 [-9.40, +15.27] | 1.000 | 310.26 |
| requests_completed | 8.59 | 7.05 | -1.54 [-2.17, -0.95] | 0.004 | 13.73 |
| requests_pending_at_end | 17.64 | 18.89 | +1.25 [+0.41, +2.11] | 0.023 | 15.00 |
| restoration | 0.35 | 0.41 | +0.06 [-0.03, +0.15] | 0.438 | — |
| mean_hospital_functional_level | 0.67 | 0.66 | -0.01 [-0.02, -0.00] | 0.039 | 0.67 |
| unserved_energy_mwh | 39.43 | 37.58 | -1.85 [-11.68, +8.19] | 0.734 | 1.67 |
| cascading_failure_count | 1.88 | 1.92 | +0.04 [+0.00, +0.12] | 1.000 | 2.50 |
| episode_reward | -4775.48 | -4839.80 | -64.32 [-265.38, +111.91] | 0.910 | -4718.97 |

**Decision (protocol §4): REJECT D**

```
{
  "(a) restoration: higher than A with p < 0.05": false,
  "(b) non-inferior (upper CI of X-A <= +10% of A)": {
    "patient_deaths": false,
    "unmet_patient_hours": true
  },
  "verdict": "REJECT D"
}
```
