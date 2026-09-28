# Reward ablation C vs A — validation split

Scored with A's reward definition. Per-scenario means (5 training seeds × 3 eval seeds).

| Metric | A | C | C − A [95% CI] | p | Rule-based |
|---|---|---|---|---|---|
| patient_deaths | 61.46 | 62.15 | +0.69 [-3.53, +4.63] | 0.910 | 57.03 |
| unmet_patient_hours | 328.60 | 319.95 | -8.65 [-24.25, +5.08] | 0.359 | 310.26 |
| requests_completed | 8.59 | 7.49 | -1.09 [-1.73, -0.41] | 0.035 | 13.73 |
| requests_pending_at_end | 17.64 | 18.77 | +1.13 [+0.45, +1.75] | 0.020 | 15.00 |
| restoration | 0.35 | 0.40 | +0.04 [-0.05, +0.13] | 0.562 | — |
| mean_hospital_functional_level | 0.67 | 0.67 | +0.00 [-0.00, +0.01] | 0.195 | 0.67 |
| unserved_energy_mwh | 39.43 | 69.28 | +29.85 [+22.10, +36.00] | 0.002 | 1.67 |
| cascading_failure_count | 1.88 | 1.86 | -0.02 [-0.12, +0.06] | 1.000 | 2.50 |
| episode_reward | -4775.48 | -4881.50 | -106.02 [-325.71, +119.54] | 0.375 | -4718.97 |

**Decision (protocol §4): REJECT C**

```
{
  "(a) requests_completed: higher than A with p < 0.05": false,
  "(b) non-inferior (upper CI of X-A <= +10% of A)": {
    "patient_deaths": true,
    "unmet_patient_hours": true
  },
  "verdict": "REJECT C"
}
```
