# Twin-v3 design gates (train split)

Protocol 2026-09-29_twin_v3_success_criteria §6 + addendum 7. X better than Y is positive.

| Gate | X vs Y | Mechanism (margin) | Improvement [95% CI] | Mech. | Outcome check | Gate |
|---|---|---|---|---|---|---|
| A-health | rbs vs a_health_fixed | unmet_patient_hours (37.378) | +199.840 [+153.440, +248.717] | pass (better: X) | patient_deaths +53.17 [+39.13, +67.63] Holm p=0.000 | **PASS** |
| A-power | rbs vs a_power_fixed | mean_critical_functional_level (0.020) | +0.106 [+0.073, +0.141] | pass (better: X) | patient_deaths +39.68 [+19.58, +61.38] Holm p=0.004; unmet_patient_hours +133.83 [+68.76, +204.07] Holm p=0.004 | **PASS** |
| A-transport | rbs vs a_transport_fixed | mean_casualty_time_to_admission_hours (0.347) | +0.193 [+0.101, +0.291] | fail (better: X) | patient_deaths +3.45 [+1.45, +5.80] Holm p=0.003; unmet_patient_hours +11.68 [+5.42, +17.46] Holm p=0.003 | **FAIL** |
| B-destination | rbs vs b_nearest_functioning | mean_casualty_time_to_admission_hours (0.347) | +0.036 [-0.021, +0.090] | fail (better: None) | patient_deaths +1.07 [+0.18, +2.20] Holm p=0.034 | **FAIL** |
| C-fleet | c_calls_first vs c_transfers_first | jobs_completed (3.501) | -0.267 [-1.200, +0.650] | fail (better: None) | patient_deaths +0.03 [-0.60, +0.62] Holm p=0.570; unmet_patient_hours -10.44 [-17.99, -4.97] Holm p=0.001 | **FAIL** |
| D-repair-order | rbs vs d_nearest_reachable | mean_critical_functional_level (0.020) | +0.009 [+0.003, +0.018] | fail (better: None) | unmet_patient_hours +5.77 [-7.21, +20.91] Holm p=0.600 | **FAIL** |
| E-coordination | e_coordinated vs a_transport_fixed | patient_deaths (7.267) | +3.417 [+1.317, +5.867] | fail (better: X) | unmet_patient_hours +12.80 [+7.07, +18.25] Holm p=0.002; mean_critical_functional_level -0.00 [-0.00, +0.00] Holm p=1.000 | **FAIL** |

**F-headroom:** RB-S deaths 65.42; ceiling improvement -0.40 (-0.6%, CI [-1.37, +0.48]); threshold 15% -> **FAIL**

**All gates pass: False**
