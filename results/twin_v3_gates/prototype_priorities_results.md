# Feasibility test 4A: changing priorities (prototype only)

Plan: prototype_hardq_plan.md. 10 moderate/severe train scenarios, seed k = 0. O1 (patients) from the oracle diagnostic: +1.6%.

| Objective | RB-S mean J | Oracle mean J | Oracle better by | Reading |
|---|---|---|---|---|
| O2_restore_infrastructure (n=10) | 1.259 | 1.252 | +0.6% | **no meaningful headroom (< 15%)** |
| O3_fast_response (n=10) | 1.282 | 1.165 | +9.1% | **no meaningful headroom (< 15%)** |

Detail (means): O2_restore_infrastructure: deaths RB-S 56.5 / oracle 55.2, critical function 0.676 / 0.676; O3_fast_response: deaths RB-S 56.5 / oracle 57.7, critical function 0.676 / 0.676
