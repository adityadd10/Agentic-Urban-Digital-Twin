# Feasibility test 4C: evacuation under uncertainty (prototype only)

Plan: prototype_hardq_plan.md. 10 moderate/severe train scenarios, seed k = 0. Mean deaths:

| Policy | Mean deaths |
|---|---|
| Fixed E0_never | 73.20 |
| Fixed E1_buffer_lt_2h | 74.00 |
| Fixed E2_depth_ge_0.2 | 80.60 |
| Fixed E3_depth_ge_0.4 (best fixed) | 70.60 |
| Fixed E4_aggressive | 105.40 |
| Fair planner (no foresight, 4 resampled rollouts) | 71.80 |
| Clairvoyant oracle (reference only) | 67.10 |

Fair planner vs best fixed rule: -1.7%; clairvoyant vs best fixed: +5.0%

**Reading (fixed in the plan): no meaningful headroom (< 15%)**

Fair planner's hourly choices: E0_never: 160, E1_buffer_lt_2h: 49, E2_depth_ge_0.2: 19, E4_aggressive: 6, E3_depth_ge_0.4: 6
