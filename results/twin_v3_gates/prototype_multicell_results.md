# Feasibility test 3: two-cell rainfall + acuity (prototype only)

Plan: prototype_multicell_plan.md. 10 moderate/severe train scenarios, seed k = 0.

| Condition | Mean deaths | Mean unmet p-h |
|---|---|---|
| 1. RB-S (acuity-blind) | 152.40 | 515.4 |
| 2. Simple acuity-aware rule | 154.30 | 512.7 |
| 3. Oracle (45 strategies) | 150.80 | 494.2 |

Headroom (best of 1-2 -> 3): +1.0%

**Reading (fixed in the plan): no meaningful headroom (< 15%)**

| Scenario | RB-S | Rule | Oracle |
|---|---|---|---|
| flood_v3_train_00 | 265 | 268 | 264 |
| flood_v3_train_01 | 136 | 149 | 150 |
| flood_v3_train_02 | 49 | 51 | 37 |
| flood_v3_train_04 | 59 | 59 | 54 |
| flood_v3_train_06 | 215 | 220 | 215 |
| flood_v3_train_07 | 147 | 147 | 145 |
| flood_v3_train_09 | 228 | 229 | 225 |
| flood_v3_train_10 | 214 | 220 | 214 |
| flood_v3_train_11 | 10 | 10 | 9 |
| flood_v3_train_12 | 201 | 190 | 195 |

Most common oracle choices: RuleBasedStrongV3|RuleBasedStrongV3|RuleBasedStrongV3 (188); RuleBasedStrongV3|RuleBasedStrongV3|TransportFixed (14); RuleBasedStrongV3|RuleBasedStrongV3|TransfersFirst (10); RuleBasedStrongV3|RuleBasedStrongV3|CallsFirst (9); Coordinated|RuleBasedStrongV3|RuleBasedStrongV3 (8)
