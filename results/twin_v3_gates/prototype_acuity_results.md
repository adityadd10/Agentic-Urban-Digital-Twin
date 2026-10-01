# Feasibility test 2: patient acuity (prototype only)

Plan: prototype_acuity_plan.md. 10 moderate/severe train scenarios, seed k = 0.

| Condition | Mean deaths | Mean unmet p-h |
|---|---|---|
| 1. RB-S (acuity-blind) | 72.30 | 306.1 |
| 2. Simple acuity-aware rule | 72.00 | 304.0 |
| 3. Oracle (45 strategies, perfect foresight) | 71.40 | 296.6 |

Headroom (best of 1-2 -> 3): +0.8%

**Reading (fixed in the plan): no meaningful headroom (< 15%)**

| Scenario | RB-S | Rule | Oracle |
|---|---|---|---|
| flood_v3_train_00 | 3 | 3 | 3 |
| flood_v3_train_01 | 40 | 42 | 40 |
| flood_v3_train_02 | 25 | 25 | 26 |
| flood_v3_train_04 | 53 | 53 | 53 |
| flood_v3_train_06 | 90 | 90 | 87 |
| flood_v3_train_07 | 15 | 15 | 15 |
| flood_v3_train_09 | 213 | 208 | 211 |
| flood_v3_train_10 | 134 | 129 | 126 |
| flood_v3_train_11 | 6 | 5 | 6 |
| flood_v3_train_12 | 144 | 150 | 147 |

Most common oracle choices: RuleBasedStrongV3|RuleBasedStrongV3|RuleBasedStrongV3 (197); RuleBasedStrongV3|RuleBasedStrongV3|TransportFixed (18); RuleBasedStrongV3|RuleBasedStrongV3|TransfersFirst (11); Coordinated|RuleBasedStrongV3|RuleBasedStrongV3 (7); RuleBasedStrongV3|RuleBasedStrongV3|CallsFirst (5)
