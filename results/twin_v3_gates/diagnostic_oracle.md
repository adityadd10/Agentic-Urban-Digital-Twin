# Exploratory diagnostic: oracle ceiling and physics floor (train, seed k = 0)

Plan: diagnostic_plan_oracle.md. Exploratory; cannot change gate verdicts.

Scenarios: 20. Mean deaths: RB-S 67.30 | oracle ceiling 66.25 (+1.6% vs RB-S) | physics floor (no facility damage) 5.40

**Reading (fixed in the plan): no meaningful headroom (< 15%)**

| Scenario | RB-S | Oracle | Floor |
|---|---|---|---|
| flood_v3_train_00 | 2 | 2 | 2 |
| flood_v3_train_01 | 27 | 25 | 7 |
| flood_v3_train_02 | 16 | 16 | 2 |
| flood_v3_train_03 | 26 | 25 | 2 |
| flood_v3_train_04 | 43 | 43 | 5 |
| flood_v3_train_05 | 1 | 1 | 1 |
| flood_v3_train_06 | 68 | 70 | 7 |
| flood_v3_train_07 | 16 | 16 | 6 |
| flood_v3_train_08 | 6 | 3 | 1 |
| flood_v3_train_09 | 167 | 171 | 6 |
| flood_v3_train_10 | 111 | 107 | 9 |
| flood_v3_train_11 | 6 | 6 | 6 |
| flood_v3_train_12 | 109 | 110 | 16 |
| flood_v3_train_13 | 124 | 119 | 9 |
| flood_v3_train_14 | 93 | 87 | 6 |
| flood_v3_train_15 | 2 | 2 | 2 |
| flood_v3_train_16 | 177 | 178 | 4 |
| flood_v3_train_17 | 162 | 159 | 16 |
| flood_v3_train_18 | 9 | 8 | 0 |
| flood_v3_train_19 | 181 | 177 | 1 |

Oracle's hourly choices (most common):
- RuleBasedStrongV3|RuleBasedStrongV3|RuleBasedStrongV3: 395
- RuleBasedStrongV3|RuleBasedStrongV3|TransfersFirst: 23
- RuleBasedStrongV3|RuleBasedStrongV3|TransportFixed: 21
- RuleBasedStrongV3|RuleBasedStrongV3|CallsFirst: 15
- Coordinated|RuleBasedStrongV3|RuleBasedStrongV3: 14
- RuleBasedStrongV3|RuleBasedStrongV3|NearestFunctioningDestination: 6
- Coordinated|RuleBasedStrongV3|NearestFunctioningDestination: 1
- Coordinated|RuleBasedStrongV3|TransportFixed: 1
