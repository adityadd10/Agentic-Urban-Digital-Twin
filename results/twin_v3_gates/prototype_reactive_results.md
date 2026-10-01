# Feasibility test: reactive levers (prototype only)

Plan: prototype_reactive_plan.md. 10 moderate/severe train scenarios, seed k = 0.

Prototype with no resources reproduces RB-S exactly: True

| Condition | Mean deaths | Mean unmet p-h |
|---|---|---|
| 1. RB-S (no resources) | 56.50 | 332.7 |
| 2. RB-S + reactive rule | 45.40 | 264.9 |
| 3. Oracle placement | 46.10 | 271.1 |

Lever value (1 -> 2): +19.6% deaths. Headroom (2 -> 3): -1.5%.

**Reading (fixed in the plan): no meaningful headroom (< 15%)**

| Scenario | RB-S | Rule | Oracle |
|---|---|---|---|
| flood_v3_train_00 | 2 | 2 | 2 |
| flood_v3_train_01 | 27 | 26 | 27 |
| flood_v3_train_02 | 16 | 2 | 12 |
| flood_v3_train_04 | 43 | 46 | 43 |
| flood_v3_train_06 | 68 | 26 | 26 |
| flood_v3_train_07 | 16 | 14 | 14 |
| flood_v3_train_09 | 167 | 167 | 167 |
| flood_v3_train_10 | 111 | 48 | 47 |
| flood_v3_train_11 | 6 | 6 | 6 |
| flood_v3_train_12 | 109 | 117 | 117 |

Oracle hourly picks [generator, pump] (0 = keep): (0, 0): 234, (0, 1): 3, (1, 2): 1, (2, 0): 1, (1, 0): 1

**Status (2026-10-01): rejected and archived.** The levers cut deaths, but they create no headroom
over a simple rule, so they will not be added to the twin. The script moved to
`scripts/archive/prototype_reactive_levers.py`; it is kept so these numbers stay reproducible.
