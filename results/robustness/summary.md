# Robustness study — results

Protocol: `results/protocol/2026-09-28_robustness_study.md` (pass rules fixed before any 2M result).
Test split, 10 scenarios × 3 eval seeds; categories vs rule-based by two-sided Wilcoxon, α = 0.05.

## Verdicts (pre-registered rules)

| Algorithm / primary metric | Nominal | Unchanged in | Robust (≥ 18/22) | Flipped in |
|---|---|---|---|---|
| ppo_2M/patient_deaths | no difference | 22/22 | yes | — |
| ppo_2M/unmet_patient_hours | no difference | 22/22 | yes | — |
| mappo_2M/patient_deaths | no difference | 22/22 | yes | — |
| mappo_2M/unmet_patient_hours | no difference | 22/22 | yes | — |

Rule 3 (ordering of do-nothing, rule-based, PPO, MAPPO by mean death proxy): median Spearman ρ = 0.80 (min 0.80) across the 22 conditions.

## Mean death proxy / unmet patient-hours per condition

| Condition | Rule-based | PPO 2M | MAPPO 2M | MAPPO 300k | Do-nothing |
|---|---|---|---|---|---|
| nominal: no perturbation | 43.6 / 273 | 49.8 / 286 | 43.4 / 272 | 43.4 / 278 | 104.9 / 479 |
| R1_low: dependency floors x0.5 | 49.8 / 298 | 58.4 / 312 | 50.7 / 296 | 52.4 / 309 | 111.2 / 504 |
| R1_high: dependency floors x1.5 (cap 0.95) | 30.4 / 214 | 31.0 / 204 | 30.5 / 211 | 30.6 / 221 | 92.3 / 424 |
| R2_low: buffers x0.5 | 55.7 / 317 | 60.3 / 331 | 54.8 / 314 | 56.1 / 326 | 116.8 / 526 |
| R2_high: buffers x1.5 | 25.0 / 191 | 34.6 / 207 | 28.5 / 205 | 29.2 / 202 | 86.4 / 417 |
| R3_low: demand shares x0.5 | 33.8 / 245 | 43.0 / 259 | 36.2 / 248 | 38.1 / 257 | 104.9 / 479 |
| R3_high: demand shares x2 | 47.8 / 289 | 51.8 / 292 | 46.7 / 287 | 48.8 / 304 | 104.9 / 479 |
| R4_low: patient arrival rate x0.5 | 22.9 / 141 | 23.1 / 135 | 21.0 / 129 | 20.8 / 134 | 53.7 / 247 |
| R4_high: patient arrival rate x1.5 | 72.5 / 484 | 83.7 / 477 | 74.5 / 478 | 77.4 / 502 | 163.8 / 771 |
| R5_low: length of stay 24 h | 33.4 / 223 | 31.9 / 225 | 26.6 / 202 | 27.8 / 210 | 99.1 / 461 |
| R5_high: length of stay 72 h | 48.0 / 297 | 55.3 / 300 | 50.7 / 311 | 51.9 / 324 | 106.6 / 488 |
| R6_low: death-proxy deadline 2 h | 61.3 / 160 | 64.8 / 160 | 59.2 / 157 | 60.1 / 159 | 116.4 / 256 |
| R6_high: death-proxy deadline 6 h | 27.2 / 354 | 36.6 / 386 | 29.7 / 356 | 28.8 / 367 | 90.9 / 677 |
| R8_low: fragility depths x0.75 | 63.5 / 341 | 69.9 / 354 | 62.2 / 346 | 65.2 / 358 | 109.7 / 497 |
| R8_high: fragility depths x1.25 | 37.4 / 238 | 46.4 / 262 | 39.6 / 253 | 38.1 / 250 | 93.6 / 429 |
| R9_low: road blockage scale 0.27 m | 45.8 / 283 | 51.7 / 294 | 44.7 / 279 | 45.5 / 287 | 105.8 / 484 |
| R9_high: road blockage scale 0.9 m | 39.2 / 258 | 46.2 / 273 | 41.1 / 262 | 41.6 / 271 | 101.7 / 468 |
| R10_low: repair rate x0.5 | 67.6 / 370 | 60.6 / 334 | 55.5 / 326 | 58.2 / 345 | 104.9 / 479 |
| R10_high: repair rate x2 | 22.7 / 199 | 39.5 / 235 | 32.5 / 228 | 33.3 / 229 | 104.9 / 479 |
| R11_low: load surge 0.25, base load 50% | 32.3 / 226 | 42.4 / 253 | 36.8 / 247 | 38.6 / 257 | 104.9 / 479 |
| R11_high: load surge 1.0, base load 80% | 54.8 / 338 | 60.9 / 338 | 50.0 / 304 | 51.4 / 317 | 122.5 / 560 |
| R13: aggregation: min instead of product | 40.2 / 258 | 47.9 / 278 | 41.3 / 264 | 42.1 / 276 | 104.9 / 479 |
| R14: travel-time noise ±15% | 43.6 / 273 | 49.8 / 286 | 43.4 / 272 | 43.3 / 278 | 104.9 / 479 |
| R7_low: 1 ambulance per hospital (rule-based only) | 43.5 / 272 | — | — | — | 104.9 / 479 |
| R7_high: 3 ambulances per hospital (rule-based only) | 43.6 / 273 | — | — | — | 104.9 / 479 |
| R12: second synthetic substation (rule-based only) | 67.4 / 354 | — | — | — | 78.7 / 365 |
