# Load-shedding pre-check (protocol addendum 2), train split

Frozen RB-S with its shedding rule (on) vs every shed tier forced to 0 (off).
20 scenarios x 3 seeds, per-scenario means, paired.

| Metric | On | Off | On - Off [95% CI] | Holm p | Passes |
|---|---|---|---|---|---|
| patient_deaths | 65.42 | 65.42 | +0.00 [+0.00, +0.00] | 1.000 | False |
| unmet_patient_hours | 375.42 | 375.42 | +0.00 [+0.00, +0.00] | 1.000 | False |

**Decision: REMOVE the shed head (no causal effect)**

**Why the difference is exactly zero:** RB-S did shed (32 MWh across 3 of 60 episodes), yet
deaths and unmet patient-hours were identical in every episode. Where shedding happened, it
changed nothing for patients.

**Exploratory probe (not part of the pre-registered decision; recorded for transparency).**
RB-S with the maximum shed tier (3) on every substation for the whole episode, against no
shedding, on the same 60 train episodes:

| Condition | Deaths | Unmet p-h | Unserved MWh |
|---|---|---|---|
| No shedding | 65.42 | 375.4 | 0 |
| Maximum shedding | 64.28 (−1.7%) | 370.1 (−1.4%) | 827.6 |

Even extreme shedding moves patient outcomes by under 2%, below the pre-registered 5% threshold,
at a large energy cost. Removing the shed head therefore loses no meaningful lever.
