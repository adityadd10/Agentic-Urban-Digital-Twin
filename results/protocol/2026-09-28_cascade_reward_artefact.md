# Note: cascade term in the reward (recorded before the 2M run)

Dated 2026-09-28, written **before** the pre-registered 2M full-budget training starts. It
changes nothing in that experiment. It records a suspected reward-design artefact so the finding
can't be mistaken for something noticed only after seeing results.

## The artefact

The cascade metric (dev doc §3.5) counts a facility whose functional level is < 0.5 while its
intrinsic level is ≥ 0.5. If a flooded facility is **repaired** while its supplier is still
down, it becomes such a facility and is counted as a new cascade (observed in Experiment A:
rule-based's repaired hospitals counted at ticks 213/216 in flood_test_00).

That count is also a **reward term** (dev doc §5.4: −5 × new cascade failures, normalised).
With `configs/reward.yaml`, one cascade costs 5 / 0.009375 ≈ 533 reward units, versus
10 / 0.21024 ≈ 47.6 per death-proxy unit: **one counted cascade ≈ 11 deaths**. The reward can
therefore discourage repairing a flooded hospital while the substation is down.

## Decision (user, 2026-09-28): Option 1

Run the 2M experiment exactly as pre-registered (reward A, identical to `baseline-300k`). The
artefact is present identically in the baseline, so the budget comparison stays clean. Afterwards,
test the artefact as its own pre-declared ablation, e.g. (a) no cascade term, or (b) a cascade
count restricted to facilities that were never flood-failed.

**Prediction to test later (stated now):** if the artefact matters, policies trained without it
will repair flooded facilities more often while their supplier is down. The effect on primary
metrics is not predicted.
