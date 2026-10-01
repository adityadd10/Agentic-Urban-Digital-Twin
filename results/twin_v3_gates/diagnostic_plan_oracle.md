# Exploratory diagnostic after gates round 0: is there headroom above RB-S at all?

**Written 2026-10-01, before the diagnostic is implemented or run.** It is exploratory: it cannot
change any gate verdict (`gates.md`, round 0). It only informs the user's choice between a
twin redesign (preventive levers) and reporting the finding as it stands.

**Why:** Gate F's ceiling (7 heuristic candidates, 2 h `simulate` rollouts with no new calls)
found no improvement over RB-S (−0.6%). That could mean there is no headroom, or that the
ceiling was too weak to find it.

**D1: Oracle ceiling.**
- **Candidate decisions:** every 4 decisions (1 h), choose among 30 combined strategies =
  health {RB-S, E-coordinated health} × power {RB-S, D nearest-reachable, E-coordinated} ×
  transport {RB-S, nearest (A-fixed), nearest-functioning (B), calls-first, transfers-first}.
- **Scoring:** each candidate is scored on an exact copy of the env (`resume_state`), so the
  true future calls, arrivals and fragility outcomes are seen (perfect foresight), over 6 h: the
  candidate for 1 h, then RB-S for 5 h.
- **Choice:** the lowest deaths + unmet patient-hours wins (ties: candidate order), and the
  winner is executed for the next hour.

**D2: Physics floor.** RB-S with no facility damage (roads still flood).

**Scope:** the 20 train scenarios, evaluation seed k = 0, compared with RB-S on the same
episodes.

**Reading (fixed now):**
- If D1 beats RB-S's mean deaths by ≥ 15%: headroom exists, and Gate F's ceiling was too weak
  to find it.
- Otherwise: even perfect foresight over the current levers cannot beat RB-S. The twin then
  needs preventive levers (option 1), or the finding is reported as it stands (option 2).
- D2 shows how many deaths are physics-bound.
