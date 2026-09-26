"""Ensemble disagreement (dev doc §9.2, module M8).

Dev doc §9.2: "Ensemble disagreement: K=5 critics (from the 5 training
seeds) evaluate the chosen action; disagreement = std(returns)/
(|mean|+eps), mapped to [0,1] via val-set quantiles."

**Disclosed prerequisite gap:** "K=5 critics from the 5 training seeds"
needs 5 separately trained MAPPO runs — this session has never run
training at the real 2-5M-step budget (every M5/M6a/M6b/M7 row's own
disclosure), so there are no 5 *real* trained critics to ensemble today.
`compute_disagreement` below is generic over however many value
estimates a caller hands it — real trained critics once they exist, or
(as this module's own tests use) a handful of freshly-initialized ones,
same "mechanism verified, not validated at scale" status as every other
RL-dependent piece this session. The [0,1] normalization (via a p95
reference, not stored here) lives in `risk/engine.py`'s `normalize_to_
p95` — this module only computes the raw, unnormalized statistic dev
doc §9.2 names.
"""

from __future__ import annotations

import numpy as np

# dev doc §9.2's own "+eps" — prevents a division blow-up when every
# critic happens to estimate a return near zero.
EPSILON = 1e-6


def compute_disagreement(returns: list[float]) -> float:
    """dev doc §9.2: `std(returns) / (|mean(returns)| + eps)` — `returns`
    is one value estimate per ensemble critic for the *same* candidate
    action/state. A single-critic "ensemble" can't disagree with itself,
    so that case returns 0.0 rather than dividing by a std of an
    array of size 1 (which numpy already gives as 0.0 — this is a
    documented special case, not a numerical workaround)."""
    if not returns:
        raise ValueError("compute_disagreement needs at least one critic return")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 1:
        return 0.0
    return float(np.std(arr) / (np.abs(np.mean(arr)) + EPSILON))
