"""Deterministic RNG utilities (dev doc §0.4 and §3.6 — determinism is non-negotiable).

Nothing in `src/` may call `random.random()`, `numpy.random.rand()`, or any
other unseeded/global-state sampling. Every function that needs randomness
takes an explicit `rng: np.random.Generator` parameter, constructed here.

Two entry points:
- `make_rng(seed)` — a single top-level generator for a script/run.
- `spawn_rngs(seed, n)` — `n` statistically-independent child generators
  derived from one seed, so e.g. the twin, the incident generator, and an
  agent's exploration noise can each own a private stream without sharing
  state, while the whole run is still reproducible from one integer.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Construct a top-level RNG from an integer seed."""
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Derive `n` independent child RNGs from a single top-level seed.

    Uses `numpy.random.SeedSequence.spawn`, the documented mechanism for
    generating independent streams (avoids the correlation risk of e.g.
    seeding each child with `seed + i`).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    seed_seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in seed_seq.spawn(n)]
