"""Shared, dependency-free building blocks used by every other package.

Per the dev doc (§12.2): every cross-component message is a Pydantic model
defined in `models.py` — no bare dicts cross module boundaries. `config.py`
and `seeding.py` are the two pieces of process-wide plumbing (settings,
deterministic RNGs) that everything else depends on.
"""
