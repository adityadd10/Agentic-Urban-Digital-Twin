"""Frozen version identifiers (dev doc §3.8 / §4.3, 2026-09-27).

`ENV_VERSION` names the twin's physics plus the env interface. The policy
registry and the ID/OOD router match on it (`routing/ood.py`'s
`registry_check`), so bumping it makes every policy trained on older physics
unusable automatically, not just by convention. `v0` = the twin before the
2026-09-26 realism revision; `v1` = after it (git tag `twin-v1`).

`SUITE_VERSION` names the frozen scenario suite (`data/scenarios/`). Changing
the generator or the twin after the suite is frozen means a new suite version,
never an in-place regeneration.
"""

from __future__ import annotations

ENV_VERSION = "udt_multi_env_v1"
TWIN_GIT_TAG = "twin-v1"
SUITE_VERSION = "flood_suite_v1"
