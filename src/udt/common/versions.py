"""Frozen version identifiers (dev doc §3.8 / §4.3, 2026-09-27).

`ENV_VERSION` names the twin's physics plus the env interface. The policy
registry and the ID/OOD router match on it (`routing/ood.py`'s
`registry_check`), so bumping it makes every policy trained on older physics
unusable automatically, not just by convention. `v0` = the twin before the
2026-09-26 realism revision; `v1` = after it (git tag `twin-v1`); `v2` = v1 with
the substation fragility curve corrected to start at 0.1 m as its source says
(git tag `twin-v2`, 2026-09-27). v1 was superseded before any result from it
was reported.

`SUITE_VERSION` names the frozen scenario suite (`data/scenarios/`). Changing
the generator or the twin after the suite is frozen means a new suite version,
never an in-place regeneration.
"""

from __future__ import annotations

ENV_VERSION = "udt_multi_env_v2"
TWIN_GIT_TAG = "twin-v2"
SUITE_VERSION = "flood_suite_v2"

# Twin-v3 (dev doc §3.9): 3 hospitals / 3 substations, interdependent sector
# decisions. Built alongside v2; v2's constants above stay the default for
# every v2 code path. The git tag is set once the v3 action space is frozen.
ENV_VERSION_V3 = "udt_multi_env_v3"
TWIN_V3_GIT_TAG = "twin-v3"
SUITE_VERSION_V3 = "flood_suite_v3"
