"""Twin-v2 / twin-v3 process mode (dev doc §3.9, "process guard").

Twin-v3 behaviour is switched on through module-level settings
(`twin/ambulances.py`'s `COUNT_UNCOLLECTED_CASUALTY_DEATHS`, `twin/crew.py`'s
`REPAIR_CREW_TRAVEL`). They are process-wide, so a v2 and a v3 environment in
the same process would silently change each other's physics. The v3 env calls
`enable_twin_v3()`; v2 envs call `require_twin_v2()`, which refuses to run once
v3 settings are on. Run v2 and v3 experiments in separate processes.
"""

from __future__ import annotations

from udt.twin import ambulances, crew


def twin_v3_enabled() -> bool:
    return bool(ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS or crew.REPAIR_CREW_TRAVEL)


def enable_twin_v3() -> None:
    ambulances.COUNT_UNCOLLECTED_CASUALTY_DEATHS = True
    crew.REPAIR_CREW_TRAVEL = True


def require_twin_v2() -> None:
    if twin_v3_enabled():
        raise RuntimeError(
            "twin-v3 settings are on in this process; run twin-v2 code in a separate process "
            "(udt.twin.modes)"
        )
