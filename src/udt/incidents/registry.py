"""Degradation-function registry (dev doc §4.2, module M3).

`Incident`/`Scenario` live in `common/models.py` per the dev doc's §12.2
rule that ALL shared Pydantic models live there — so there's no separate
`incidents/models.py` in this codebase, only this registry plus
`degradations/flood.py` (the only type in scope for prototype 1; the dev
doc's full type table — substation_failure, fire, cyberattack, unknown —
is deferred, not removed).
"""

from __future__ import annotations

from collections.abc import Callable

DegradationFn = Callable[..., float]
"""`(incident, asset, tick, **context) -> float` — intrinsic-level
reduction for this asset this tick (dev doc §4.2)."""

_REGISTRY: dict[str, DegradationFn] = {}


def register_incident(incident_type: str) -> Callable[[DegradationFn], DegradationFn]:
    def decorator(fn: DegradationFn) -> DegradationFn:
        if incident_type in _REGISTRY:
            raise ValueError(f"Incident type {incident_type!r} already registered")
        _REGISTRY[incident_type] = fn
        return fn

    return decorator


def get_degradation_fn(incident_type: str) -> DegradationFn:
    try:
        return _REGISTRY[incident_type]
    except KeyError:
        raise KeyError(
            f"No degradation function registered for incident type {incident_type!r} "
            f"(registered: {sorted(_REGISTRY)})"
        ) from None


def registered_types() -> list[str]:
    return sorted(_REGISTRY)
