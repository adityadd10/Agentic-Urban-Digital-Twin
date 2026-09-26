"""Policy registry read/write (dev doc §5.5, `registry/policies.yaml`,
module M6b).

Colocated with `agents/marl/` rather than a separate top-level package —
the dev doc's own layout (§12.1) only names the *data* path
(`registry/policies.yaml`), not a source package for the code that
reads/writes it; this is the module that actually produces entries
(`scripts/train_mappo.py` calls `append_entry` after training), so it's
the natural owner. Nothing reads this back programmatically yet — the
ID/OOD router (M9b) that's meant to is still deferred — see `common/
models.py`'s `PolicyRegistryEntry` docstring for why the schema and
write path exist ahead of that reader anyway.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from udt.common.models import PolicyRegistryEntry


def load_registry(path: str | Path) -> list[PolicyRegistryEntry]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return [PolicyRegistryEntry.model_validate(row) for row in data.get("policies", [])]


def append_entry(path: str | Path, entry: PolicyRegistryEntry) -> None:
    """Appends one entry and rewrites the whole file — `policies.yaml`
    is small enough (one row per trained policy, not per training step)
    that read-modify-write is simpler than a streaming append, and
    keeps the file always valid YAML even if a previous write was
    interrupted."""
    path = Path(path)
    entries = load_registry(path)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(
            {"policies": [e.model_dump() for e in entries]},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
