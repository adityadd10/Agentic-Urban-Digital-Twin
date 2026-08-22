"""Shared cross-component Pydantic models (dev doc §12.2).

Every message that crosses a module boundary in this codebase is a Pydantic
model defined here — no bare dicts. This file is intentionally empty until
Phase 1 (twin core) adds the first models (`Asset`, `TwinState`, ...).

If a later module needs to change a model defined here, the dev doc
(`MTP_Development_Document.md`) is the source of truth: update it there
first, then here — see `MTP_Module_Planner.md` §1.
"""

from __future__ import annotations
