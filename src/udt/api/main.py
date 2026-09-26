"""FastAPI service (dev doc §11.2), module M10b.

```
GET  /twin/state                     current TwinState (summary)
POST /episodes                        start episode from scenario_id
POST /episodes/{id}/step?n=1          advance (demo driver)
POST /incidents                       inject incident (demo: "what if X happens now")
GET  /decisions?episode_id=           decision log
GET  /approvals/pending               queue for the console
POST /approvals/{decision_id}         {approve|reject|modify, modified_action?}
WS   /ws/live                         tick-by-tick state + decision events
```

Every route is a thin wrapper over M10a's `EpisodeSession` — see
`api/episode_session.py` for where the actual "route -> plan -> execute
-> constraint-check -> risk-assess -> gate -> log" work happens; this
module's own job is just process-lifetime setup (`api/context.py`'s
`AppContext`) and HTTP/WebSocket plumbing.

**Disclosed scope, same discipline as every other module this
session:**
- No Postgres/PostGIS wiring (dev doc §11.1) — episodes live in an
  in-memory `EpisodeManager` (single process, lost on restart); the
  decision log is still real and on disk regardless
  (`logging/decision_log.py`'s JSONL file per episode).
- No auth (dev doc §11.3 itself: "No auth beyond a static token; this
  is a demo surface, not a public product") — not even the static
  token in this slice; add one before exposing this beyond localhost.
- `POST /approvals/{decision_id}` annotates a decision's *recorded*
  response after the fact — it does not rewind the simulation (`api/
  schemas.py`'s `ApprovalOverrideRequest` docstring has the full
  disclosure on why a live, blocking pending-approval queue isn't built
  this slice).
- The LLM client every episode uses is real (`llm/anthropic_client.py`)
  if `UDT_LLM_API_KEY` is set, else a null client that deliberately
  triggers `llm/graph.py`'s own disclosed fallback path (`api/
  llm_client_factory.py`) — the server runs either way, at zero cost
  with no key configured.

Usage:
  uv run uvicorn udt.api.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from udt.api.context import AppContext
from udt.api.episode_manager import EpisodeManager
from udt.api.llm_client_factory import build_llm_client
from udt.api.routers import decisions, episodes
from udt.api.ws import ConnectionManager
from udt.api.ws import router as ws_router
from udt.common.config import REPO_ROOT, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    context = AppContext()
    app.state.context = context
    app.state.episodes = EpisodeManager()
    app.state.ws_manager = ConnectionManager()
    app.state.llm_client = build_llm_client(
        settings, cache_dir=str(REPO_ROOT / "runs" / "llm_cache")
    )
    yield
    context.close()


app = FastAPI(
    title="Urban Digital Twin API",
    description="dev doc §11.2 — demo surface, not a public product (no auth in this slice).",
    lifespan=lifespan,
)
app.include_router(episodes.router)
app.include_router(decisions.router)
app.include_router(ws_router)
