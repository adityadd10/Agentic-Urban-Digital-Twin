"""udt.api — module M10b (dev doc §11.2).

`main.py`: the FastAPI app + process-lifetime setup. `context.py`:
`AppContext` — the real Kurla graph/raster/road-network/registry/
calibrations, loaded once. `episode_session.py`: `EpisodeSession` — the
actual "route -> plan -> execute -> constraint-check -> risk-assess ->
gate -> log" loop (shared with `experiments/decision_loop.py`, M10a).
`episode_manager.py`: the in-memory episode registry. `llm_client_
factory.py`: picks a real or null `LLMClient` per process.
`schemas.py`: HTTP request/response models. `routers/`: the actual
endpoints. `ws.py`: the live WebSocket channel.

See `main.py`'s own module docstring for the full disclosed scope (no
Postgres/PostGIS, no auth, no live blocking approval queue) and
`MTP_Module_Planner.md`'s M10b row for the build story.
"""
