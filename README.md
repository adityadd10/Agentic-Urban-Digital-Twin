# mtp-udt — Agentic Urban Digital Twin

Code for the MTP thesis project. Full spec lives in
[`../MTP_Development_Document.md`](../MTP_Development_Document.md); live
build status and module tracking lives in
[`../MTP_Module_Planner.md`](../MTP_Module_Planner.md) — **read the planner
before starting any coding session.**

## Setup

```bash
uv sync --all-groups        # installs into .venv, pinned to Python 3.11
cp .env.example .env        # adjust if needed; never commit .env
docker compose up -d        # Postgres/PostGIS
uv run pytest               # should be green
```

## Common commands

```bash
uv run pytest -m phase0     # run one module's acceptance tests
uv run ruff check .         # lint
uv run ruff format .        # format
uv run mypy                 # typecheck core packages
```

## Layout

See dev doc §12.1. `src/udt/` packages are scaffolded empty and filled in
module-by-module per the planner; each package's `__init__.py` names which
module populates it.
