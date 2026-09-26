# Urban Digital Twin — demo frontend (dev doc §11.3, module M10c)

React + Leaflet, deliberately small (dev doc's own words). Four panes —
Map, Timeline, Approval modal, Metrics strip — talking to module M10b's
real FastAPI service.

## Run it

```bash
# 1. Backend (from the repo root, needs data/processed/ - run the
#    01-06 pipeline scripts first if you haven't):
uv run uvicorn udt.api.main:app --reload --app-dir src

# 2. Frontend (in this directory):
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173/). The dev
server proxies `/episodes`, `/twin`, `/incidents`, `/decisions`,
`/approvals`, `/ws` to `http://127.0.0.1:8000` (`vite.config.ts`) — no
CORS setup needed for local development.

Click **New episode**, then **Step** to advance it. Click a card in the
Timeline to see its full detail (plan, simulated outcome, risk,
constraint check, approval) in the modal.

No `UDT_LLM_API_KEY` configured → every decision falls back to the
rule-based/safe-hold defaults (visibly flagged in the Timeline as "LLM
fallback") — see `src/udt/llm/anthropic_client.py`'s module docstring
for how to enable real Claude calls.

## Build

```bash
npm run build
```

Type-checks with `tsc`, then bundles with Vite into `dist/`. Set
`VITE_API_BASE_URL` if the backend isn't on the same origin as the
built frontend (e.g. `VITE_API_BASE_URL=https://api.example.com npm run build`).

## Disclosed gaps (see individual component docstrings for the full detail)

- **No auth** — dev doc §11.3's own words: "No auth beyond a static
  token; this is a demo surface, not a public product." Not even the
  static token in this build.
- **No Postgres** — episodes live in the backend's in-memory store
  (module M10b); a backend restart loses all running episodes (the
  decision log itself is still real and on disk).
- **Ambulances don't animate** — `twin/ambulances.py`'s own `Asset`
  geometry never moves (only `status` changes); shown as static markers
  at their home hospital instead.
- **No incident-footprint overlay** — needs an endpoint exposing the
  current incident's geometry, which doesn't exist yet.
- **"Modify" is disabled** in the approval modal — the backend has no
  path yet that takes a `modified_action` and re-executes anything with
  it (`POST /approvals/{decision_id}` only annotates the recorded
  response for audit purposes).
- **Not visually verified by an AI agent** — this was built and its
  TypeScript/production build verified (`npm run build` succeeds, and a
  real dev-server smoke test confirmed the API proxy round-trips real
  data), but no automated browser test renders the actual UI. Open it
  yourself and look.
