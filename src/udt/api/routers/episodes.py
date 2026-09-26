"""Episode lifecycle + twin-state + incident-injection routes (dev doc
§11.2), module M10b.

```
GET  /twin/state?episode_id=          current TwinState (summary)
GET  /twin/graph                      static dependency graph (assets + edges)
POST /episodes                        start episode from scenario_id
POST /episodes/{id}/step?n=1          advance (demo driver)
POST /incidents                       inject incident (demo: "what if X happens now")
```

**`GET /twin/graph` (M10c addition):** not one of dev doc §11.2's own
named routes — added because M10c's Map pane needs the *static*
topology (asset geometry + dependency edges) to draw the base map and
"dependency edges toggled on hover", which `TwinState` alone doesn't
carry (it only has per-tick `functional_level`/`intrinsic_level`, not
edges). The same real `dep_graph` every episode is built from — one
ward, one topology, so this doesn't need an `episode_id`.

**Disclosed deviation, `POST /episodes`:** dev doc says "start episode
from scenario_id" — no frozen scenario suite exists yet (M3's own
deferral), so there's no `scenario_id` to reference. `CreateEpisodeRequest`
carries what `generate_flood_scenario` actually needs to build one
fresh instead (`seed`, `severity_range`), same disclosed stand-in every
other "no frozen suite" workaround this session has used.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

from udt.api.deps import get_context, get_episode_manager, get_llm_client, get_ws_manager
from udt.api.episode_session import EpisodeSession, RuleBasedPolicyExecutor
from udt.api.schemas import (
    CreateEpisodeRequest,
    CreateEpisodeResponse,
    InjectIncidentRequest,
    StepResponse,
)
from udt.common.config import REPO_ROOT
from udt.common.models import DependencyGraph, Incident, TwinState
from udt.incidents.degradations.flood import make_flood_degradation_fn
from udt.risk.human_model import HumanModel
from udt.scenarios.generator import (
    apply_initial_conditions,
    generate_flood_scenario,
    onset_hour_of_day,
)
from udt.twin.simulator import Simulator

router = APIRouter()

# dev doc §9.2's own "K=5" - disclosed stand-in (no 5 trained MAPPO
# seeds exist yet, every M5-M7 row's own disclosure). A fixed neutral
# value here, not even a fresh-network forward pass - this layer's job
# is serving requests, not re-deriving `scripts/calibrate_risk.py`'s
# own critic-scoring machinery.
_FRESH_CRITIC_RETURNS = [0.0] * 5


def _get_session(req: Request, episode_id: str) -> EpisodeSession:
    session = get_episode_manager(req).get(episode_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"episode {episode_id!r} not found")
    return session


@router.post("/episodes", response_model=CreateEpisodeResponse)
def create_episode(request: CreateEpisodeRequest, req: Request) -> CreateEpisodeResponse:
    ctx = get_context(req)
    episode_id = f"ep_{uuid.uuid4().hex[:12]}"

    scenario = generate_flood_scenario(
        scenario_id=episode_id,
        ward_boundary_geojson=ctx.ward_boundary,
        seed=request.seed,
        severity_range=request.severity_range,
    )
    incident = scenario.incident
    degradation_fn = make_flood_degradation_fn(incident, ctx.raster)
    sim = Simulator(
        apply_initial_conditions(ctx.dep_graph, scenario),
        seed=request.seed,
        road_network=ctx.road_network,
        onset_hour_of_day=onset_hour_of_day(scenario),
    )

    session = EpisodeSession(
        episode_id=episode_id,
        sim=sim,
        incident=incident,
        dep_graph=ctx.dep_graph,
        degradation_fn=degradation_fn,
        road_network=ctx.road_network,
        ward_polygon=ctx.ward_polygon,
        llm_client=get_llm_client(req),
        policy_executor=RuleBasedPolicyExecutor(),
        env_version=ctx.env_version,
        registry_entries=ctx.registry_entries,
        router_calibration=ctx.router_calibration,
        risk_calibration=ctx.risk_calibration,
        critic_returns=_FRESH_CRITIC_RETURNS,
        human_model=HumanModel(),
        decision_log_path=REPO_ROOT / "runs" / episode_id / "decisions.jsonl",
        n_ticks_max=request.n_ticks_max,
    )
    get_episode_manager(req).add(session)
    return CreateEpisodeResponse(
        episode_id=episode_id, incident_id=incident.incident_id, severity=incident.severity
    )


@router.post("/episodes/{episode_id}/step", response_model=StepResponse)
async def step_episode(episode_id: str, req: Request, n: int = 1) -> StepResponse:
    session = _get_session(req, episode_id)
    n_decisions_before = len(session.decisions)
    snapshots = session.step(n)
    new_decisions = session.decisions[n_decisions_before:]

    ws_manager = get_ws_manager(req)
    await ws_manager.broadcast(
        episode_id,
        {
            "type": "step",
            "tick": session.sim.tick,
            "n_snapshots": len(snapshots),
            "n_decisions": len(new_decisions),
        },
    )

    return StepResponse(
        tick=session.sim.tick,
        snapshots=snapshots,
        decisions=new_decisions,
        episode_ended=session.ended,
    )


@router.get("/twin/graph", response_model=DependencyGraph)
def get_twin_graph(req: Request) -> DependencyGraph:
    return get_context(req).dep_graph


@router.get("/twin/state", response_model=TwinState)
def get_twin_state(episode_id: str, req: Request) -> TwinState:
    session = _get_session(req, episode_id)
    state = session.latest_state
    if state is None:
        raise HTTPException(
            status_code=409, detail="episode has not been stepped yet - call step first"
        )
    return state


@router.post("/incidents")
async def inject_incident(request: InjectIncidentRequest, req: Request) -> dict[str, str]:
    ctx = get_context(req)
    session = _get_session(req, request.episode_id)

    onset_tick = request.onset_tick if request.onset_tick is not None else session.sim.tick
    incident = Incident(
        incident_id=f"{request.episode_id}_injected_{onset_tick}",
        type=request.incident_type,
        location=ctx.ward_boundary["features"][0]["geometry"],
        onset_tick=onset_tick,
        severity=request.severity,
        directly_affected_assets=[],
    )
    degradation_fn = (
        make_flood_degradation_fn(incident, ctx.raster)
        if request.incident_type == "flood"
        else None
    )
    session.inject_incident(incident, degradation_fn)

    ws_manager = get_ws_manager(req)
    await ws_manager.broadcast(
        request.episode_id,
        {
            "type": "incident_injected",
            "incident_id": incident.incident_id,
            "severity": incident.severity,
        },
    )
    return {"incident_id": incident.incident_id}
