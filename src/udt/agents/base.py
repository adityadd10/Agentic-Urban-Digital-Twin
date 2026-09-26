"""Agent interface (dev doc §5.6: "the same `Agent.act(obs) -> actions`
interface as the learned agents so the harness treats all policies
identically").

Prototype-1 scope: `act` takes the twin's live graph directly, not a Gym
observation vector — `envs/single_env.py`/`multi_env.py` (dev doc §5.1)
don't exist yet (M5/M6a), so there is no observation-vector contract to
conform to yet. When those land, an M5+ agent will still satisfy this same
`Protocol` — only the shape of what "obs" means changes (graph now, a
flattened vector later), not the calling convention `experiments/runner.py`
depends on.

`road_network` (M4 addition, ambulance-dispatch slice): optional because
not every agent needs it (`DoNothingAgent` ignores it; a future goal-
conditioned MARL agent might route differently). Passed explicitly rather
than bolted onto the graph itself because it's a query interface (real
Dijkstra travel times), not twin state — dev doc §5.3 says the dispatch
*decision* itself, not just its enactment, "routes via Dijkstra on
current travel times", so the agent legitimately needs to query it to
pick the nearest idle ambulance, not just receive the routing result
after the fact.

`edge_states` (M4 addition, patient-transfer slice): dev doc §5.6's
transfer rule triggers on a hospital "predicted [...] to lose power
within its buffer window" — that remaining-buffer number is
`Simulator`-owned mutable state (`twin/cascade.py`'s `EdgeRuntimeState`),
not part of the graph's `Asset`/`DependencyEdge` objects (which only
carry each edge's static *capacity*), so it has to be passed in
separately, same reasoning as `road_network`.
"""

from __future__ import annotations

from typing import Protocol

import networkx as nx

from udt.common.models import AgentAction
from udt.twin.cascade import EdgeRuntimeState
from udt.twin.road_network import RoadNetwork


class Agent(Protocol):
    """Structural interface — `RuleBasedAgent`/`DoNothingAgent`
    (`rule_based.py`) and, later, M5+'s learned agents all satisfy this
    without inheriting from it (a `Protocol`, not an ABC, per dev doc
    §12.2's preference for structural typing over inheritance here)."""

    name: str

    def act(
        self,
        graph: nx.DiGraph[str],
        tick: int,
        road_network: RoadNetwork | None = None,
        edge_states: dict[str, EdgeRuntimeState] | None = None,
    ) -> AgentAction: ...
