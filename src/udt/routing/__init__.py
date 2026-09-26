"""udt.routing — module M9b (dev doc §6).

`ood.py`: `route(incident, dep_graph, ...) -> RoutingDecision` — the
ID/OOD router: a registry-membership check (`registry_check`) plus a
k-NN + conformal-p-value novelty check (`featurize_incident`,
`knn_novelty_score`, `conformal_p_value`), both of which must pass for
`path="ID"`. Built for the *open-set* case (explicit user instruction) —
genuine novelty detection, not just a fixed-list membership check — even
though only one incident type ("flood") exists to route against today.

See `ood.py`'s own module docstring for the real, disclosed limitation
in today's actual scenario-generation data (4 of the router's 6 feature
dimensions are currently constant across every flood scenario this
codebase generates) and `MTP_Module_Planner.md`'s M9b row for the full
build story.
"""
