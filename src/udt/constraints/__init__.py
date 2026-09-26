"""udt.constraints — module M7 (dev doc §8).

`engine.py`: `check(graph, action, road_network=None) -> ConstraintReport`,
a function-registry of per-rule checks matching `configs/
constraints.yaml`'s declared rules. See `engine.py`'s module docstring
for exactly which of the dev doc's 3 example rules are implemented vs.
disclosed as not-yet-applicable, and `MTP_Module_Planner.md`'s M7 row
for the build story and what's deferred to slice 2 (pre-hoc action
masking + Lagrangian penalty, the training-side half of "constrained
MARL").
"""
