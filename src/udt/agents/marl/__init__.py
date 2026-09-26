"""udt.agents.marl

M6a (dev doc §5.5) added `networks.py` (`MultiCategoricalActor`,
`CentralizedCritic` — independent per-agent actors + one shared critic,
see `networks.py`'s docstring for why that's not literally "parameter-
shared actors with agent-ID embedding"), `buffer.py` (`RolloutBuffer` +
GAE), `mappo.py` (`MAPPOTrainer`). `masking.py` (M7's constraint-engine
action masking) still empty.

See MTP_Module_Planner.md for module status and MTP_Development_Document.md
for the referenced spec section.
"""
