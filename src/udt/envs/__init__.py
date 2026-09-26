"""udt.envs

M5 (dev doc §5.1-§5.4) added `single_env.py` (`UDTSingleAgentEnv`) —
reduced observation/action scope, see its own docstring. M6a added
`multi_env.py` (`UDTMultiAgentEnv`, PettingZoo `ParallelEnv`, 3 agents),
extended in M6b with goal-conditioning (sampled `g`, observed by every
agent, weights the reward) — see its own docstring for scope.
`wrappers.py` still empty.

See MTP_Module_Planner.md for module status and MTP_Development_Document.md
for the referenced spec section.
"""
