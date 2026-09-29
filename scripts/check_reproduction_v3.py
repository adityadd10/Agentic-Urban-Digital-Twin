#!/usr/bin/env python3
"""Twin-v3 reproduction check (dev doc §3.9; protocol addendum 2's freeze step).

Runs the frozen RB-S through the frozen twin-v3 env (default settings: no shed head)
on the 20 train scenarios x seeds k = 0, 1, 2 and requires every episode's metrics to
match `results/twin_v3_gates/rbs_v3_train_reference.json` exactly. The reference is
the shedding pre-check's "off" run, i.e. the same physics recorded before the freeze.
Prints PASS or the first mismatch; exits non-zero on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tune_rbs_v3 import EVAL_SEED_STRIDE, EVAL_SEEDS, SUITE  # noqa: E402
from udt.agents.rbs_v3 import RuleBasedStrongV3  # noqa: E402
from udt.envs.multi_env_v3 import UDTMultiAgentEnvV3  # noqa: E402
from udt.logging.metrics import compute_episode_metrics  # noqa: E402

REFERENCE = REPO_ROOT / "results" / "twin_v3_gates" / "rbs_v3_train_reference.json"
IGNORED = {"agent_name"}


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    env = UDTMultiAgentEnvV3(
        processed_dir=REPO_ROOT / "data/processed",
        reward_config=REPO_ROOT / "configs/reward.yaml",
        suite_dir=SUITE,
        scenario_split="train",
    )
    rbs = RuleBasedStrongV3.frozen()
    i = 0
    for index, sc in enumerate(env._scenarios):
        for k in range(EVAL_SEEDS):
            seed = sc.seed + k * EVAL_SEED_STRIDE
            env.reset(seed=seed, options={"scenario_index": index, "goal": [1, 1, 1, 1]})
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(rbs.act(env))
                done = any(term.values()) or any(trunc.values())
            got = {
                **compute_episode_metrics(sc.scenario_id, "rbs_v3", env.episode_trace).model_dump(),
                "eval_seed": seed,
            }
            want = reference[i]
            diff = {key for key in want if key not in IGNORED and want[key] != got.get(key)}
            if diff:
                print(f"MISMATCH {sc.scenario_id} seed {seed}: {sorted(diff)}")
                raise SystemExit(1)
            i += 1
    print(f"TWIN-V3 REPRODUCTION CHECK: PASS ({i} episodes identical)")


if __name__ == "__main__":
    main()
