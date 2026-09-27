# Note to protocol 2026-09-27: simulator speed-up and cross-machine runs

Dated 2026-09-28. This is an addendum; the protocol file itself is unchanged.

## 1. Speed-up does not change training (verified)

Protocol §7 requires the full-budget configuration to be identical to `baseline-300k` except for
the budget. Three performance-only changes were made to the twin (commits `de84ae2`, the snapshot
commit, and the allocation-group commit, all 2026-09-28):

1. routing weights computed once per flood-depth scale; one cached Dijkstra search per
   (source, target) at a given scale;
2. per-tick snapshots share the never-modified asset geometry (attributes still deep-copied);
3. supply-allocation groups (static structure) cached per graph.

**Evidence of identical training:**
- `scripts/check_reproduction.py`: MAPPO seeds 0–4 reproduce the first 20 iterations of the
  `baseline-300k` training runs **bit-identically** after each change (every logged loss).
- PPO seed 0: the first 2 updates' logged SB3 statistics (reward, KL, clip fraction, entropy,
  explained variance, losses) are identical to the baseline log, to SB3's printed precision.
- `tests/integration/test_mappo_resume.py` and the full suite (327 tests) pass.

**Throughput** (single MAPPO run, idle MacBook Air M5): 56 → 132 env steps/s (2.4×).

## 2. Runs on a different machine are not bit-identical

The baseline was trained on an Apple M5 (arm64). On a different CPU architecture or BLAS/torch
build, the same seed is **not** expected to reproduce the same numbers bit for bit (different
floating-point kernels), although results should match statistically. Therefore:
- a full-budget run is a **new experiment**, compared with `baseline-300k` statistically (per
  protocol §1), not by bit-identity;
- **a run is never moved between machines mid-training** (resume is exact only on the machine that
  wrote the checkpoint);
- each full-budget run records the machine it ran on (`platform`, CPU model, torch version) in its
  run folder, and the reproduction check is run on that machine first. If it doesn't match the
  baseline bit for bit, the machine difference is reported, not treated as a code change.
