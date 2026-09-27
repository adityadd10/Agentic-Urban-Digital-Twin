# baseline-300k — frozen preliminary learned-policy result

Immutable reference for Step 8 (git tag `baseline-300k` → commit `52840c7`). **Do not modify.**

- PPO: 3 seeds × 301,056 steps · MAPPO: 5 seeds × 307,200 steps (1,200 iterations × 256)
- twin-v2 · `udt_multi_env_v2` · `flood_suite_v2` · trained on the train split at commit `847297e`
- Reported checkpoint: final
- Results: `../experiment_bc_flood_suite_v2/` (vs rule-based: `../experiment_a_flood_suite_v2/`)
- `manifest.json`: per-model type, seed, true step count, env version, original path, SHA-256
- `models/`: archived copies (checksums verified against the originals)

Verify integrity: `shasum -a 256 models/*` must match `manifest.json`.
