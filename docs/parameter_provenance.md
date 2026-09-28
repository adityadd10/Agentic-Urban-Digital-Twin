# Parameter provenance

Every parameter that shapes simulated outcomes, with its source. Compiled 2026-09-28 from the
code at tag `train-2M-start` (`d24ae43`, twin-v2). Values are read from the code, not the dev doc
(where they differ, the code is what ran).

**Categories**

| | Meaning |
|---|---|
| **A** | Empirically measured: real data for this place |
| **B** | Literature-derived: taken from a cited study |
| **C** | Geographically derived: computed from real geodata by a stated method |
| **D** | Engineering assumption: a deliberate modelling choice with a stated rationale, no source |
| **E** | Synthetic: random draw or placeholder standing in for data that doesn't exist |

**R-id** = perturbed in the pre-registered robustness study
(`results/protocol/2026-09-28_robustness_study.md`). "—" = not perturbed (reason given where
it matters).

## 1. Geography and assets

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Ward boundary | L Ward (Kurla) | `data/processed/ward_boundary.geojson` | A | BMC ArcGIS `BMC_Ward` | — |
| Hospital locations, names, types | 2 hospitals | `facilities.geojson` | A | BMC ArcGIS `Health_Facilities` | — |
| Substation location | 1 (S0) | `facilities.geojson` | A | OSM `power=substation` (only one tagged in the ward) | R12 |
| Water facilities | 2, placed near hospitals | `data/manual_facilities.yaml` | E | None public (checked: OSM, BMC CDP 2005–25) | — |
| Road network | 10,933 nodes / 21,547 edges | `roads_full.graphml` | A | OSM official API | — |
| Access roads per facility | within 100 m | `configs/data.yaml` `facility_road_buffer_m` | D | Proximity rule | — |
| Elevation | Copernicus GLO-30 | `dem.tif` | A | Copernicus DEM | — |
| Drainage | 53 OSM waterways | `drainage.geojson` | A | OSM `waterway=*` | — |
| Land cover (impervious flag) | OSM landuse + buildings | `landuse.geojson` | A | OSM | — |
| Hospital beds | Uniform(60, 300) per hospital (seed 0) | `06_build_dependency_graph.py` `HOSPITAL_BEDS_RANGE` | **E** | Placeholder; real bed counts not used | R4 (via demand) |
| ICU beds | 10% of beds | `06_build…` | D | Rule of thumb | — |
| Initial bed occupancy | U(0.6, 0.9) per scenario | `scenarios/generator.py` | D | Dev doc §4.3 | — |
| Substation capacity | Uniform(20, 50) MW (S0: 20.5) | `06_build…` `SUBSTATION_CAPACITY_MW_RANGE` | **E** | Placeholder | R11 (via load share) |
| Substation base load | 40–70% of capacity (S0: 64%) | `06_build…` | **E** | Placeholder | R11 |
| Water output | U(50, 200) L/s | `06_build…` | **E** | Placeholder (not used in the cascade) | — |
| Pump power | U(0.5, 2.0) MW | `06_build…` | **E** | Placeholder; sets power→water demand | R3 |

## 2. Dependencies (cascade)

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Edge types (power, water, access → hospital; power → water) | — | `06_build…` | B | Rinaldi et al. 2001; Nukavarapu & Durbha 2017/2020; NDMA 2016 | — |
| Supplier assignment | nearest supplier | `06_build…` | B (method) / E (instance) | Voronoi/nearest-facility method (Schneider et al. 2025; Karagiannis et al. 2019); real feeder topology not public | R12 |
| Aggregation | F = intrinsic × Π sat(d) | `twin/cascade.py` | D | Dev doc §3.3; losses treated as independent | **R13** |
| Floor power→hospital | 0.3 | `06_build…` | D | "Life support persists partially" (dev doc §3.3) | **R1** |
| Floor water→hospital | 0.5 | `06_build…` | D | Dev doc §3.3 | **R1** |
| Floor access→hospital | 0.7 | `06_build…` | D | "Staff/patients still trickle in" | **R1** |
| Floor access→other; power→water | 0.5; 0.0 | `06_build…` | D | Access not used for non-hospitals (twin-v2); pumps stop | — |
| Hospital power demand share | 0.1 of substation | `HOSPITAL_POWER_SHARE_DEFAULT` | D | No load estimate available | **R3** |
| Hospital water demand share | 0.1 of facility | `HOSPITAL_WATER_SHARE_DEFAULT` | D | As above | **R3** |
| Power→water demand | pump power ÷ capacity | `06_build…` | E (derived from E) | Derived from placeholders | R3 |
| Backup generator buffer | 8 h | `HOSPITAL_BACKUP_GEN_HOURS` | D | Plausible fuel autonomy; unsourced | **R2** |
| Water reserve buffer | 6 h | `HOSPITAL_WATER_RESERVE_HOURS` | D | Unsourced | **R2** |
| Buffer refill rate | ½ drain rate | `twin/cascade.py` | D | Dev doc §3.5 | — |
| Background load / shed fractions | tiers 0/20/40/60% | `twin/power.py` | D | Dev doc §3.2 fixes tier 3 = 60%; tiers 1–2 interpolated | — |
| Cascade threshold | F < 0.5 while intrinsic ≥ 0.5 | `twin/simulator.py` | D | Metric definition (dev doc §3.5) | — |
| Fixed-point tolerance / max iterations | 1e-6 / 20 | `twin/cascade.py` | D | Numerical; graph is acyclic → unique solution (tested: 200 random inits, max diff 0.0) | — |

## 3. Flood hazard

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Susceptibility form | weighted sum of 5-class quantile ratings | `05_flood_susceptibility.py` | B | Shrestha et al. 2025; AlAli et al. 2023 (WLC/AHP) | — |
| Susceptibility weights | 0.341 / 0.374 / 0.285 | `05_flood_susceptibility.py` | B | Mann & Gupta 2023 (Mumbai AHP), renormalised; slope → elevation is a substitution (D) | — |
| Pervious damping | 0.7 | `PERVIOUS_DAMPENING` | D | Unsourced | — |
| Max depth at severity 1 | 2.0 m | `flood.py` `MAX_DEPTH_AT_SEVERITY_1_M` | D | Scale choice | — |
| Severity range | U(0.2, 1.0) | `scenarios/generator.py` | D | Scenario design | — |
| Footprint σ | U(800, 2500) m | `scenarios/generator.py` | D | Scaled to the 4.7 × 9 km ward | — |
| Grow / hold / recede | U(1,3) / U(4,10) / U(4,8) h | `scenarios/generator.py` | D | Scenario design | — |
| Road blockage scale | clip(depth / 0.6 m) | `ROAD_BLOCKAGE_DEPTH_SCALE_M` | D | Dev doc §4.2; DISruptionMap uses 0.27 m (B) | **R9** |
| Impassable blockage | ≥ 0.95 | `IMPASSABLE_BLOCKAGE` | D | Routing cut-off | — |
| Substation fragility | 0.1 m→0.333 … 0.6 m→1.0; 0 below 0.1 m | `flood.py` | **B** | Nukavarapu & Durbha 2020, Table 1 | **R8** |
| Hospital / water fragility | substation shape, median 0.6 m | `flood.py` | B + D | Median from the same paper (Hospital A floods ~0.6 m); shape rescaling is an assumption | **R8** |
| Failed residual | substation 0, water 0, hospital 0.3 | `FAILED_RESIDUAL` | D | Substation switched off (JRC 2019); hospital upper floors assumed usable | — |
| Damage persistence | until repaired | `flood.py` | D | Twin-v2 design (verified: no automatic facility recovery) | — |

## 4. Demand, health and mortality proxy

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Patient arrivals | Poisson, 0.02 /bed/h | `twin/demand.py` | **E** | Placeholder in the spirit of dev doc §2.3 | **R4** |
| Diurnal cycle | ±30%, peak 10:00 | `twin/demand.py` | E | Shape placeholder | — |
| Length of stay | exponential, mean 48 h | `MEAN_LENGTH_OF_STAY_HOURS` | E | Placeholder | **R5** |
| Death proxy | queued > 4 h counts as a death | `PATIENT_WAIT_DEADLINE_HOURS` | D | **Proxy, not clinical mortality** (dev doc §5.4, §3.8 item 8) | **R6** |
| In-hospital deaths from power loss | not modelled | — | — | Known gap (Chennai 2015 MIOT failure mode) | — |
| Usable beds | beds × functional level | `twin/demand.py` | D | Capacity scales with function | — |

## 5. Response resources

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Ambulances | 2 per hospital (4 total) | `N_AMBULANCES_PER_HOSPITAL` | **E** | Invented fleet size | R7 (rule-based only) |
| Emergency call rate | 2 /h × severity | `REQUEST_RATE_PER_HOUR_AT_SEVERITY_1_DEFAULT` | **E** | Invented | — |
| Travel speed | 30 km/h | `ASSUMED_SPEED_KMPH` | D | Urban assumption | — |
| Travel-time noise | none (±15% in dev doc §3.6 is **not implemented**) | — | — | — | **R14** |
| Repair rate | 0.05 per applied repair; applied once per 15-min decision (0 → 1 in about 5 h; the earlier "100 min" was wrong, corrected 2026-09-29) | `DEFAULT_REPAIR_RATE_PER_TICK` | **E** | Invented; deterministic (dev doc repair noise not implemented) | **R10** |
| Load surge | up to +50% × severity × envelope | `LOAD_SURGE_FACTOR` | E | Invented coupling | **R11** |
| Overload threshold / damage | 95% / 0.05 per tick | `twin/power.py` | D / E | Threshold from dev doc §5.6; damage rate invented | — |
| *Twin-v3 only:* Surge capacity | +20% of nominal beds (× functional level) | `SURGE_BED_FRACTION` (`twin/demand.py`) | **D** | Engineering assumption (dev doc §3.9 mechanic 2) | — |
| *Twin-v3 only:* Surge duration budget | 12 h per hospital per episode | `SURGE_MAX_HOURS` | **D** | Staff-endurance assumption | — |
| *Twin-v3 only:* Diversion share | 50% of new walk-ins redirected to the nearest accepting hospital; walk-in travel time ignored | `DIVERT_SHARE` | **D** | Assumption: a public diversion notice reaches about half of arrivals | — |
| *Twin-v3 only:* Ambulances | 2 per hospital (6 total, 3 hospitals) | `N_AMBULANCES_PER_HOSPITAL` | **E** | Same invented rate as twin-v2 | — |

## 6. Decision, reward and evaluation settings

| Parameter | Value | Code | Cat | Source / justification | R-id |
|---|---|---|---|---|---|
| Tick / episode | 5 min / 288 ticks | `simulator.py`, envs | D | Dev doc §3.1 | — |
| Decision interval | 3 ticks (15 min) | envs | D | Dev doc §5.1 | — |
| Reward coefficients | deaths ×10, cascades ×5, violations ×20, others ×1 | envs | D | Dev doc §5.4 | — |
| Reward normalisers | fitted, frozen | `configs/reward.yaml` | Derived | Rule-based on train split; energy floor = disclosed deviation | — |
| Stabilisation | all ≥ 0.9 for 12 ticks | envs | D | Dev doc §3.1 | — |
| Route flood-safety constraint | route depth < 0.4 m | `configs/constraints.yaml` | D | Dev doc §8 | — |
| Counterfactual horizon / rollouts | 72 ticks / 10 | `twin/counterfactual.py` | D | Dev doc §3.7 | — |
| P(failure) threshold | hospital F < 0.3 | `counterfactual.py` | D | Dev doc §9.1 | — |
| Autonomy gate | risk ≤ 0.2 & conf ≥ 0.8 (auto); ≤ 0.5 & ≥ 0.5 (flag) | `configs/autonomy.yaml` | D | Dev doc §9.3 | — |
| OOD router | k = 5, p < 0.05 | `routing/ood.py` | D | Dev doc §6 | — |

## Summary

- Geography and flood **inputs**: mostly **A/B/C**.
- Infrastructure **behaviour** (capacities, demand shares, floors, buffers, fleet, rates): mostly
  **D/E**.
- The pre-registered robustness study perturbs the D/E parameters that most plausibly drive the
  comparison between policies (R1–R14). Results are claims about behaviour *under this model*,
  not about Mumbai's real infrastructure.
