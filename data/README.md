# Data directory

## Ward selection (dev doc §2.1)

**Chosen ward: L Ward (Kurla)** (BMC code `L`)

Scoring, `scripts/00_score_wards.py` against BMC's public ArcGIS REST feature services (boundary + hospitals) and OpenStreetMap's official REST API (roads + substations, since BMC has no public power-grid layer):

| Ward | Boundary | Hospitals | Road edges | Substations | Meets >=2 hospitals |
|---|---|---|---|---|---|
| L Ward (Kurla) | yes | 2 | 6506 | 3 | **YES** |
| H/E Ward (Bandra East) | yes | 0 | 3117 | 2 | no |

Rule (dev doc §2.1): pick the ward with >= 2 hospitals and best OSM completeness (road edges + weighted substation count). Hospital count = BMC `Health_Facilities` Main + Special + Peripheral + Polyclinic layers filtered to this ward's code; dispensaries/maternity homes/health posts excluded (not hospitals in the dev doc's sense). Substation/road counts are OSM-only since Mumbai power infrastructure has no public feed anywhere (checked; see dev doc §2).

**Session note:** the active ward was briefly switched to S Ward (Powai) mid-session by direct
user instruction, then switched back to L (Kurla) above — see `MTP_Module_Planner.md`'s session
log if that history matters later.

## Full pipeline status (as of 2026-08-31, water facilities revised 2026-09-13, drainage/landuse/responders/transit added 2026-09-18–19)

`01_get_boundary.py`, `02_extract_roads.py`, `03_extract_facilities.py --ward L`,
`03b_extract_drainage.py`, `03c_extract_responders.py --ward L`, `03d_extract_transit.py`,
`03e_extract_landuse.py`, `04_get_dem.py`, `05_flood_susceptibility.py`, and
`06_build_dependency_graph.py` have all been run successfully against real data for L-ward
(Kurla) — `dependency_graph.json` is real, not the `tests/fixtures/sample_graph.py` placeholder.
Notable real-data findings, not bugs:
- **Drainage — real data found and wired in (2026-09-18).** `05_flood_susceptibility.py`'s
  "distance to drainage" term used to be a DEM-only proxy (bottom decile of elevation cells
  treated as if they were drainage), kept even after OSM access was restored below on the
  (incorrect, as it turned out) assumption it avoided an OSM dependency for that one stage.
  `03b_extract_drainage.py` fetches the same OSM bbox already used for roads/facilities and
  finds **53 real drainage features in the Kurla ward** (42 `drain`, 5 `stream`, 4 `river`
  including the real Mithi River, 2 `canal`) — genuinely mapped, unlike the water-facilities gap
  below. `05_flood_susceptibility.py` now measures real distance-to-drainage from
  `data/processed/drainage.geojson` instead of the elevation proxy (mean susceptibility shifted
  0.698 → 0.600 on the same DEM). See `MTP_Development_Document.md` §2.4 and
  `MTP_Module_Planner.md`'s M2 row for the full trail.
- **Land use / land cover — real data added as a third flood-susceptibility term (2026-09-19).**
  `03e_extract_landuse.py` pulls OSM `landuse=*` + `building=*` polygons from the same bbox fetch
  (284 landuse polygons + 2,922 buildings in Kurla). `05_flood_susceptibility.py` now damps
  pervious cells (grass/forest/farmland/...) by a disclosed `PERVIOUS_DAMPENING=0.7` constant
  relative to impervious (paved/built) cells, which keep full susceptibility — mean susceptibility
  shifted 0.600 → 0.460 on the same DEM+drainage inputs.
- **Emergency responders + transit — real data, deliberately data-only (2026-09-19).**
  `03c_extract_responders.py` (BMC ArcGIS `Fire_Station`/`Police_Stations`, `WARD`-filterable: 1
  fire station + 5 police stations in Kurla) and `03d_extract_transit.py` (BMC ArcGIS suburban
  rail + metro, spatially clipped since these layers have no ward field: 3 suburban stations, 5
  line segments, 2 metro stations, 1 metro line) write `responders.geojson`/`transit.geojson` —
  real, but **explicitly not** added to the cascade dependency graph or `AssetType`; that would
  shift the RL action/observation space shapes the same way the water-facilities fix did, and was
  scoped out for now rather than done partially (user's explicit choice). Checked and rejected as
  not usable: BMC's 70k-record historical incident log, hoped to hold real waterlogging-hotspot
  data — only 3 "Water Logging" records exist city-wide, none in Kurla.
- **1 substation** in the capped 100-road testbed's OSM data for Kurla — Mumbai power
  infrastructure is genuinely sparse in OSM tagging (both hospitals get a power edge to the
  single substation found).
- **Water facilities — real-data gap found, then closed with a disclosed synthetic fix
  (2026-09-13).** OSM has zero tagged water infrastructure anywhere in this ward (re-confirmed
  live against `overpass.osm.ch` while investigating this), and BMC's own City Development Plan
  2005-2025 doesn't geocode anything ward-local either — it names only two citywide treated-water
  reservoirs (Bhandup Complex, Yewai) without saying which serves Kurla, and the one "Kurla
  pumping station" it does name (built 1955) is sewerage infrastructure, not water supply, wrong
  asset semantics to reuse. A real, ward-local, precisely-geocoded water-supply facility for
  Kurla is genuinely not publicly obtainable — same conclusion the dev doc §2.4 already reached
  for the power grid. Wiring in the real-but-15km-away Bhandup Complex instead was considered and
  rejected: it sits outside this ward's flood-susceptibility raster and isn't fed by the local
  substation, so it would never degrade during a local Kurla flood — geographically real, but
  functionally inert for the exact substation->water->hospital cascade chain (dev doc §2.2's own
  "main cascade chain") this was meant to restore. **Resolved:** `data/manual_facilities.yaml`
  adds 2 flagged-synthetic local water facilities (`W_synth_0`/`W_synth_1`, one near each real
  hospital, `provenance: synthetic_placeholder` in both the YAML and the resulting `Asset.
  attributes`) — same disclosure tier as this graph's other placeholder numbers (hospital bed
  counts, substation capacity_mw, ambulance fleet size). The cascade now genuinely fires: a
  severity-0.82 flood run (`scripts/run_flood_demo.py --seed 0`) drives both hospitals, the
  substation, and both water facilities to functional_level 0.0 at peak and back to 1.0 on full
  recession. See `MTP_Development_Document.md` §2.4 and `MTP_Module_Planner.md`'s M2 row for the
  full research trail (what was checked, what was rejected, and why).
- The degree-based road segment cap (top ~100 of ~6,500+ roads by connectivity) initially missed
  both hospitals' actual nearby roads entirely — fixed in `06_build_dependency_graph.py` with a
  fallback to the full uncapped network (`roads_full.graphml`) so every real facility keeps at
  least one access edge (`RF_*`-prefixed road assets in the graph mark these fallback additions).
- OSM data is now fetched from the **official OSM REST API** (`api.openstreetmap.org`), not
  Overpass — every Overpass mirror tried was unreachable or non-functional on 2026-08-31 (see
  `scripts/_pipeline_common.py`'s "OSM via the official REST API" section and
  `MTP_Development_Document.md` §2.4 for the full story). `osmnx` is no longer a dependency.
