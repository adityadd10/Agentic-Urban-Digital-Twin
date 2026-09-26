"""Hand-built test graph (dev doc Module Planner M1 row: "hand-built
10-asset test graph"). 10 assets: 2 substations, 1 water facility, 2
hospitals, 5 roads — covers the substation -> water -> hospital cascade
chain (dev doc §2.2) and the access/multi-road case `cascade.py`'s module
docstring documents (H1 has two nearby roads, so a single flooded one
shouldn't fully cut it off).

Coordinates sit inside the real Kurla (L-ward) boundary this prototype
fetched from BMC (`data/processed/ward_boundary.geojson`: lon
72.86-72.91, lat 19.05-19.13) so this fixture can be driven by the real
DEM-derived susceptibility raster (`data/processed/flood_susceptibility.tif`)
for the demo script — but the facility/road *positions* themselves are
illustrative placeholders, not the real OSM-derived topology
(`scripts/02_extract_roads.py`/`03_extract_facilities.py` produce that,
pending OSM access — see MTP_Module_Planner.md's session log).
"""

from __future__ import annotations

from udt.common.models import Asset, AssetType, DependencyEdge, DependencyGraph


def _point(lon: float, lat: float) -> dict:
    return {"type": "Point", "coordinates": [lon, lat]}


def _line(lon1: float, lat1: float, lon2: float, lat2: float) -> dict:
    return {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}


def build_sample_dependency_graph() -> DependencyGraph:
    assets = [
        Asset(
            asset_id="S1",
            asset_type=AssetType.SUBSTATION,
            geometry=_point(72.878, 19.070),
            attributes={"capacity_mw": 35.0, "load_mw": 20.0, "shed_tier": 0},
        ),
        Asset(
            asset_id="S2",
            asset_type=AssetType.SUBSTATION,
            geometry=_point(72.895, 19.090),
            attributes={"capacity_mw": 30.0, "load_mw": 18.0, "shed_tier": 0},
        ),
        Asset(
            asset_id="W1",
            asset_type=AssetType.WATER,
            geometry=_point(72.882, 19.075),
            attributes={"output_lps": 120.0, "reservoir_hours": 4.0, "pump_power_mw": 1.2},
        ),
        Asset(
            asset_id="H1",
            asset_type=AssetType.HOSPITAL,
            geometry=_point(72.880, 19.072),
            attributes={
                "beds_total": 150,
                "beds_occupied": 110,
                "icu_total": 15,
                "icu_occupied": 8,
                "backup_gen_hours": 8.0,
                "water_reserve_hours": 6.0,
                "patient_queue": 0,
            },
        ),
        Asset(
            asset_id="H2",
            asset_type=AssetType.HOSPITAL,
            geometry=_point(72.897, 19.092),
            attributes={
                "beds_total": 200,
                "beds_occupied": 150,
                "icu_total": 20,
                "icu_occupied": 12,
                "backup_gen_hours": 8.0,
                "water_reserve_hours": 6.0,
                "patient_queue": 0,
            },
        ),
        Asset(
            asset_id="R1",
            asset_type=AssetType.ROAD,
            geometry=_line(72.879, 19.071, 72.880, 19.072),
            attributes={
                "length_m": 150.0,
                "base_travel_min": 2.0,
                "blockage": 0.0,
                "flood_depth_m": 0.0,
            },
        ),
        Asset(
            asset_id="R2",
            asset_type=AssetType.ROAD,
            geometry=_line(72.8805, 19.0715, 72.880, 19.072),
            attributes={
                "length_m": 90.0,
                "base_travel_min": 1.5,
                "blockage": 0.0,
                "flood_depth_m": 0.0,
            },
        ),
        Asset(
            asset_id="R3",
            asset_type=AssetType.ROAD,
            geometry=_line(72.896, 19.091, 72.897, 19.092),
            attributes={
                "length_m": 130.0,
                "base_travel_min": 2.0,
                "blockage": 0.0,
                "flood_depth_m": 0.0,
            },
        ),
        Asset(
            asset_id="R4",
            asset_type=AssetType.ROAD,
            geometry=_line(72.878, 19.070, 72.882, 19.075),
            attributes={
                "length_m": 650.0,
                "base_travel_min": 6.0,
                "blockage": 0.0,
                "flood_depth_m": 0.0,
            },
        ),
        Asset(
            asset_id="R5",
            asset_type=AssetType.ROAD,
            geometry=_line(72.882, 19.075, 72.895, 19.090),
            attributes={
                "length_m": 2100.0,
                "base_travel_min": 18.0,
                "blockage": 0.0,
                "flood_depth_m": 0.0,
            },
        ),
    ]

    edges = [
        # power -> hospital (floor 0.3, buffer = backup_gen_hours, dev doc §3.3 exactly)
        DependencyEdge(
            edge_id="E1_power",
            supplier="S1",
            consumer="H1",
            kind="power",
            demand=0.1,
            criticality=0.9,
            buffer_hours=8.0,
            floor=0.3,
        ),
        DependencyEdge(
            edge_id="E2_power",
            supplier="S2",
            consumer="H2",
            kind="power",
            demand=0.1,
            criticality=0.9,
            buffer_hours=8.0,
            floor=0.3,
        ),
        # water -> hospital (floor 0.5, buffer = water_reserve_hours, dev doc §3.3 exactly)
        DependencyEdge(
            edge_id="E3_water",
            supplier="W1",
            consumer="H1",
            kind="water",
            demand=0.1,
            criticality=0.7,
            buffer_hours=6.0,
            floor=0.5,
        ),
        DependencyEdge(
            edge_id="E4_water",
            supplier="W1",
            consumer="H2",
            kind="water",
            demand=0.1,
            criticality=0.7,
            buffer_hours=6.0,
            floor=0.5,
        ),
        # power -> water (floor 0.0, no buffer, "pumps just stop", dev doc §3.3 exactly)
        DependencyEdge(
            edge_id="E5_power",
            supplier="S1",
            consumer="W1",
            kind="power",
            demand=1.2 / 35.0,
            criticality=0.9,
            buffer_hours=0.0,
            floor=0.0,
        ),
        # access -> H1: two roads (R1, R2) so a single blockage shouldn't fully cut H1 off
        DependencyEdge(
            edge_id="E6_access",
            supplier="R1",
            consumer="H1",
            kind="access",
            demand=1.0,
            criticality=0.6,
            buffer_hours=0.0,
            floor=0.7,
        ),
        DependencyEdge(
            edge_id="E7_access",
            supplier="R2",
            consumer="H1",
            kind="access",
            demand=1.0,
            criticality=0.6,
            buffer_hours=0.0,
            floor=0.7,
        ),
        # access -> H2: single road
        DependencyEdge(
            edge_id="E8_access",
            supplier="R3",
            consumer="H2",
            kind="access",
            demand=1.0,
            criticality=0.6,
            buffer_hours=0.0,
            floor=0.7,
        ),
    ]

    return DependencyGraph(assets=assets, edges=edges)
