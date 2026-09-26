"""Shared helpers for the numbered data-pipeline scripts (dev doc §2.2).

Not a `src/udt` package — the dev doc's repo layout (§12.1) keeps
`01_get_boundary.py` ... `07_load_postgis.py` as standalone CLI scripts
under `scripts/`, not part of any importable `udt.*` package. This module
holds the logic every one of them needs (config loading, structlog setup,
provenance manifests, the BMC ArcGIS REST client) so it lives in one place
instead of seven copies.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a pipeline config YAML (dev doc §2.2: every script takes
    `--config configs/data.yaml`)."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def configure_logging() -> structlog.stdlib.BoundLogger:
    """Dev doc §12.2: structlog everywhere, no bare `print` in scripts either."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )
    return structlog.get_logger()  # type: ignore[no-any-return]


def resolve_path(path_str: str | Path) -> Path:
    """Resolve a config-declared path relative to the repo root."""
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    output_path: str | Path,
    *,
    inputs: dict[str, Any],
    row_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a `manifest.json` next to `output_path` (dev doc §2.2:
    "write a manifest.json (inputs hash, row counts, timestamp) next to
    each output for provenance"). `inputs` is a dict of {label: value} —
    for local file inputs, values are hashed; for remote sources
    (BMC/Overpass/DEM URLs), the value itself (the URL/query) is recorded
    since there's no local file to hash.
    """
    out = resolve_path(output_path)
    manifest_path = out.parent / f"{out.stem}.manifest.json"

    input_record: dict[str, Any] = {}
    for label, value in inputs.items():
        if isinstance(value, Path) and value.exists():
            input_record[label] = {"path": str(value), "sha256": _sha256_of(value)}
        else:
            input_record[label] = value

    manifest = {
        "output": str(out),
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": input_record,
        "row_count": row_count,
        "extra": extra or {},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


class BmcArcGisClient:
    """Thin client over BMC's public ArcGIS REST feature services (no auth
    needed — confirmed reachable directly). Always requests WGS84
    (`outSR=4326`) since the services' native spatial reference is UTM 43N
    (EPSG:32643), not lon/lat.

    Retries each request a few times: this sandbox's outbound HTTPS to
    various hosts (BMC included, not just the Overpass mirrors) showed
    intermittent TLS-level connection resets during development — same
    class of flakiness `overpass_query` retries around, so the same
    tolerance applies here.
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0, retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self._client = httpx.Client(timeout=timeout_s)

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2.0 * attempt)
        raise RuntimeError(f"BMC ArcGIS request failed after {self.retries} attempts: {last_error}")

    def query_geojson(
        self,
        service: str,
        layer_id: int,
        *,
        where: str = "1=1",
        out_fields: str = "*",
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{service}/FeatureServer/{layer_id}/query"
        params = {
            "where": where,
            "outFields": out_fields,
            "outSR": "4326",
            "f": "geojson",
        }
        data = self._get(url, params)
        if "error" in data:
            raise RuntimeError(f"ArcGIS error querying {service}/{layer_id}: {data['error']}")
        return data

    def count(self, service: str, layer_id: int, *, where: str = "1=1") -> int:
        url = f"{self.base_url}/{service}/FeatureServer/{layer_id}/query"
        params = {"where": where, "returnCountOnly": "true", "f": "json"}
        data = self._get(url, params)
        if "error" in data:
            raise RuntimeError(f"ArcGIS error counting {service}/{layer_id}: {data['error']}")
        return int(data["count"])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BmcArcGisClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_OVERPASS_USER_AGENT = (
    "mtp-udt-data-pipeline/0.1 (research prototype; contact adityadd2000@gmail.com)"
)
# ^ name kept for continuity with earlier commits/manifests that reference it, even though
# this pipeline no longer talks to Overpass — see the "OSM via the official REST API"
# section below. `configure_osmnx`/`overpass_query` (the Overpass-era helpers) were removed
# 2026-08-31 once every Overpass mirror tried proved unreachable or non-functional; retrieve
# them from version control if Overpass is ever worth returning to.


# ---------------------------------------------------------------------------
# OSM via the official REST API (api.openstreetmap.org/api/0.6), not Overpass.
#
# Added 2026-08-31: every third-party Overpass mirror tried (overpass-api.de,
# overpass.kumi.systems, overpass.openstreetmap.fr, overpass.osm.ch [reachable
# but Europe-only regional data], overpass.monicz.dev, z.overpass-api.de) was
# either unreachable (TLS resets) or non-functional (hangs, wrong coverage) —
# confirmed independently from two unrelated networks. The official OSM API
# (`api.openstreetmap.org`) stayed reachable and functional throughout. It's
# not a general Overpass replacement (no tag-query language, no arbitrary
# spatial filtering) — it returns every element in a bounding box, full stop
# — but that's sufficient here: fetch the ward's bbox once (its own API
# capabilities cap area at 0.25 deg^2; every ward this project uses is
# ~0.004-0.03 deg^2, comfortably under that in a single request), then filter
# by tag and clip to the ward polygon locally. `osmnx` is no longer a
# dependency of this pipeline as of this change.
# ---------------------------------------------------------------------------

OSM_API_BASE_URL = "https://api.openstreetmap.org/api/0.6"
OSM_API_MAX_AREA_DEG2 = 0.2  # official cap is 0.25 (checked via /api/capabilities); margin kept

DRIVABLE_HIGHWAY_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "living_street",
    "service",
    "road",
}


@dataclass
class OsmNode:
    id: int
    lon: float
    lat: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmWay:
    id: int
    node_refs: list[int]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmData:
    nodes: dict[int, OsmNode] = field(default_factory=dict)
    ways: dict[int, OsmWay] = field(default_factory=dict)


def _parse_osm_xml_into(xml_text: str, nodes: dict[int, OsmNode], ways: dict[int, OsmWay]) -> None:
    root = ET.fromstring(xml_text)
    for el in root.findall("node"):
        nid = int(el.attrib["id"])
        nodes[nid] = OsmNode(
            id=nid,
            lon=float(el.attrib["lon"]),
            lat=float(el.attrib["lat"]),
            tags={t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")},
        )
    for el in root.findall("way"):
        wid = int(el.attrib["id"])
        ways[wid] = OsmWay(
            id=wid,
            node_refs=[int(nd.attrib["ref"]) for nd in el.findall("nd")],
            tags={t.attrib["k"]: t.attrib["v"] for t in el.findall("tag")},
        )


def fetch_osm_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    base_url: str = OSM_API_BASE_URL,
    timeout_s: float = 60.0,
    retries: int = 3,
    max_area_deg2: float = OSM_API_MAX_AREA_DEG2,
    max_split_depth: int = 8,
) -> OsmData:
    """Fetch every OSM node/way in a bbox from the official `/map` endpoint.

    Dense urban areas (Mumbai included) hit the API's 50,000-node-per-
    request cap at areas far smaller than its 0.25 deg^2 area cap — an
    empirically-measured ~0.0006 deg^2 tile in Kurla already had ~27,000
    nodes, almost entirely buildings/POIs incidental to what this pipeline
    actually wants (roads + a few tagged points). Rather than guess a fixed
    tile size that would either waste requests (too small) or still hit
    the node cap in denser areas (too large), this adaptively quarters any
    tile the API rejects with "too many nodes" and retries, down to
    `max_area_deg2` as a hard floor / `max_split_depth` as a recursion
    guard against a pathological single point.
    """
    nodes: dict[int, OsmNode] = {}
    ways: dict[int, OsmWay] = {}
    headers = {"User-Agent": _OVERPASS_USER_AGENT}

    with httpx.Client(timeout=timeout_s, headers=headers) as client:

        def fetch_tile(w: float, s: float, e: float, n: float, depth: int) -> None:
            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    resp = client.get(f"{base_url}/map", params={"bbox": f"{w},{s},{e},{n}"})
                    if resp.status_code == 400 and "too many" in resp.text.lower():
                        if depth >= max_split_depth:
                            raise RuntimeError(
                                f"Tile ({w},{s},{e},{n}) still too dense after "
                                f"{max_split_depth} splits — OSM API: {resp.text.strip()}"
                            )
                        mid_lon, mid_lat = (w + e) / 2, (s + n) / 2
                        for sub in (
                            (w, s, mid_lon, mid_lat),
                            (mid_lon, s, e, mid_lat),
                            (w, mid_lat, mid_lon, n),
                            (mid_lon, mid_lat, e, n),
                        ):
                            fetch_tile(*sub, depth + 1)
                        return
                    resp.raise_for_status()
                    _parse_osm_xml_into(resp.text, nodes, ways)
                    return
                except (httpx.HTTPError, ET.ParseError) as exc:
                    last_error = exc
                    if attempt < retries:
                        time.sleep(2.0 * attempt)
            raise RuntimeError(f"OSM API fetch failed for tile ({w},{s},{e},{n}): {last_error}")

        area = (east - west) * (north - south)
        if area <= max_area_deg2:
            fetch_tile(west, south, east, north, depth=0)
        else:
            tile_side = math.sqrt(max_area_deg2)
            n_lon = max(1, math.ceil((east - west) / tile_side))
            n_lat = max(1, math.ceil((north - south) / tile_side))
            dlon, dlat = (east - west) / n_lon, (north - south) / n_lat
            for i in range(n_lon):
                for j in range(n_lat):
                    fetch_tile(
                        west + i * dlon,
                        south + j * dlat,
                        west + (i + 1) * dlon,
                        south + (j + 1) * dlat,
                        depth=0,
                    )

    return OsmData(nodes=nodes, ways=ways)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_road_graph(osm: OsmData, polygon: Any) -> Any:
    """Build a NetworkX `MultiDiGraph` of the drivable road network from
    raw OSM data, clipped to `polygon` (a shapely geometry) — the local
    equivalent of `osmnx.graph_from_polygon(..., network_type="drive")`,
    scoped to just what this pipeline needs (no simplification pass; that
    already happens downstream via the SCC + segment-cap logic in
    `02_extract_roads.py`)."""
    import networkx as nx
    from shapely.geometry import Point

    g: Any = nx.MultiDiGraph()
    for way in osm.ways.values():
        highway = way.tags.get("highway")
        if highway not in DRIVABLE_HIGHWAY_TYPES or way.tags.get("area") == "yes":
            continue
        oneway = way.tags.get("oneway") in ("yes", "true", "1", "-1")
        refs = [r for r in way.node_refs if r in osm.nodes]
        for u, v in zip(refs, refs[1:], strict=False):
            nu, nv = osm.nodes[u], osm.nodes[v]
            length_m = _haversine_m(nu.lon, nu.lat, nv.lon, nv.lat)
            g.add_node(u, x=nu.lon, y=nu.lat)
            g.add_node(v, x=nv.lon, y=nv.lat)
            g.add_edge(u, v, key=0, osmid=way.id, highway=highway, length=length_m)
            if not oneway:
                g.add_edge(v, u, key=0, osmid=way.id, highway=highway, length=length_m)

    # Clip to the ward polygon: an edge fetched from the (rectangular) bbox
    # request may fall outside the actual ward boundary — keep it only if
    # at least one endpoint is inside the polygon, then drop now-isolated
    # nodes outside it too.
    inside = {n: polygon.contains(Point(d["x"], d["y"])) for n, d in g.nodes(data=True)}
    edges_to_drop = [
        (u, v, k) for u, v, k in g.edges(keys=True) if not (inside.get(u) or inside.get(v))
    ]
    g.remove_edges_from(edges_to_drop)
    isolated_outside = [n for n in g.nodes if g.degree(n) == 0 and not inside.get(n)]
    g.remove_nodes_from(isolated_outside)
    return g


def extract_tagged_points(
    osm: OsmData, polygon: Any, tag_key: str, tag_values: set[str] | None = None
) -> list[dict[str, Any]]:
    """Every node or way tagged `tag_key` (optionally restricted to
    `tag_values`) whose location falls inside `polygon` — ways use their
    node-ref centroid. Returns `{"osm_id", "osm_type", "lon", "lat",
    "tags"}` dicts; not geometry-typed, since callers just need a point."""
    from shapely.geometry import Point

    results: list[dict[str, Any]] = []

    for node in osm.nodes.values():
        val = node.tags.get(tag_key)
        if (
            val
            and (tag_values is None or val in tag_values)
            and polygon.contains(Point(node.lon, node.lat))
        ):
            results.append(
                {
                    "osm_id": node.id,
                    "osm_type": "node",
                    "lon": node.lon,
                    "lat": node.lat,
                    "tags": node.tags,
                }
            )

    for way in osm.ways.values():
        val = way.tags.get(tag_key)
        if not (val and (tag_values is None or val in tag_values)):
            continue
        refs = [r for r in way.node_refs if r in osm.nodes]
        if not refs:
            continue
        lon = sum(osm.nodes[r].lon for r in refs) / len(refs)
        lat = sum(osm.nodes[r].lat for r in refs) / len(refs)
        if polygon.contains(Point(lon, lat)):
            results.append(
                {"osm_id": way.id, "osm_type": "way", "lon": lon, "lat": lat, "tags": way.tags}
            )

    return results


def extract_tagged_lines(
    osm: OsmData, polygon: Any, tag_key: str, tag_values: set[str] | None = None
) -> list[dict[str, Any]]:
    """Every way tagged `tag_key` (optionally restricted to `tag_values`)
    whose line geometry intersects `polygon`, clipped to it. Unlike
    `extract_tagged_points` (which reduces a way to its centroid — fine for
    a substation, wrong for a river), callers here need the real line
    shape, e.g. for real distance-to-drainage. Returns `{"osm_id",
    "geometry" (a clipped shapely LineString/MultiLineString), "tags"}`."""
    from shapely.geometry import LineString

    results: list[dict[str, Any]] = []
    for way in osm.ways.values():
        val = way.tags.get(tag_key)
        if not (val and (tag_values is None or val in tag_values)):
            continue
        refs = [r for r in way.node_refs if r in osm.nodes]
        if len(refs) < 2:
            continue
        line = LineString([(osm.nodes[r].lon, osm.nodes[r].lat) for r in refs])
        if not line.intersects(polygon):
            continue
        clipped = line.intersection(polygon)
        if clipped.is_empty:
            continue
        results.append({"osm_id": way.id, "geometry": clipped, "tags": way.tags})

    return results


def extract_tagged_polygons(
    osm: OsmData, polygon: Any, tag_key: str, tag_values: set[str] | None = None
) -> list[dict[str, Any]]:
    """Every closed way tagged `tag_key` (optionally restricted to
    `tag_values`) whose polygon geometry intersects `polygon`, clipped to
    it. A way counts as closed under the OSM area-way convention (first and
    last node refs match). Returns `{"osm_id", "geometry" (a clipped
    shapely Polygon/MultiPolygon), "tags"}` — used for landuse/building
    footprints, where `extract_tagged_points`'s centroid reduction and
    `extract_tagged_lines`'s open-line assumption both lose the shape an
    area coverage/impervious-surface calculation needs."""
    from shapely.geometry import Polygon

    results: list[dict[str, Any]] = []
    for way in osm.ways.values():
        val = way.tags.get(tag_key)
        if not (val and (tag_values is None or val in tag_values)):
            continue
        refs = [r for r in way.node_refs if r in osm.nodes]
        if len(refs) < 4 or refs[0] != refs[-1]:
            continue  # not a closed ring
        coords = [(osm.nodes[r].lon, osm.nodes[r].lat) for r in refs]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)  # standard fix for a self-intersecting OSM ring
        if poly.is_empty or not poly.intersects(polygon):
            continue
        clipped = poly.intersection(polygon)
        if clipped.is_empty:
            continue
        results.append({"osm_id": way.id, "geometry": clipped, "tags": way.tags})

    return results
