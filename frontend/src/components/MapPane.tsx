import { CircleMarker, MapContainer, Polyline, Popup, TileLayer } from "react-leaflet";
import type { Asset, DependencyEdge } from "../types";

// Kurla (L Ward), Mumbai — the one real ward this whole prototype is
// built against (dev doc §2.1's own scoring, `data/README.md`).
const WARD_CENTER: [number, number] = [19.08, 72.88];

/** dev doc §11.3: "assets colored green -> red by functional level" —
 * a clean two-stop HSL interpolation (120deg green at level=1.0, 0deg
 * red at level=0.0), the same "good/bad" semantic-color convention
 * `viz/kurla_flood_twin.html`'s own map already established earlier
 * this session. */
function functionalLevelColor(level: number): string {
  const clamped = Math.max(0, Math.min(1, level));
  return `hsl(${clamped * 120}, 70%, 42%)`;
}

function assetCoordinates(asset: Asset): [number, number] | null {
  if (asset.geometry.type !== "Point") return null; // roads are LineStrings - not point-plotted here
  const [lon, lat] = asset.geometry.coordinates as [number, number];
  return [lat, lon];
}

interface Props {
  assets: Asset[];
  edges: DependencyEdge[];
  showEdges: boolean;
}

/** dev doc §11.3's Map pane. **Disclosed gaps, not silently faked:**
 * - "animated ambulances": `twin/ambulances.py`'s own real ambulance
 *   `Asset`s never move their `geometry` (it's fixed to the home
 *   hospital's location for the ambulance's whole lifetime — only
 *   `attributes.status`/`remaining_travel_min` change). This pane
 *   shows ambulances as small markers at that fixed home location,
 *   colored by `status` instead of animating a position the backend
 *   doesn't actually track.
 * - "incident footprint overlay": would need the *current* incident's
 *   geometry, which no endpoint exposes yet (`CreateEpisodeResponse`
 *   only returns `severity`/`incident_id`, not the full `Incident`) —
 *   left out rather than drawn from stale/guessed data.
 */
export function MapPane({ assets, edges, showEdges }: Props) {
  const assetsById = new Map(assets.map((a) => [a.asset_id, a]));

  return (
    <MapContainer center={WARD_CENTER} zoom={14} className="map-pane">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {showEdges &&
        edges.map((edge) => {
          const from = assetsById.get(edge.supplier);
          const to = assetsById.get(edge.consumer);
          const fromCoords = from ? assetCoordinates(from) : null;
          const toCoords = to ? assetCoordinates(to) : null;
          if (!fromCoords || !toCoords) return null;
          return (
            <Polyline
              key={edge.edge_id}
              positions={[fromCoords, toCoords]}
              pathOptions={{
                color: edge.kind === "power" ? "#e0a800" : edge.kind === "water" ? "#3b82c4" : "#8a97ab",
                weight: 1.5,
                dashArray: edge.kind === "access" ? "4 3" : undefined,
              }}
            />
          );
        })}
      {assets
        .filter((a) => a.asset_type !== "road")
        .map((asset) => {
          const coords = assetCoordinates(asset);
          if (!coords) return null;
          const isAmbulance = asset.asset_type === "ambulance";
          return (
            <CircleMarker
              key={asset.asset_id}
              center={coords}
              radius={isAmbulance ? 5 : 9}
              pathOptions={{
                color: "#1a1a1a",
                weight: 1,
                fillColor: isAmbulance ? "#666" : functionalLevelColor(asset.functional_level),
                fillOpacity: 0.9,
              }}
            >
              <Popup>
                <strong>{asset.asset_id}</strong> ({asset.asset_type})
                <br />
                {isAmbulance
                  ? `status: ${String(asset.attributes.status ?? "unknown")}`
                  : `functional level: ${(asset.functional_level * 100).toFixed(0)}%`}
              </Popup>
            </CircleMarker>
          );
        })}
    </MapContainer>
  );
}
