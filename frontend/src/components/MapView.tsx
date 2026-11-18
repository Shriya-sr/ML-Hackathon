import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLng, Prediction } from "../types";
import { bearingDeg, lerp, projectOnSegment, routePolylines } from "../geo";

interface Props {
  pred: Prediction | null;
  loading: boolean;
  onFix: (i: number) => void;
}

const MAX_DOTS = 5000;
const COLORS = { gps: "#38bdf8", network: "#64748b", clf: "#f97316", hmm: "#22c55e", inferred: "#a3e635" };

function FitBounds({ bounds }: { bounds: L.LatLngBoundsExpression | null }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [30, 30] });
  }, [bounds, map]);
  return null;
}

const arrowIcon = (deg: number) =>
  L.divIcon({
    className: "arrow-icon",
    html: `<div style="transform: rotate(${deg}deg)">&#9650;</div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });

export default function MapView({ pred, loading, onFix }: Props) {
  const [layers, setLayers] = useState({ basemap: true, network: true, gps: true, clf: false, hmm: true, issues: true });
  const [snap, setSnap] = useState(0); // 0 = raw GPS, 1 = snapped onto matched arcs
  const [replay, setReplay] = useState<number | null>(null);
  const anim = useRef<number | null>(null);

  useEffect(() => {
    setSnap(0);
    setReplay(null);
  }, [pred?.trip_id]);
  useEffect(() => () => { if (anim.current) cancelAnimationFrame(anim.current); }, []);

  const data = useMemo(() => {
    if (!pred) return null;
    const n = pred.gps.lat.length;
    const stride = Math.max(1, Math.ceil(n / MAX_DOTS));
    const idx: number[] = [];
    for (let i = 0; i < n; i += stride) idx.push(i);
    const raw: LatLng[] = pred.gps.lat.map((la, i) => [la, pred.gps.lon[i]]);
    const snapped: LatLng[] = raw.map((p, i) => {
      const g = pred.arcs[String(pred.fixes.hmm_arc[i])];
      return g ? projectOnSegment(p, g[0], g[1]) : p;
    });
    const hmmLines = routePolylines(pred.route.hmm, pred.arcs);
    const inferred = pred.route.hmm
      .filter((_, k) => pred.route.hmm_inferred[k])
      .map((id) => pred.arcs[String(id)])
      .filter(Boolean) as [LatLng, LatLng][];
    const clfLines = routePolylines(pred.route.classifier, pred.arcs);
    const arrows: { pos: LatLng; deg: number }[] = [];
    const step = Math.max(1, Math.floor(pred.route.hmm.length / 40));
    for (let k = 0; k < pred.route.hmm.length; k += step) {
      const g = pred.arcs[String(pred.route.hmm[k])];
      if (g) arrows.push({ pos: lerp(g[0], g[1], 0.5), deg: bearingDeg(g[0], g[1]) });
    }
    const lats = pred.gps.lat, lons = pred.gps.lon;
    const bounds: L.LatLngBoundsExpression = [
      [Math.min(...lats), Math.min(...lons)],
      [Math.max(...lats), Math.max(...lons)],
    ];
    const low = new Set(pred.robustness.low_confidence_fixes);
    return { idx, raw, snapped, hmmLines, inferred, clfLines, arrows, bounds, low };
  }, [pred]);

  const animateSnap = () => {
    if (anim.current) cancelAnimationFrame(anim.current);
    const target = snap < 0.5 ? 1 : 0;
    const start = performance.now(), from = snap;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / 1400);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      setSnap(from + (target - from) * e);
      if (t < 1) anim.current = requestAnimationFrame(tick);
    };
    anim.current = requestAnimationFrame(tick);
  };

  const playReplay = () => {
    if (!pred || !data) return;
    if (anim.current) cancelAnimationFrame(anim.current);
    const n = data.raw.length;
    const start = performance.now();
    const dur = Math.min(15000, Math.max(5000, n * 3));
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      setReplay(Math.floor(t * (n - 1)));
      if (t < 1) anim.current = requestAnimationFrame(tick);
    };
    anim.current = requestAnimationFrame(tick);
  };

  const toggle = (k: keyof typeof layers) => setLayers((s) => ({ ...s, [k]: !s[k] }));
  const upto = replay ?? Infinity;
  const shown = data ? data.idx.filter((i) => i <= upto) : [];
  const vehicle = data && replay !== null ? data.snapped[replay] : null;

  return (
    <div className="map">
      <MapContainer center={[20, 0]} zoom={2} preferCanvas className="leaflet-root" zoomControl>
        {layers.basemap && (
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            opacity={0.55}
          />
        )}
        {data && <FitBounds bounds={data.bounds} />}
        {pred && layers.network && (
          <Polyline positions={pred.network} pathOptions={{ color: COLORS.network, weight: 1, opacity: 0.55 }} />
        )}
        {data && layers.clf && (
          <Polyline positions={data.clfLines} pathOptions={{ color: COLORS.clf, weight: 3, opacity: 0.9, dashArray: "6 6" }} />
        )}
        {data && layers.hmm && (
          <>
            <Polyline positions={data.hmmLines} pathOptions={{ color: COLORS.hmm, weight: 5, opacity: 0.85 }} />
            <Polyline positions={data.inferred} pathOptions={{ color: COLORS.inferred, weight: 5, opacity: 0.9, dashArray: "2 6" }} />
            {data.arrows.map((a, k) => (
              <Marker key={k} position={a.pos} icon={arrowIcon(a.deg)} interactive={false} />
            ))}
          </>
        )}
        {data && pred && layers.gps &&
          shown.map((i) => (
            <CircleMarker
              key={i}
              center={lerp(data.raw[i], data.snapped[i], snap)}
              radius={2.5}
              pathOptions={{
                color: data.low.has(i) ? "#facc15" : COLORS.gps,
                fillOpacity: 0.85,
                weight: 0,
                fillColor: data.low.has(i) ? "#facc15" : COLORS.gps,
              }}
              eventHandlers={{ click: () => onFix(i) }}
            >
              <Tooltip direction="top" offset={[0, -4]}>
                fix #{i} · t={pred.gps.t[i]} s<br />
                HMM arc {pred.fixes.hmm_arc[i]} (p={pred.fixes.hmm_prob[i] ?? "n/a"})<br />
                classifier arc {pred.fixes.clf_arc[i]}<br />
                <i>click to explain</i>
              </Tooltip>
            </CircleMarker>
          ))}
        {data && pred && layers.issues && (
          <>
            {pred.robustness.gps_gaps.map((g) => (
              <CircleMarker key={`g${g.index}`} center={data.raw[g.index]} radius={9}
                pathOptions={{ color: "#ef4444", weight: 2, fillOpacity: 0 }}>
                <Tooltip>GPS gap of {g.dt_s.toFixed(0)} s before fix #{g.index}</Tooltip>
              </CircleMarker>
            ))}
            {pred.robustness.chain_breaks.filter((b) => b.reason !== "no_candidates").map((b) => (
              <CircleMarker key={`b${b.index}`} center={data.raw[b.index]} radius={7}
                pathOptions={{ color: "#a855f7", weight: 2, fillOpacity: 0.2 }}>
                <Tooltip>HMM chain restarted at fix #{b.index}: {b.reason.replace(/_/g, " ")}</Tooltip>
              </CircleMarker>
            ))}
          </>
        )}
        {data && (
          <>
            <CircleMarker center={data.raw[0]} radius={8} pathOptions={{ color: "#fff", weight: 2, fillColor: "#16a34a", fillOpacity: 1 }}>
              <Tooltip permanent direction="right">Start</Tooltip>
            </CircleMarker>
            <CircleMarker center={data.raw[data.raw.length - 1]} radius={8}
              pathOptions={{ color: "#fff", weight: 2, fillColor: "#dc2626", fillOpacity: 1 }}>
              <Tooltip permanent direction="right">End</Tooltip>
            </CircleMarker>
          </>
        )}
        {vehicle && (
          <CircleMarker center={vehicle} radius={9} pathOptions={{ color: "#fff", weight: 3, fillColor: "#f43f5e", fillOpacity: 1 }} />
        )}
      </MapContainer>

      {pred && (
        <div className="map-controls">
          <div className="layer-toggles" role="group" aria-label="Map layers">
            {([
              ["basemap", "Basemap", "#94a3b8"], ["network", "Road network", COLORS.network], ["gps", "GPS fixes", COLORS.gps],
              ["clf", "Classifier-only route", COLORS.clf], ["hmm", "HMM route", COLORS.hmm], ["issues", "Robustness flags", "#ef4444"],
            ] as const).map(([k, label, c]) => (
              <label key={k} className="toggle">
                <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} />
                <span className="sw" style={{ background: c }} /> {label}
              </label>
            ))}
          </div>
          <div className="anim">
            <button className="btn small" onClick={animateSnap} title="Animate each GPS fix moving onto the arc the HMM matched it to">
              {snap < 0.5 ? "Snap to road" : "Show raw GPS"}
            </button>
            <button className="btn small ghost" onClick={playReplay}>Replay trip</button>
            {replay !== null && (
              <button className="btn small ghost" onClick={() => { if (anim.current) cancelAnimationFrame(anim.current); setReplay(null); }}>
                Reset
              </button>
            )}
          </div>
          <div className="legend">
            <span><i style={{ background: COLORS.hmm }} /> HMM reconstructed route</span>
            <span><i className="dashed" style={{ borderColor: COLORS.inferred }} /> inferred connecting arcs</span>
            <span><i className="dashed" style={{ borderColor: COLORS.clf }} /> classifier-only route</span>
            <span><i className="dot" style={{ background: COLORS.gps }} /> GPS fix</span>
            <span><i className="dot" style={{ background: "#facc15" }} /> low confidence (&lt;0.5)</span>
            <span><i className="ring" style={{ borderColor: "#ef4444" }} /> GPS gap &gt; 30 s</span>
            <span><i className="ring" style={{ borderColor: "#a855f7" }} /> HMM chain break</span>
          </div>
        </div>
      )}
      {loading && (
        <div className="map-loading" role="status">
          <span className="spinner big" />
          <div>Running the pipeline on Laptop 1…</div>
          <div className="muted">candidates → features → classifier → HMM/Viterbi → scoring</div>
        </div>
      )}
    </div>
  );
}
