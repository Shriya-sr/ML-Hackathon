import type { TripSummary } from "../types";

interface Props {
  trips: TripSummary[];
  selected: string | null;
  loading: boolean;
  onSelect: (id: string) => void;
  error: string | null;
  conn: string;
  onRetry: () => void;
}

const fmtDur = (s: number) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 60)} min`);

export default function TripList({ trips, selected, loading, onSelect, error, conn, onRetry }: Props) {
  return (
    <div className="trips">
      <h3>Evaluation trips</h3>
      {conn === "offline" && (
        <div className="empty" role="alert">
          <p>{error ?? "The API is not reachable."}</p>
          <p className="muted">Check that Laptop 1 runs the backend and that both laptops are on the same network.</p>
          <button className="btn" onClick={onRetry}>Retry connection</button>
        </div>
      )}
      {conn !== "offline" && trips.length === 0 && <div className="empty muted">No trips available on the server.</div>}
      <ul>
        {trips.map((t, i) => (
          <li key={t.trip_id}>
            <button
              className={`trip ${selected === t.trip_id ? "active" : ""}`}
              onClick={() => onSelect(t.trip_id)}
              disabled={loading}
              aria-pressed={selected === t.trip_id}
            >
              <span className="trip-n">{i + 1}</span>
              <span className="trip-main">
                <span className="trip-id">Trip {t.trip_id}</span>
                <span className="trip-meta">
                  {t.n_fixes.toLocaleString()} fixes · {fmtDur(t.duration_s)}
                </span>
              </span>
              <span className={`trip-eval ${t.evaluation_available ? "yes" : "no"}`}
                title={t.evaluation_available ? "Server can score this trip (reference kept server-side)" : "No reference on server: metrics unavailable"}>
                {t.evaluation_available ? "scored" : "no ref"}
              </span>
              {loading && selected === t.trip_id && <span className="spinner" aria-label="running" />}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
