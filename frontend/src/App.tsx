import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, loadApiUrl, saveApiUrl } from "./api";
import type { Explanation, Health, ModelInfo, Prediction, SystemInfo, TripSummary } from "./types";
import TopBar from "./components/TopBar";
import TripList from "./components/TripList";
import MapView from "./components/MapView";
import MetricsPanel from "./components/MetricsPanel";
import ExplainPanel from "./components/ExplainPanel";

type Conn = "checking" | "online" | "offline";

export default function App() {
  const [apiUrl, setApiUrl] = useState<string>(loadApiUrl());
  const [conn, setConn] = useState<Conn>("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [trips, setTrips] = useState<TripSummary[]>([]);
  const [tripsError, setTripsError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explain, setExplain] = useState<Explanation | null>(null);
  const [explainErr, setExplainErr] = useState<string | null>(null);
  const reqId = useRef(0);

  const connect = useCallback(async (base: string) => {
    setConn("checking");
    setTripsError(null);
    try {
      const h = await api.health(base);
      setHealth(h);
      setConn("online");
      const [s, t] = await Promise.all([api.system(base), api.trips(base)]);
      setSystem(s);
      setTrips(t);
      try {
        setModel(await api.model(base));
      } catch {
        setModel(null);
      }
    } catch (e) {
      setConn("offline");
      setHealth(null);
      setTripsError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    connect(apiUrl);
  }, [apiUrl, connect]);

  // heartbeat: detect a backend that goes away (or comes back) during the demo
  const connRef = useRef<Conn>(conn);
  useEffect(() => {
    connRef.current = conn;
  }, [conn]);
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const h = await api.health(apiUrl);
        setHealth(h);
        if (connRef.current === "offline") connect(apiUrl);
        else setConn("online");
      } catch {
        setConn("offline");
      }
    }, 10000);
    return () => clearInterval(id);
  }, [apiUrl, connect]);

  const runPrediction = useCallback(
    async (tripId: string) => {
      const my = ++reqId.current;
      setSelected(tripId);
      setLoading(true);
      setError(null);
      setExplain(null);
      setExplainErr(null);
      try {
        const p = await api.predict(apiUrl, tripId);
        if (my === reqId.current) setPred(p);
      } catch (e) {
        if (my !== reqId.current) return;
        setPred(null);
        const err = e as ApiError;
        setError(err.message);
        if (err.status === null) setConn("offline");
      } finally {
        if (my === reqId.current) setLoading(false);
      }
    },
    [apiUrl],
  );

  // optional deep link: ?trip=<id> runs that trip once the trip list has loaded
  const autoRan = useRef(false);
  useEffect(() => {
    if (autoRan.current || trips.length === 0) return;
    const wanted = new URLSearchParams(window.location.search).get("trip");
    if (wanted && trips.some((t) => t.trip_id === wanted)) {
      autoRan.current = true;
      runPrediction(wanted);
    }
  }, [trips, runPrediction]);

  const onFix = useCallback(
    async (fix: number) => {
      if (!pred) return;
      setExplainErr(null);
      try {
        setExplain(await api.explain(apiUrl, pred.trip_id, fix));
      } catch (e) {
        setExplain(null);
        setExplainErr((e as Error).message);
      }
    },
    [apiUrl, pred],
  );

  const changeUrl = (u: string) => {
    const clean = u.trim().replace(/\/$/, "");
    saveApiUrl(clean);
    setApiUrl(clean);
    setPred(null);
    setSelected(null);
  };

  return (
    <div className="app">
      <TopBar
        apiUrl={apiUrl}
        onChangeUrl={changeUrl}
        conn={conn}
        health={health}
        system={system}
        model={model}
        onRetry={() => connect(apiUrl)}
      />
      {system?.mode === "DEVELOPMENT" && (
        <div className="banner" role="status">
          Development mode: the official evaluator trips are not on the server yet, so the student-testing trips are
          listed. Drop the reserved trips into <code>data/raw/evaluator_test</code> on Laptop 1 and restart the API.
        </div>
      )}
      <main className="layout">
        <aside className="sidebar">
          <TripList
            trips={trips}
            selected={selected}
            loading={loading}
            onSelect={runPrediction}
            error={tripsError}
            conn={conn}
            onRetry={() => connect(apiUrl)}
          />
        </aside>
        <section className="mapwrap">
          <MapView pred={pred} loading={loading} onFix={onFix} />
          {error && (
            <div className="map-error" role="alert">
              <strong>Prediction failed.</strong> {error}
              {selected && (
                <button className="btn small" onClick={() => runPrediction(selected)}>
                  Retry
                </button>
              )}
            </div>
          )}
          {!pred && !loading && !error && (
            <div className="map-empty">
              <h2>Select a trip</h2>
              <p>
                The API on Laptop 1 runs the full pipeline (cleaning, candidates, features, classifier, HMM/Viterbi)
                and scores it against its private reference data.
              </p>
            </div>
          )}
        </section>
        <aside className="panel">
          <MetricsPanel pred={pred} loading={loading} />
          <ExplainPanel explanation={explain} error={explainErr} hasPrediction={!!pred} />
        </aside>
      </main>
    </div>
  );
}
