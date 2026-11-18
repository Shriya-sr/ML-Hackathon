import { useState } from "react";
import type { Health, ModelInfo, SystemInfo } from "../types";

interface Props {
  apiUrl: string;
  onChangeUrl: (u: string) => void;
  conn: "checking" | "online" | "offline";
  health: Health | null;
  system: SystemInfo | null;
  model: ModelInfo | null;
  onRetry: () => void;
}

export default function TopBar({ apiUrl, onChangeUrl, conn, health, system, model, onRetry }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(apiUrl);
  const label = conn === "online" ? (health?.model_loaded ? "API online" : "API degraded") : conn === "checking" ? "Connecting" : "API offline";
  return (
    <header className="topbar">
      <div className="brand">
        <img src="/pathsnap.svg" alt="" width={28} height={28} />
        <div>
          <div className="brand-name">PathSnap</div>
          <div className="brand-sub">Map-matching &amp; route reconstruction</div>
        </div>
      </div>
      <div className="topstats">
        <span className={`pill ${conn === "online" && health?.model_loaded ? "ok" : conn === "checking" ? "wait" : "bad"}`}
          title={health?.model_error ?? undefined} data-testid="conn-pill">
          <span className="dot" /> {label}
        </span>
        {system && (
          <span className={`pill ${system.mode === "EVALUATION" ? "ok" : "warn"}`} title={system.note}>
            {system.mode}
          </span>
        )}
        {model && (
          <span className="stat" title={`trained ${model.trained_at ?? "?"}, commit ${model.git_commit ?? "?"}`}>
            Model <b>{model.model}</b> · features {model.feature_version}
          </span>
        )}
        {system && <span className="stat">Trips <b>{system.n_trips}</b></span>}
      </div>
      <div className="conn">
        {editing ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              onChangeUrl(draft);
              setEditing(false);
            }}
          >
            <input aria-label="API URL" value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus />
            <button className="btn small" type="submit">Connect</button>
          </form>
        ) : (
          <>
            <code title="Laptop 1 API address">{apiUrl}</code>
            <button className="btn small ghost" onClick={() => { setDraft(apiUrl); setEditing(true); }}>Change</button>
            {conn === "offline" && <button className="btn small" onClick={onRetry}>Retry</button>}
          </>
        )}
      </div>
    </header>
  );
}
