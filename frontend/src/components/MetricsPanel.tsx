import type { DecoderMetrics, Prediction } from "../types";
import { pct } from "../geo";

const HELP: Record<string, string> = {
  top1: "Share of GPS fixes whose matched arc equals the server's reference arc (all labelled fixes, misses count as errors).",
  f1: "Macro-F1 of the selection decision over candidate rows (selected = 1, others = 0), averaged over both classes.",
  cont: "Share of consecutive arcs in the output route that are directly connected in the road network.",
  miss: "Share of fixes whose reference arc was not among the generated candidates (an upper bound on achievable accuracy).",
  time: "Wall-clock time of the full pipeline on Laptop 1 for this trip.",
  routef1: "Length-weighted overlap between the predicted set of arcs and the reference set of arcs.",
  edit: "1 − normalised edit distance between predicted and reference arc sequences (order-aware).",
};

function Bar({ a, b, label, help }: { a?: number | null; b?: number | null; label: string; help: string }) {
  const va = a ?? 0, vb = b ?? 0;
  const delta = a != null && b != null ? b - a : null;
  return (
    <div className="cmp" title={help}>
      <div className="cmp-head">
        <span>{label}</span>
        {delta !== null && (
          <span className={`delta ${delta >= 0 ? "up" : "down"}`}>{delta >= 0 ? "+" : ""}{(delta * 100).toFixed(1)} pts</span>
        )}
      </div>
      <div className="cmp-row"><span className="cmp-lbl">Classifier</span><div className="track"><div className="fill clf" style={{ width: `${va * 100}%` }} /></div><span className="cmp-v">{pct(a)}</span></div>
      <div className="cmp-row"><span className="cmp-lbl">+ HMM</span><div className="track"><div className="fill hmm" style={{ width: `${vb * 100}%` }} /></div><span className="cmp-v">{pct(b)}</span></div>
    </div>
  );
}

function Kpi({ label, value, help, accent }: { label: string; value: string; help: string; accent?: boolean }) {
  return (
    <div className={`kpi ${accent ? "accent" : ""}`} title={help}>
      <div className="kpi-v">{value}</div>
      <div className="kpi-l">{label} <span className="info" aria-label={help}>i</span></div>
    </div>
  );
}

export default function MetricsPanel({ pred, loading }: { pred: Prediction | null; loading: boolean }) {
  if (loading && !pred) return <div className="card"><h3>Evaluation</h3><div className="skeleton" /><div className="skeleton" /></div>;
  if (!pred) return <div className="card"><h3>Evaluation</h3><p className="muted">Results appear here after a trip is processed.</p></div>;
  const m = pred.metrics;
  const c: Partial<DecoderMetrics> = m.classifier ?? {};
  const h: Partial<DecoderMetrics> = m.hmm ?? {};
  const t = pred.timings;
  const r = pred.robustness;
  return (
    <>
      <div className="card">
        <h3>Evaluation <span className="muted small">trip {pred.trip_id}</span></h3>
        {!m.available ? (
          <p className="warn-text">Metrics unavailable: {m.reason}</p>
        ) : (
          <>
            <div className="kpis">
              <Kpi label="Top-1 accuracy" value={pct(h.top1_accuracy)} help={HELP.top1} accent />
              <Kpi label="Macro-F1" value={pct(h.selection_macro_f1)} help={HELP.f1} />
              <Kpi label="Route continuity" value={pct(h.route_continuity)} help={HELP.cont} />
              <Kpi label="Candidate miss rate" value={pct(m.candidate_miss_rate, 2)} help={HELP.miss} />
              <Kpi label="Inference time" value={`${t.total_s?.toFixed(2)} s`} help={HELP.time} />
              <Kpi label="Route F1" value={pct(h.route_f1)} help={HELP.routef1} />
            </div>
            <p className="muted small">Scored on Laptop 1 against its private reference ({m.n_labelled_fixes?.toLocaleString()} labelled of {m.n_fixes?.toLocaleString()} fixes). The reference route is never sent to this UI.</p>
          </>
        )}
      </div>
      {m.available && (
        <div className="card">
          <h3>Classifier-only vs classifier + HMM</h3>
          <Bar label="Top-1 accuracy" a={c.top1_accuracy} b={h.top1_accuracy} help={HELP.top1} />
          <Bar label="Selection Macro-F1" a={c.selection_macro_f1} b={h.selection_macro_f1} help={HELP.f1} />
          <Bar label="Route continuity" a={c.route_continuity} b={h.route_continuity} help={HELP.cont} />
          <Bar label="Route F1" a={c.route_f1} b={h.route_f1} help={HELP.routef1} />
          <Bar label="Sequence edit similarity" a={c.edit_similarity} b={h.edit_similarity} help={HELP.edit} />
          <p className="muted small">
            Invalid transitions: {c.invalid_transitions} → {h.invalid_transitions}. Classifier Macro-F1 at threshold 0.5: {pct(c.macro_f1_threshold_0_5)}, log loss {c.log_loss?.toFixed(4) ?? "n/a"}.
          </p>
        </div>
      )}
      <div className="card">
        <h3>Robustness</h3>
        <ul className="flags">
          <li className={r.gps_gaps.length ? "flag-bad" : ""}>GPS gaps &gt; 30 s: <b>{r.gps_gaps.length}</b></li>
          <li className={r.chain_breaks.length ? "flag-warn" : ""}>HMM chain breaks: <b>{r.chain_breaks.length}</b>
            {r.chain_breaks.length > 0 && <span className="muted"> ({[...new Set(r.chain_breaks.map((b) => b.reason.replace(/_/g, " ")))].join(", ")})</span>}</li>
          <li>Fixes without candidates: <b>{r.no_candidate_fixes.length}</b></li>
          <li className={r.low_confidence_fixes.length ? "flag-warn" : ""}>Low-confidence fixes (p &lt; 0.5): <b>{r.low_confidence_fixes.length}</b></li>
          <li>Unfilled route gaps: <b>{r.unfilled_route_gaps}</b></li>
          <li>Dropped invalid / duplicate fixes: <b>{r.n_dropped_invalid} / {r.n_dropped_duplicate}</b></li>
        </ul>
      </div>
      <div className="card">
        <h3>Pipeline timing</h3>
        <ul className="timings">
          {[["preprocess_s", "Cleaning + projection"], ["candidates_s", "Candidate generation"], ["features_s", "Features"],
            ["classifier_s", "Classifier"], ["hmm_s", "HMM / Viterbi"], ["reconstruction_s", "Route reconstruction"]].map(([k, l]) => (
            <li key={k}><span>{l}</span><span className="mono">{(t[k] ?? 0).toFixed(3)} s</span></li>
          ))}
        </ul>
        <p className="muted small">{pred.diagnostics.n_candidates?.toLocaleString()} candidate rows · {pred.diagnostics.n_chains} chain(s) · UTM EPSG:{pred.diagnostics.utm_epsg}</p>
      </div>
    </>
  );
}
