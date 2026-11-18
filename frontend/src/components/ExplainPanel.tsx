import type { Explanation } from "../types";

const FEAT_LABEL: Record<string, string> = {
  dist_m: "distance (m)",
  cos_heading: "heading agreement (cos)",
  win_mean_dist_m: "±3-fix mean distance (m)",
  dist_rank: "distance rank",
  cos_long_bearing: "long-window bearing (cos)",
  speed_mps: "speed (m/s)",
};

export default function ExplainPanel({ explanation, error, hasPrediction }: { explanation: Explanation | null; error: string | null; hasPrediction: boolean }) {
  if (!hasPrediction) return null;
  return (
    <div className="card">
      <h3>Why this road?</h3>
      {error && <p className="warn-text">{error}</p>}
      {!explanation && !error && <p className="muted">Click any GPS fix on the map to see its candidate roads, the classifier's probability for each, and the features behind it.</p>}
      {explanation && (
        <>
          <p className="small">Fix <b>#{explanation.fix_index}</b> at t = {explanation.timestamp} s · {explanation.candidates.length} candidates</p>
          <table className="explain">
            <thead>
              <tr><th>arc</th><th>P(correct)</th>{Object.keys(FEAT_LABEL).slice(0, 3).map((f) => <th key={f}>{FEAT_LABEL[f]}</th>)}</tr>
            </thead>
            <tbody>
              {explanation.candidates.slice(0, 6).map((c) => (
                <tr key={c.arc} className={c.chosen_by_hmm ? "chosen" : ""}>
                  <td className="mono">{c.arc}{c.chosen_by_hmm && <span className="tag hmm">HMM</span>}{c.chosen_by_classifier && <span className="tag clf">CLF</span>}</td>
                  <td><div className="prob"><div style={{ width: `${c.probability * 100}%` }} /></div>{c.probability.toFixed(3)}</td>
                  {Object.keys(FEAT_LABEL).slice(0, 3).map((f) => <td key={f} className="mono">{c.features[f]}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Probability is the classifier's estimate before sequence decoding; the HMM may pick a lower-probability candidate when it gives a drivable route. Confidence is not a guarantee of correctness.</p>
        </>
      )}
    </div>
  );
}
