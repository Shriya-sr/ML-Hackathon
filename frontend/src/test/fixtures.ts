import type { Prediction, TripSummary } from "../types";

/** Minimal shapes for component tests (UI-only; never used as evidence of ML results). */
export const trips: TripSummary[] = [
  { trip_id: "00000052", n_fixes: 262, duration_s: 270, center: [56.81, 35.87], evaluation_available: true },
  { trip_id: "00000077", n_fixes: 3911, duration_s: 3950, center: [61.66, 50.81], evaluation_available: false },
];

export const prediction: Prediction = {
  trip_id: "00000052",
  mode: "DEVELOPMENT",
  model: { model: "test-model" },
  n_fixes: 3,
  gps: { lat: [56.8, 56.8005, 56.801], lon: [35.87, 35.8705, 35.871], t: [1, 2, 3] },
  fixes: { hmm_arc: [1, 1, 2], hmm_prob: [0.9, 0.4, 0.95], clf_arc: [1, 3, 2], clf_prob: [0.9, 0.6, 0.95], chain: [0, 0, 0], orig_index: [0, 1, 2] },
  arcs: { "1": [[56.8, 35.87], [56.8006, 35.8706]], "2": [[56.8006, 35.8706], [56.8012, 35.8712]], "3": [[56.8, 35.869], [56.801, 35.869]] },
  network: [[[56.8, 35.87], [56.8006, 35.8706]]],
  route: { hmm: [1, 2], hmm_inferred: [false, false], classifier: [1, 3, 2] },
  robustness: { gps_gaps: [], chain_breaks: [], no_candidate_fixes: [], low_confidence_fixes: [1], unfilled_route_gaps: 0, n_dropped_invalid: 0, n_dropped_duplicate: 0, was_resorted: false },
  metrics: {
    available: true, n_fixes: 3, n_labelled_fixes: 3, candidate_miss_rate: 0,
    classifier: { top1_accuracy: 0.6667, selection_macro_f1: 0.7, route_continuity: 0.5, route_f1: 0.8, edit_similarity: 0.6, invalid_transitions: 1, macro_f1_threshold_0_5: 0.7, log_loss: 0.2 },
    hmm: { top1_accuracy: 1, selection_macro_f1: 1, route_continuity: 1, route_f1: 1, edit_similarity: 1, invalid_transitions: 0 },
  },
  timings: { total_s: 0.42, preprocess_s: 0.01, candidates_s: 0.02, features_s: 0.01, classifier_s: 0.05, hmm_s: 0.3, reconstruction_s: 0.01 },
  diagnostics: { n_candidates: 20, n_chains: 1, utm_epsg: 32636 },
};
