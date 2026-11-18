export interface Health {
  status: string;
  model_loaded: boolean;
  model_error?: string | null;
  mode: string;
  n_trips: number;
  uptime_s: number;
}

export interface SystemInfo {
  mode: string;
  partition: string;
  n_trips: number;
  api_version: string;
  note: string;
}

export interface ModelInfo {
  model: string | null;
  model_family: string | null;
  feature_version: string | null;
  n_features: number;
  trained_at: string | null;
  git_commit: string | null;
  cv_macro_f1: number | null;
  student_test_top1: number | null;
  hmm_params: Record<string, number>;
}

export interface TripSummary {
  trip_id: string;
  n_fixes: number;
  duration_s: number;
  center: [number, number];
  evaluation_available: boolean;
}

export type LatLng = [number, number];

export interface DecoderMetrics {
  top1_accuracy: number | null;
  selection_macro_f1: number | null;
  route_continuity: number;
  route_f1: number;
  edit_similarity: number;
  invalid_transitions: number;
  macro_f1_threshold_0_5?: number | null;
  log_loss?: number | null;
  route_precision?: number;
  route_recall?: number;
  lcs_recall?: number;
  missing_segments?: number;
  extra_segments?: number;
}

export interface Metrics {
  available: boolean;
  reason?: string;
  n_fixes?: number;
  n_labelled_fixes?: number;
  candidate_miss_rate?: number;
  classifier?: DecoderMetrics;
  hmm?: DecoderMetrics;
}

export interface Prediction {
  trip_id: string;
  mode: string;
  model: Partial<ModelInfo>;
  n_fixes: number;
  gps: { lat: number[]; lon: number[]; t: number[] };
  fixes: {
    hmm_arc: number[];
    hmm_prob: (number | null)[];
    clf_arc: number[];
    clf_prob: (number | null)[];
    chain: number[];
    orig_index: number[];
  };
  arcs: Record<string, [LatLng, LatLng]>;
  network: [LatLng, LatLng][];
  route: { hmm: number[]; hmm_inferred: boolean[]; classifier: number[] };
  robustness: {
    gps_gaps: { index: number; dt_s: number }[];
    chain_breaks: { index: number; reason: string }[];
    no_candidate_fixes: number[];
    low_confidence_fixes: number[];
    unfilled_route_gaps: number;
    n_dropped_invalid: number;
    n_dropped_duplicate: number;
    was_resorted: boolean;
  };
  metrics: Metrics;
  timings: Record<string, number>;
  diagnostics: Record<string, number>;
}

export interface Explanation {
  trip_id: string;
  fix_index: number;
  timestamp: number;
  hmm_arc: number;
  classifier_arc: number;
  candidates: {
    arc: number;
    probability: number;
    chosen_by_hmm: boolean;
    chosen_by_classifier: boolean;
    features: Record<string, number>;
  }[];
}
