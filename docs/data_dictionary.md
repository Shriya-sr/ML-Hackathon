# Data Dictionary

## Raw files (per trip folder `<id>/`)

| File | Columns (whitespace, no header) | Type | Notes |
|---|---|---|---|
| `<id>.track` | longitude, latitude, timestamp_s | float, float, float | one GPS fix per row, nominally 1 Hz |
| `<id>.nodes` | longitude, latitude | float, float | **node_id = 0-based row index** |
| `<id>.arcs` | from_node_id, to_node_id | int, int | **arc_id = 0-based row index**; directed; most roads appear in both directions |
| `<id>.route` | arc_id | int | ordered reference arcs; **not** one row per fix |

Verified on all 90 trips by `scripts/run_eda.py` (see `reports/eda/data_quality_report.md`).

## Processed tables (`data/processed/`, rebuilt by `scripts/build_features.py`)

### `<partition>_obs.pkl`: one row per cleaned GPS fix

| Column | Meaning |
|---|---|
| trip_id | trip folder name |
| obs | index of the fix after cleaning (0..N-1) |
| orig_index | row in the raw `.track` file |
| t, lon, lat | timestamp and coordinates |
| x, y | UTM metres (trip's own zone) |
| label | reference arc id from the monotone alignment, −1 if > 60 m away (server-side/training only) |
| route_pos | aligned position in `.route` |
| label_dist_m | distance fix → aligned arc |
| has_candidates | at least one candidate generated |
| true_in_candidates | reference arc is among the candidates (complement = candidate miss) |

### `<partition>_candidates.pkl`: one row per (fix, candidate arc)

Keys: `trip_id, obs, arc`; supervision: `label` (0/1), `true_arc`; 21 model features:

| Feature | Definition |
|---|---|
| dist_m | perpendicular distance fix → segment (m) |
| dist_rank | 0 = nearest candidate of this fix |
| dist_minus_min | dist_m − nearest candidate's distance |
| dist_ratio | (dist_m + 1) / (nearest + 1) |
| n_candidates | candidates of this fix |
| proj_frac | projection position along the arc, clipped to [0, 1] |
| at_endpoint | 1 if the projection falls beyond the segment (clipped) |
| endpoint_dist_m | distance from the projection to the nearer arc end |
| arc_length_m | segment length |
| cos_heading | cos(angle between motion over fixes i±2 and arc direction); 0 if heading unreliable |
| heading_diff_deg | same angle in degrees; 90 if unreliable |
| heading_valid | motion over i±2 ≥ 3 m |
| cos_long_bearing | as cos_heading with an i±5 baseline (≥ 5 m) |
| win_mean_dist_m / win_max_dist_m | mean / max distance of fixes i−3..i+3 to this arc |
| speed_mps | displacement over i±2 / elapsed time |
| dt_prev / dt_next | time to previous / next fix (s) |
| from_in_degree | arcs entering the arc's start node |
| to_out_degree | arcs leaving the arc's end node |
| is_two_way | the reverse arc exists |
