# PathSnap work handoff

## Current state

- Work is on the `sudeeksh` branch of `Shriya-sr/ML-Hackathon`.
- The raw dataset stays outside the repository at `../Student folder/Student folder`.
- The available data contains 80 `train` trips and 10 `student_test` trips. No evaluator-test partition is present.
- `results/dataset_inventory.csv` and `results/dataset_summary.json` are generated from the supplied data, including data quality and road connectivity fields.
- No model artifact or model comparison results are currently included. An earlier experiment run had a label backtracking bug; those artifacts were removed and must not be reported.

## Implementation

- `backend/data.py`: reads trip files and removes non-finite GPS rows.
- `backend/features.py`: projects coordinates locally, aligns GPS to the ordered route, selects the eight nearest road candidates by segment geometry, and creates candidate features.
- `backend/modeling.py`: five candidate classifiers, trip-grouped cross-validation, Bayesian random-forest search, and student-test comparison for direct and decoded predictions.
- `backend/decoder.py`: candidate-segment HMM-style scoring with Viterbi decoding and route continuity.
- `backend/api.py` and `frontend/index.html`: FastAPI prediction endpoint and browser interface.
- `backend/export_maps.py`: creates an interactive Folium map and a report figure from a student-test trip.
- `scripts/build_report.py`: builds the report PDF after valid training outputs and map figure exist.

## Corrections already made

The route-alignment backpointer originally selected the earliest route position when later positions held the cumulative minimum. It now selects the latest minimum position so monotone backtracking follows the route. Candidate selection also ranks a larger pool of road-segment midpoints by exact point-to-segment distance, which prevents long arcs from being missed solely because their midpoint is far away.

A one-trip check on training trip `00000001` after these fixes produced 458 distinct aligned route arcs, GPS-to-route distances with median about 4.3 m, and a candidate miss rate of about 6.5%. This is a diagnostic for one trip only, not a final model result.

## Next steps

1. Run the full experiment again after the alignment fix:

   ```powershell
   python -m backend.train --data-dir "..\Student folder\Student folder" --output-dir artifacts --results-dir results --bayes-trials 6
   ```

2. Inspect candidate miss rates and model metrics across all 10 student-test trips. If the corrected label/candidate pipeline still has high misses, investigate before reporting model scores.
3. Generate the map and PDF:

   ```powershell
   python -m backend.export_maps --data-dir "..\Student folder\Student folder"
   python scripts/build_report.py
   ```

4. Start the local API with `uvicorn backend.api:app --host 0.0.0.0 --port 8000`. The current UI lists student-test trips for development; do not describe them as reserved evaluator trips.
5. Ask the organizers for the reserved evaluator partition before final demo integration. Keep its labels server-side and out of model selection.
6. Capture real UI/API screenshots during the two-laptop demo and add them to the report.

## Team work plan

- Sudeeksh: integration, training, evaluation, and release coordination.
- Shubashitha Gowtham: audit dataset quality and validate route/candidate alignment assumptions.
- Shriya: review and own the API/UI integration.
- Sujal Yadav: review and own the decoder and continuity analysis.

Each teammate should review and commit their own changes with their own Git identity. The current implementation has not been authored by them, so do not attribute its Git commits to them.
