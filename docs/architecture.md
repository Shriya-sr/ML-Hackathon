# Architecture

```
                       ┌──────────────────────────── Laptop 1 (Model & API) ─────────────────────────────┐
 data/raw/<partition>/ │                                                                                 │
   <id>.track ─────────┼─► load_trip(with_route=False) ─► clean_track ─► UTM projection                   │
   <id>.nodes ─────────┼─► RoadNetwork (CSR graph, STRtree, reverse arcs)                                 │
   <id>.arcs  ─────────┤        │                                                                        │
                       │        ▼                                                                        │
                       │  generate_candidates (R=50 m, K=10) ─► extract_features (21, schema v1)         │
                       │        │                                                                        │
                       │        ▼                                                                        │
                       │  final_classifier.joblib ─► P(candidate is correct)                             │
                       │        │                                                                        │
                       │        ├─► classifier-only argmax (baseline)                                    │
                       │        ▼                                                                        │
                       │  HMMDecoder: emissions + bounded-Dijkstra transitions ─► segmented Viterbi      │
                       │        ▼                                                                        │
                       │  reconstruct_route (shortest-path infill inside chains)                         │
                       │        │                                                                        │
   <id>.route ─────────┼─► RouteEvaluator (server-side only) ─► aggregate metrics                        │
                       │        │                                                                        │
                       │  FastAPI  /health  /api/trips  /api/predict/{id}  /api/results/{id}[/explain/k]  │
                       └────────┼────────────────────────────────────────────────────────────────────────┘
                                │  HTTP/JSON over LAN (CORS, gzip). No reference route in any response.
                       ┌────────▼──────────── Laptop 2 (UI) ─────────────┐
                       │ React + TypeScript + Vite + react-leaflet        │
                       │ trip list · map layers · metrics · explanations │
                       └─────────────────────────────────────────────────┘
```

## Code map

| Concern | Module |
|---|---|
| raw readers, validation, cleaning, labels | `src/pathsnap/data/{loader,validator,preprocessing,labeling,dataset}.py` |
| projection, geometry, spatial index, graph | `src/pathsnap/geography/*` |
| candidates, features, schema | `src/pathsnap/features/*` |
| model zoo | `src/pathsnap/models/model_factory.py` |
| grouped CV, tracking | `src/pathsnap/training/*` |
| metrics, route metrics, decoder comparison | `src/pathsnap/evaluation/*` |
| Bayesian optimisation | `src/pathsnap/optimisation/*` |
| HMM, emissions, transitions, Viterbi | `src/pathsnap/decoding/*` |
| inference + server-side evaluator | `src/pathsnap/inference/pipeline.py` |
| plots, Folium maps | `src/pathsnap/visualisation/*` |
| API | `backend/` (`main.py`, `routers/`, `services/`, `schemas/`) |
| UI | `frontend/src/` (`App.tsx`, `components/`, `api.ts`) |
| experiment drivers | `scripts/` |

## Request lifecycle (`POST /api/predict/{trip_id}`)

1. The trip id is checked against the whitelist of folders in the active partition (404 otherwise).
2. `MapMatcher.predict` runs in a worker thread under `asyncio.wait_for` (504 on timeout).
3. `RouteEvaluator.evaluate` aligns the reference to the fixes on the server and computes metrics.
4. The payload carries GPS fixes, per-fix decisions, geometry of *predicted* arcs and nearby roads,
   robustness flags, timings and aggregate metrics. The result is cached for `/api/results` and `/explain`.
5. The access-log middleware prints `client METHOD path -> status (ms)` to the Laptop 1 console.
