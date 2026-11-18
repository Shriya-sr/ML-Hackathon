# PathSnap: Complete Viva Guide

How to use this guide: each concept has a **simple** explanation (say this first), a **technical** one (say this
when probed), **our example**, the **file** to open, and a likely **question → answer**. Numbers are in
`QUICK_REVISION.md`, which is generated from the experiment records, so they always match the code.

---

### 1. What is GPS map matching?
* **Simple:** GPS tells us roughly where the car was; map matching works out which road it was actually on.
* **Technical:** Given fixes z₁…z_T and a road graph, find the sequence of road segments (arcs) that best explains the fixes and forms a drivable path.
* **Ours:** For each fix we output an arc_id, plus one reconstructed drivable arc sequence for the trip.
* **File:** `src/pathsnap/inference/pipeline.py`
* **Q:** Why not just snap each fix to the nearest road? **A:** The nearest road is often a parallel street or the wrong direction. Our classifier-only baseline already beats nearest-road, and the HMM fixes most remaining jumps.

### 2. Why is ordinary GPS inaccurate?
* **Simple:** Signals bounce off buildings and get blocked, so positions wobble and sometimes jump or disappear.
* **Technical:** Multipath, atmospheric delay, satellite geometry and receiver noise give errors of metres to tens of metres. Dropouts and low sampling create gaps.
* **Ours:** Median fix-to-reference distance ≈ 4–5 m per trip, but long tails, and gaps up to ~42 h in one trip.
* **File:** `reports/eda/data_quality_report.md`
* **Q:** What did you see in the data? **A:** Parked cars producing jitter clouds off the network, long gaps, route breaks.

### 3. What is a road network?
* **Simple:** A map turned into dots (junctions/shape points) and lines (road pieces).
* **Technical:** A directed graph G = (V, E). Our arcs are straight segments between nodes. Two-way streets appear as two opposite arcs.
* **Ours:** 299 to 155k nodes per trip; most arcs have a reverse twin.
* **File:** `src/pathsnap/geography/road_network.py`
* **Q:** Why does direction matter? **A:** The label is a *directed* arc, so the model must also pick the driving direction.

### 4. Nodes and arcs?
* **Simple:** A node is a point; an arc is a one-way road piece from one node to another.
* **Technical:** `.nodes` row index = node_id (lon, lat). `.arcs` row index = arc_id (from_node, to_node).
* **Ours:** `RoadNetwork.u[arc]`, `.v[arc]` give the endpoints; `.reverse_arc[arc]` gives the opposite arc or −1.
* **Q:** Can you confuse node ids and arc ids? **A:** They are separate index spaces. Arcs index into nodes, never the reverse.

### 5. What is a trajectory?
* **Simple:** The list of GPS points of one trip, in time order.
* **Technical:** A time-ordered sequence of (lon, lat, t). Ours is ~1 Hz with irregular gaps.
* **File:** `.track` files, `data/loader.py::read_track`

### 6. How do .track, .nodes, .arcs and .route relate?
* **Simple:** `.track` is where the car seemed to be; `.nodes` + `.arcs` are the map; `.route` is the list of road pieces it actually drove.
* **Technical:** No row correspondence between `.track` and `.route` (routes are ~5× shorter). We derive a label per fix by monotone DP alignment of fixes to the route.
* **File:** `src/pathsnap/data/labeling.py`
* **Q:** How do you know which route arc a fix belongs to? **A:** We align fixes to the ordered route with DP, forcing positions not to go backwards in time and minimising distance plus a heading penalty.

### 7. What is preprocessing?
* **Simple:** Cleaning the data and converting it into a usable form.
* **Technical:** Drop invalid rows, re-sort by time, remove duplicate timestamps, project to UTM metres, flag gaps.
* **File:** `src/pathsnap/data/preprocessing.py`, `geography/projection.py`
* **Q:** Why UTM? **A:** Degrees are not metres, and a degree of longitude shrinks with latitude. UTM per trip keeps errors < 0.1%.

### 8. What is feature engineering?
* **Simple:** Turning raw data into numbers that describe each candidate so a model can judge it.
* **Technical:** 21 features in five groups: geometric, heading, context, kinematic, network.
* **File:** `src/pathsnap/features/feature_extraction.py`, `feature_schema.py`
* **Q:** Your best feature? **A:** See `permutation_importance.csv`. Distance-type features and the ±3-fix context distance lead, with heading cosine deciding direction.

### 9. Why generate candidates?
* **Simple:** There are too many roads to check them all, so we only consider those near each GPS point.
* **Technical:** STRtree query within 50 m, keep the 10 nearest, fallback nearest within 150 m.
* **File:** `src/pathsnap/features/candidate_generation.py`
* **Q:** How did you choose 50 m? **A:** From the training sweep: the miss rate flattens beyond ~50 m while the candidate count keeps growing.

### 10. What is candidate miss rate?
* **Simple:** How often the correct road was not even in our shortlist.
* **Technical:** Share of labelled fixes whose reference arc ∉ candidates. It is an upper bound on Top-1 (1 − miss rate).
* **Ours:** ~1.1% train, ~0.15% student-test.
* **Q:** Could you cheat it down? **A:** Only by inserting the true arc, which is leakage. We never do this (tested).

### 11. How does supervised classification work here?
* **Simple:** For each (point, nearby road) pair the model answers: "is this the right road?"
* **Technical:** Binary classifier on (fix, candidate) rows; label = candidate is the aligned reference arc. At inference, take the argmax probability per fix (classifier-only) or feed the probabilities to the HMM.
* **File:** `src/pathsnap/training/cross_validation.py`

### 12. Why did we choose each model?
* **Simple:** To cover different ways of learning: straight lines, probabilities, neighbours, trees, boosting, kernels, neural networks.
* **Technical:** LR (linear baseline, calibrated), Gaussian NB (generative baseline), KNN (instance-based), RF and Extra Trees (bagging ensembles), HGB (boosting), RBF-SVM (kernel), MLP (neural network).
* **File:** `src/pathsnap/models/model_factory.py`, `docs/model_explanations.md`
* **Q:** Why HGB in the end? **A:** It is tied with RF on CV within noise, has good log loss for the HMM, is ~20× faster to train (making Bayesian search feasible) and is much smaller to deploy.

### 13. What is Macro-F1?
* **Simple:** An F1 score that treats the rare "correct road" class as important as the common "wrong road" class.
* **Technical:** Average of the per-class F1 = 2PR/(P+R). It is insensitive to class imbalance, unlike accuracy.
* **File:** `evaluation/metrics.py::classification_metrics`

### 14. What is log loss?
* **Simple:** It punishes the model for being confidently wrong.
* **Technical:** −mean[y log p + (1−y) log(1−p)]. A proper scoring rule that measures probability quality.
* **Q:** Why do you care? **A:** The probabilities become HMM emissions. Over-confident wrong probabilities would mislead the decoder.

### 15. What is Top-1 accuracy?
* **Simple:** For each GPS point: did our best guess equal the true road?
* **Technical:** argmax over candidates vs aligned reference arc. Denominator = all labelled fixes, so misses count as errors.
* **File:** `evaluation/metrics.py::top1_accuracy`

### 16. What is group-aware cross-validation?
* **Simple:** We never test on a trip the model has trained on.
* **Technical:** GroupKFold(5) with group = trip_id. All rows of a trip stay in one fold.
* **Q:** What happens with a random split? **A:** Neighbouring fixes (1 s apart) are almost identical, so the scores would be inflated.

### 17. What is overfitting?
* **Simple:** Memorising the training data instead of learning the general pattern.
* **Technical:** Training score ≫ validation score. Check the gap, fold variance and test vs CV.
* **Ours:** `model_generalisation_gap.png`. Fully grown trees show the largest train/val gap, yet their validation stays high and stable.

### 18. What is underfitting?
* **Simple:** The model is too simple to capture the pattern.
* **Technical:** Training and validation scores both low and close.
* **Ours:** Gaussian NB is the clearest case; LR shows mild underfitting compared with the tree ensembles.

### 19. What is Bayesian optimisation?
* **Simple:** A smart way to try hyperparameters: learn from previous tries which settings look promising.
* **Technical:** Optuna TPE builds densities of good/bad trials and samples where good/bad is highest. A median pruner stops weak trials early. Objective = mean CV Macro-F1 over the same folds.
* **File:** `src/pathsnap/optimisation/`, `scripts/optimise.py`

### 20. What is an HMM?
* **Simple:** A model where the real state (the road) is hidden and we only see noisy clues (GPS). It uses how states follow each other to guess the hidden sequence.
* **Technical:** Markov chain over hidden states with transition probabilities, plus emission probabilities of observations given states.
* **File:** `src/pathsnap/decoding/hmm.py`

### 21. What are the hidden states?
* **Simple:** The possible roads for each GPS point.
* **Technical:** The candidate arcs of fix t (top 8 by emission), each with arc_id, projection fraction and probability.

### 22. Emission probabilities?
* **Simple:** How well a road explains one GPS point.
* **Technical:** w_clf·log P_classifier (+ optional Gaussian distance term), floored to avoid log 0.
* **File:** `decoding/emissions.py`

### 23. Transition probabilities?
* **Simple:** How believable it is to drive from road A to road B in one step.
* **Technical:** exp(−|network distance − GPS distance| / β), minus a U-turn penalty. Impossible if unreachable or faster than 55 m/s.
* **File:** `decoding/transitions.py`, `geography/road_network.py::network_distance`

### 24. What is the Viterbi algorithm?
* **Simple:** An efficient way to find the single best chain of roads without trying every combination.
* **Technical:** Dynamic programming: δ_t(s) = max_r[δ_{t−1}(r) + log T(r,s)] + log E_t(s); store argmax pointers; backtrack. O(T·K²).
* **File:** `decoding/viterbi.py`
* **Q:** What if no path exists? **A:** We start a new chain at that point and record the reason. We never invent a link.

### 25. How does route reconstruction work?
* **Simple:** After choosing roads for each point, we fill in the short road pieces between them so the route is continuous.
* **Technical:** Inside a Viterbi chain, consecutive arcs that don't touch are joined by the shortest network path (the one the transition scored). Gaps across chain breaks stay gaps.
* **File:** `decoding/hmm.py::reconstruct_route`

### 26. How is route continuity measured?
* **Simple:** The share of consecutive road pieces in our route that actually connect.
* **Technical:** mean[to_node(a_k) == from_node(a_{k+1})] over the collapsed route.
* **File:** `evaluation/route_metrics.py`

### 27. What is Folium?
* **Simple:** A Python library that makes interactive web maps.
* **Technical:** Python wrapper around Leaflet.js; we add layers (GPS, network, reference, predictions), a legend and a layer control, and save as HTML.
* **File:** `visualisation/folium_maps.py`, `scripts/generate_maps.py`, `maps/html/`

### 28. Why FastAPI?
* **Simple:** It makes it quick to build a reliable web API in Python with automatic docs.
* **Technical:** Pydantic validation, OpenAPI docs at /docs, async with thread offloading for CPU-bound inference, Uvicorn server.
* **File:** `backend/main.py`, `backend/routers/trips.py`

### 29. How does the frontend talk to the backend?
* **Simple:** The web page sends web requests to Laptop 1 and draws what comes back.
* **Technical:** `fetch` to `VITE_API_URL` (or the URL set in the UI): GET /api/trips, POST /api/predict/{id}, GET explain. Timeouts, retries and offline detection are built in. CORS is enabled on the API.
* **File:** `frontend/src/api.ts`, `App.tsx`

### 30. How does the two-laptop architecture work?
* **Simple:** Laptop 1 does the thinking (model + data + scoring); Laptop 2 only shows the results.
* **Technical:** The API binds 0.0.0.0:8000, the firewall allows TCP 8000, and the UI is configured with Laptop 1's LAN IP. Reference routes never leave Laptop 1; the console logs every request.
* **File:** `docs/deployment.md`
* **Q:** Prove nothing is hardcoded. **A:** Show the Laptop 1 log lines appearing per click and timings that differ by trip size. Open `prediction_service.py`: it calls the pipeline on the raw files.
