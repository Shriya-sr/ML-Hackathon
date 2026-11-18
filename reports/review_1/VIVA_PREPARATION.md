# Viva Preparation: Review 1

| Component | What | Why | How | Algorithm | Files | Key decision | Limitation |
|---|---|---|---|---|---|---|---|
| Data audit | inventory + quality checks for 90 trips | must understand files before modelling | pandas parsing, validator checks, scipy components | connected components | `data/loader.py`, `data/validator.py` | report, never silently fix | evaluator partition absent |
| Projection | lon/lat → metres | degrees are not distances; world-wide trips | pyproj UTM zone of trip centroid | Transverse Mercator | `geography/projection.py` | per-trip zone | none at trip scale |
| Labels | one reference arc per fix | `.route` is a chain, not per fix | DP with prefix minima, monotone in time | dynamic programming | `data/labeling.py` | heading term, 60 m cut-off | labels approximate near junctions |
| Candidates | ≤ 10 arcs within 50 m | restrict the search to plausible roads | STRtree `dwithin` | R-tree (STR packing) | `features/candidate_generation.py` | chosen by sweep on train | 1.14% miss rate on train |
| Features | 21 numbers per (fix, arc) | describe closeness, direction, context | vectorised numpy | n/a | `features/feature_extraction.py` | relative + context features | straight-segment geometry |
| Models | 8 classifiers | brief requires ≥ 5 families | sklearn pipelines, GroupKFold | LR, NB, KNN, RF, ET, HGB, SVM, MLP | `models/model_factory.py`, `scripts/train_all.py` | same folds/features for all | KNN/SVM subsampled |
| HMM | coherent route | independent picks jump between roads | emissions = classifier, transitions = network distance | Viterbi | `decoding/*` | chain breaks instead of fake links | parameters still default at Review 1 |

**Likely Review 1 questions**

1. *Why not a random train/validation split?* Fixes 1 s apart are near-duplicates, so a random split leaks. We group by trip.
2. *How did you get per-fix labels?* Monotone alignment (DP) of fixes to the ordered route (`labeling.py`).
3. *What limits your accuracy?* Candidate misses (~1%), wrong-direction confusions when stationary, ambiguous junctions.
4. *Why does the HMM help so much on continuity?* Transitions forbid unreachable jumps and penalise detours, so
   consecutive decisions must form a drivable path.
5. *What's left?* Optimisation, HMM tuning, final artifacts, maps, report. See the Phase 2 list in `REVIEW_1_REPORT.md`.
