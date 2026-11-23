# Debugging PathSnap HMM / Viterbi Route Decoding

This guide helps contributors investigate cases where individual road-candidate scores look plausible but the final decoded route is incorrect or discontinuous.

## Background

PathSnap scores nearby road candidates for each GPS fix using a classifier and then uses Hidden Markov Model (HMM) / Viterbi decoding to select a consistent sequence. Road-network transition distances, U-turn penalties, and chain breaks for GPS gaps all affect the result. A locally high-scoring candidate is not necessarily the best candidate for the trip as a whole.

## Investigation checklist

1. **Identify the trip and fix range.** Record the trip ID and the first fix at which the decoded route differs from the expected road. Do not copy or publish evaluator reference-route data.
2. **Check GPS input quality.** Look for missing or duplicate timestamps, long time gaps, unrealistic speed, and large jumps in the raw fixes. Determine whether the decoder should have started a new segment at a gap.
3. **Check candidate coverage.** Confirm that the expected road was included among the candidates at the affected fixes. If it was not, investigate candidate generation and search radius before changing the decoder.
4. **Compare candidate emissions.** Inspect the candidate probabilities and feature values at nearby fixes. A low probability for the correct road can originate upstream from the HMM.
5. **Check road connectivity.** Confirm that proposed consecutive candidate arcs are connected in the directed road network. Look for prohibited turns, one-way restrictions, and transition distances that exceed what the speed and elapsed time support.
6. **Check penalties and chain breaks.** Determine whether a U-turn penalty, transition penalty, or break on a GPS gap explains the selected path.
7. **Compare independent and sequential predictions.** Compare classifier-only road choices with HMM-decoded choices around the failure. Record whether decoding fixes jumps or introduces a new error.
8. **Inspect map output.** Plot raw GPS fixes, candidate arcs, and the decoded route on the same map when working with data approved for development. Avoid exposing restricted reference routes.
9. **Reproduce with a minimal test.** Construct a small synthetic candidate sequence and transition matrix capturing the problem. Verify the selected Viterbi path against brute-force enumeration of all possible paths for a very small example.
10. **Run regression tests.** Execute `python -m pytest -p no:warnings` before proposing a fix. For a UI change, also run `cd frontend && npx vitest run` and `npx tsc -b`.

## Recommended bug report template

```text
Title: HMM decoder selects an unexpected road after [condition]

Environment:
- Commit / branch:
- Development trip or synthetic fixture:
- Relevant fix indices:

Expected behavior:
Actual behavior:

Observations:
- Input timestamps / GPS gaps:
- Candidate coverage:
- Classifier-only choice:
- HMM choice:
- Transition connectivity / penalties:

Reproduction steps:
1.
2.
3.

Tests run:
```

## Important safeguards

- Keep evaluator-trip reference routes server-side and do not include them in issues, screenshots, tests, or pull requests.
- Do not change reported metrics or generated figures manually. Re-run the appropriate pipeline before reporting updated results.
- Keep any regression tests deterministic; the project uses seed 42 for randomness.

Related documentation: [`methodology.md`](methodology.md), [`architecture.md`](architecture.md), and the repository's [`CONTRIBUTING.md`](../CONTRIBUTING.md).
