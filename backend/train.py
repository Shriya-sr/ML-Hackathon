from __future__ import annotations

import argparse
from pathlib import Path
import json
from .data import discover_trips, partition_name
from .modeling import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bayes-trials", type=int, default=8)
    args = parser.parse_args()
    trips = discover_trips(args.data_dir)
    for trip in trips:
        trip.partition = partition_name(trip.partition)
    comp, trials, evaluation = train(trips, args.output_dir, args.results_dir, args.bayes_trials)
    (Path(args.results_dir) / "training_summary.json").write_text(json.dumps({"comparison": comp, "best_trials": trials}, indent=2))
    print(evaluation.groupby("method")[["top1_accuracy", "macro_f1", "route_continuity", "candidate_miss_rate"]].mean().to_string())


if __name__ == "__main__":
    main()
