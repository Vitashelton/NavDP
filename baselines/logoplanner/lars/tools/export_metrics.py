#!/usr/bin/env python
"""Export LARS metrics from log files to CSV/JSON.

Usage:
    python tools/export_metrics.py --log-dir logs/ --output metrics.csv
    python tools/export_metrics.py --log-dir logs/ --output metrics.json --format json
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lars.metrics import compute_episode_metrics, compute_aggregate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Export LARS metrics")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Directory containing .jsonl log files")
    parser.add_argument("--output", type=str, default="metrics.csv",
                        help="Output file path")
    parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv",
                        help="Output format")
    parser.add_argument("--per-episode", action="store_true",
                        help="Export per-episode metrics instead of aggregate")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.per_episode:
        all_metrics = []
        for fname in sorted(os.listdir(args.log_dir)):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(args.log_dir, fname)
            m = compute_episode_metrics(fpath)
            m["episode"] = fname
            all_metrics.append(m)

        if not all_metrics:
            print("No log files found.")
            return

        if args.format == "csv":
            fieldnames = sorted(all_metrics[0].keys())
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metrics)

        else:
            with open(args.output, "w") as f:
                json.dump(all_metrics, f, indent=2)

    else:
        agg = compute_aggregate_metrics(args.log_dir)
        if args.format == "csv":
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(agg.keys()))
                writer.writeheader()
                writer.writerow(agg)
        else:
            with open(args.output, "w") as f:
                json.dump(agg, f, indent=2)

    print(f"Metrics exported to {args.output}")


if __name__ == "__main__":
    main()
