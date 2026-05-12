#!/usr/bin/env python
"""Build LARS training dataset from collected log files.

Reads JSONL log files from a directory, constructs labeled samples
using the labeling rules in risk_dataset.py, and saves a .pt file
ready for train_lars.py.

Usage:
    python build_lars_dataset.py --log-dir logs/ --output logs/lars_dataset.pt
"""

import argparse
import os
import sys

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lars.risk_dataset import build_samples_from_logs, save_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LARS training dataset from logs"
    )
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Directory containing .jsonl log files")
    parser.add_argument("--output", type=str, default="logs/lars_dataset.pt",
                        help="Output .pt file path")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Timestep between frames in seconds")
    parser.add_argument("--print-stats", action="store_true",
                        help="Print dataset statistics")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Building dataset from: {args.log_dir}")
    samples = build_samples_from_logs(args.log_dir, dt=args.dt)

    if not samples:
        print("ERROR: No samples found. Check log directory.")
        sys.exit(1)

    save_dataset(samples, args.output)

    if args.print_stats:
        N = len(samples)
        n_risk = sum(1 for s in samples if s["risk_label"] == 1)
        n_safe = N - n_risk
        residuals = np.array([s["residual_label"] for s in samples])

        print(f"\nDataset statistics:")
        print(f"  Total samples: {N}")
        print(f"  Risk-positive: {n_risk} ({100*n_risk/N:.1f}%)")
        print(f"  Risk-negative: {n_safe} ({100*n_safe/N:.1f}%)")
        print(f"  Residual mean: {residuals.mean(axis=0)}")
        print(f"  Residual std:  {residuals.std(axis=0)}")


if __name__ == "__main__":
    main()
