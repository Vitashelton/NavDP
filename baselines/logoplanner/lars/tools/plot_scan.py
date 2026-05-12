#!/usr/bin/env python
"""Plot pseudo-LiDAR scans from log files.

Usage:
    python tools/plot_scan.py --log logs/lars_log_20250101_120000.jsonl \
        --step 10 --output scan_plot.png
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pseudo-LiDAR scan")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to JSONL log file")
    parser.add_argument("--step", type=int, default=0,
                        help="Step index to plot (0 = first)")
    parser.add_argument("--output", type=str, default="scan_plot.png",
                        help="Output image path")
    parser.add_argument("--max-range", type=float, default=8.0,
                        help="Max range for polar plot")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load log
    with open(args.log, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("Empty log file.")
        return

    if args.step >= len(entries):
        print(f"Step {args.step} out of range (max {len(entries)-1}), using last step")
        args.step = len(entries) - 1

    entry = entries[args.step]
    scan = np.array(entry["scan"], dtype=np.float32)

    n = len(scan)
    angles = np.linspace(-np.pi / 2, np.pi / 2, n)  # assume -90 to +90 deg

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Polar plot
    ax = axes[0]
    ax.fill_between(angles, 0, scan, alpha=0.3)
    ax.plot(angles, scan, "b-", linewidth=1.5)
    ax.axhline(y=0.3, color="r", linestyle="--", alpha=0.5, label="stop_threshold")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="slowdown_threshold")
    ax.set_title(f"Pseudo-LiDAR Scan (step {args.step})")
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Distance (m)")
    ax.set_ylim(0, args.max_range)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cartesian birds-eye
    ax = axes[1]
    xs = scan * np.cos(angles)
    ys = scan * np.sin(angles)
    ax.scatter(xs, ys, c=scan, cmap="RdYlGn", s=10, vmin=0, vmax=args.max_range)
    ax.scatter([0], [0], c="blue", marker="*", s=100, label="Robot")
    ax.set_xlim(-args.max_range, args.max_range)
    ax.set_ylim(0, args.max_range)
    ax.set_aspect("equal")
    ax.set_title("Bird's-Eye View")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
