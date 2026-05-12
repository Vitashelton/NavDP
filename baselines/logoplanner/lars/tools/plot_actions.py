#!/usr/bin/env python
"""Plot action time series from LARS log files.

Shows raw_action, learned_action, final_action over time.

Usage:
    python tools/plot_actions.py --log logs/lars_log_20250101_120000.jsonl \
        --output action_plot.png
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot action time series")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to JSONL log file")
    parser.add_argument("--output", type=str, default="action_plot.png",
                        help="Output image path")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.log, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("Empty log file.")
        return

    raw = np.array([e["raw_action"] for e in entries])
    learned = np.array([e.get("learned_action", raw[i]) for i, e in enumerate(entries)])
    final = np.array([e["final_action"] for e in entries])
    risk_scores = np.array([e.get("risk_score", 0.0) for e in entries])
    safety = np.array([e.get("safety_trigger", False) for e in entries])

    t = np.arange(len(entries)) * 0.1  # assume 10Hz

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    labels = ["vx (forward)", "vy (lateral)", "omega (angular)"]
    colors = ["blue", "green", "red"]

    for i in range(3):
        ax = axes[i]
        ax.plot(t, raw[:, i], ":", color=colors[i], alpha=0.5, label="raw")
        ax.plot(t, learned[:, i], "--", color=colors[i], alpha=0.7, label="learned")
        ax.plot(t, final[:, i], "-", color=colors[i], linewidth=1.5, label="final")
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Highlight safety trigger regions
        for j in range(len(t)):
            if safety[j]:
                ax.axvline(x=t[j], color="red", alpha=0.1, linewidth=0.5)

    # Risk score
    ax = axes[3]
    ax.plot(t, risk_scores, "-", color="purple", linewidth=1.5)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="threshold")
    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    for j in range(len(t)):
        if safety[j]:
            ax.axvline(x=t[j], color="red", alpha=0.1, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
