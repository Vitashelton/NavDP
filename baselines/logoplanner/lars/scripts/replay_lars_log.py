#!/usr/bin/env python
"""Replay a LARS log file through the runtime for debugging/visualization.

Usage:
    python replay_lars_log.py --log logs/lars_log_20250101_120000.jsonl \
        --config configs/lars.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lars.lars_runtime import LARSRuntime


def parse_args():
    parser = argparse.ArgumentParser(description="Replay LARS log for debugging")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to JSONL log file")
    parser.add_argument("--config", type=str, default="configs/lars.yaml",
                        help="Path to LARS config")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Replay speed multiplier (1.0 = real-time)")
    parser.add_argument("--print-steps", action="store_true",
                        help="Print every step's action and risk")
    return parser.parse_args()


def main():
    args = parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, args.config)

    runtime = LARSRuntime.from_config(config_path)
    runtime.reset()

    with open(args.log, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print("Empty log file.")
        return

    print(f"Replaying {len(entries)} steps from {args.log}")
    print(f"Replay speed: {args.replay_speed}x")

    prev_timestamp = None

    for i, entry in enumerate(entries):
        # Respect timing
        if prev_timestamp is not None and args.replay_speed > 0:
            dt = (entry["timestamp"] - prev_timestamp) / args.replay_speed
            if dt > 0:
                time.sleep(min(dt, 1.0))  # cap at 1s

        prev_timestamp = entry["timestamp"]

        # Reconstruct inputs from log
        policy_output = np.array(entry.get("raw_policy_output", entry.get("final_action", [0, 0, 0])))
        scan = np.array(entry["scan"], dtype=np.float32)

        # Skip depth-to-scan by directly calling residual_adapter
        final_action, info = runtime.residual_adapter(
            policy_output=policy_output,
            scan=scan,
            goal_heading=entry.get("goal_heading", 0.0),
        )

        if args.print_steps:
            print(
                f"Step {i:4d} | "
                f"raw=[{info['raw_action'][0]:+.3f},{info['raw_action'][1]:+.3f},{info['raw_action'][2]:+.3f}] "
                f"final=[{final_action[0]:+.3f},{final_action[1]:+.3f},{final_action[2]:+.3f}] "
                f"risk={info['risk_score']:.3f} "
                f"trigger={info['safety_trigger']}"
            )

    # Print summary
    from lars.metrics import compute_episode_metrics
    metrics = compute_episode_metrics(args.log)
    from lars.metrics import print_metrics
    print_metrics(metrics, prefix="Log Replay")

    runtime.close()


if __name__ == "__main__":
    main()
