"""Metrics computation from LARS log files."""

import json
import os
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def compute_episode_metrics(log_path: str) -> Dict:
    """Compute metrics for a single episode log file (JSONL).

    Args:
        log_path: Path to a .jsonl log file.

    Returns:
        dict with computed metrics.
    """
    entries = _load_log(log_path)
    if not entries:
        return _empty_metrics()

    actions = np.array([e["final_action"] for e in entries])
    risk_scores = np.array([e.get("risk_score", 0.0) for e in entries])
    safety_triggers = np.array([e.get("safety_trigger", False) for e in entries])
    latencies = np.array([e.get("latency_ms", 0.0) for e in entries])

    # Terminal flags
    collision = any(e.get("collision_flag", False) for e in entries)
    timeout = any(e.get("timeout_flag", False) for e in entries)
    takeover = any(e.get("manual_takeover_flag", False) for e in entries)

    # Success: no collision, no timeout, no takeover
    success = not (collision or timeout or takeover)

    # Action smoothness (mean absolute jerk)
    if len(actions) >= 3:
        jerk = np.diff(actions, n=2, axis=0)
        smoothness = float(np.mean(np.abs(jerk)))
    else:
        smoothness = 0.0

    # Average risk score before collision (if collision occurred)
    avg_risk_before_collision = 0.0
    if collision:
        for i, e in enumerate(entries):
            if e.get("collision_flag", False):
                if i > 0:
                    avg_risk_before_collision = float(np.mean(risk_scores[:i]))
                break

    metrics = {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "manual_takeover": takeover,
        "num_steps": len(entries),
        "avg_linear_velocity": float(np.mean(np.linalg.norm(actions[:, :2], axis=1))),
        "max_linear_velocity": float(np.max(np.linalg.norm(actions[:, :2], axis=1))),
        "avg_angular_velocity": float(np.mean(np.abs(actions[:, 2]))),
        "max_angular_velocity": float(np.max(np.abs(actions[:, 2]))),
        "action_smoothness": smoothness,
        "safety_trigger_count": int(np.sum(safety_triggers)),
        "safety_trigger_ratio": float(np.mean(safety_triggers)),
        "avg_risk_score": float(np.mean(risk_scores)),
        "max_risk_score": float(np.max(risk_scores)) if len(risk_scores) > 0 else 0.0,
        "avg_risk_before_collision": avg_risk_before_collision,
        "avg_latency_ms": float(np.mean(latencies)),
        "max_latency_ms": float(np.max(latencies)) if len(latencies) > 0 else 0.0,
    }

    return metrics


def compute_aggregate_metrics(log_dir: str) -> Dict:
    """Compute aggregate metrics across all episodes in a directory.

    Args:
        log_dir: Directory containing .jsonl log files.

    Returns:
        dict with aggregate statistics.
    """
    all_metrics = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(log_dir, fname)
        m = compute_episode_metrics(fpath)
        if m["num_steps"] > 0:
            all_metrics.append(m)

    if not all_metrics:
        return {"num_episodes": 0}

    N = len(all_metrics)

    aggregate = {
        "num_episodes": N,
        "success_rate": float(np.mean([m["success"] for m in all_metrics])),
        "collision_rate": float(np.mean([m["collision"] for m in all_metrics])),
        "timeout_rate": float(np.mean([m["timeout"] for m in all_metrics])),
        "manual_takeover_rate": float(np.mean([m["manual_takeover"] for m in all_metrics])),
        "avg_navigation_steps": float(np.mean([m["num_steps"] for m in all_metrics])),
        "avg_linear_velocity": float(np.mean([m["avg_linear_velocity"] for m in all_metrics])),
        "avg_angular_velocity": float(np.mean([m["avg_angular_velocity"] for m in all_metrics])),
        "max_angular_velocity_peak": float(np.max([m["max_angular_velocity"] for m in all_metrics])),
        "avg_action_smoothness": float(np.mean([m["action_smoothness"] for m in all_metrics])),
        "avg_safety_trigger_count": float(np.mean([m["safety_trigger_count"] for m in all_metrics])),
        "avg_risk_score": float(np.mean([m["avg_risk_score"] for m in all_metrics])),
        "avg_risk_before_collision": float(
            np.mean([m["avg_risk_before_collision"] for m in all_metrics if m["collision"]])
        ),
        "avg_latency_ms": float(np.mean([m["avg_latency_ms"] for m in all_metrics])),
    }

    return aggregate


def _load_log(log_path: str) -> List[Dict]:
    """Load a JSONL log file into a list of dicts."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _empty_metrics() -> Dict:
    """Return zero-valued metrics dict."""
    return {
        "success": True,
        "collision": False,
        "timeout": False,
        "manual_takeover": False,
        "num_steps": 0,
        "avg_linear_velocity": 0.0,
        "max_linear_velocity": 0.0,
        "avg_angular_velocity": 0.0,
        "max_angular_velocity": 0.0,
        "action_smoothness": 0.0,
        "safety_trigger_count": 0,
        "safety_trigger_ratio": 0.0,
        "avg_risk_score": 0.0,
        "max_risk_score": 0.0,
        "avg_risk_before_collision": 0.0,
        "avg_latency_ms": 0.0,
        "max_latency_ms": 0.0,
    }


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*50}")
    print(f"{prefix} LARS Metrics")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}\n")
