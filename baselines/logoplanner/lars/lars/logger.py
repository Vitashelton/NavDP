"""Per-timestep logger for LARS runtime."""

import json
import os
import time
from typing import Dict, Optional
import numpy as np


class LARSLogger:
    """Logs LARS runtime data as JSONL (one JSON object per line).

    Each line contains all intermediate variables for a single timestep:
    timestamp, raw_policy_output, raw_action, learned_residual, risk_score,
    learned_action, final_action, scan, sector distances, flags, latency.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        max_log_size_mb: float = 100.0,
        enabled: bool = True,
    ):
        self.log_dir = log_dir
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.enabled = enabled
        self._log_file: Optional[str] = None
        self._file_handle = None
        self._line_count = 0

        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self._rotate()

    def _rotate(self):
        """Start a new log file."""
        if self._file_handle is not None:
            self._file_handle.close()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_file = os.path.join(self.log_dir, f"lars_log_{timestamp}.jsonl")
        self._file_handle = open(self._log_file, "w")
        self._line_count = 0

    def _check_rotation(self):
        """Rotate if the current file exceeds max size."""
        if self._file_handle is not None:
            self._file_handle.flush()
            size = os.path.getsize(self._log_file)
            if size >= self.max_log_size_bytes:
                self._rotate()

    def log(
        self,
        raw_policy_output: np.ndarray,
        raw_action: np.ndarray,
        learned_residual: np.ndarray,
        risk_score: float,
        learned_action: np.ndarray,
        final_action: np.ndarray,
        scan: np.ndarray,
        front_min: float,
        left_min: float,
        right_min: float,
        safety_trigger: bool,
        collision_flag: bool = False,
        manual_takeover_flag: bool = False,
        latency_ms: float = 0.0,
        goal_heading: float = 0.0,
        extra: Optional[Dict] = None,
    ):
        """Write one timestep entry to the log.

        Args:
            raw_policy_output: Original policy output array.
            raw_action: [vx, vy, omega] after action_adapter.
            learned_residual: [dvx, dvy, domega] from risk model.
            risk_score: Predicted collision risk [0, 1].
            learned_action: raw_action + residual_scale * learned_residual.
            final_action: After safety shield.
            scan: Pseudo-LiDAR scan array.
            front_min, left_min, right_min: Sector min distances.
            safety_trigger: Whether safety shield was triggered.
            collision_flag: Whether collision occurred this frame.
            manual_takeover_flag: Whether manual takeover occurred.
            latency_ms: Total LARS processing latency in ms.
            goal_heading: Heading to goal in radians.
            extra: Optional additional fields to log.
        """
        if not self.enabled:
            return

        self._check_rotation()

        entry = {
            "timestamp": time.time(),
            "raw_policy_output": np.asarray(raw_policy_output).tolist(),
            "raw_action": np.asarray(raw_action).tolist(),
            "learned_residual": np.asarray(learned_residual).tolist(),
            "risk_score": float(risk_score),
            "learned_action": np.asarray(learned_action).tolist(),
            "final_action": np.asarray(final_action).tolist(),
            "scan": np.asarray(scan).tolist(),
            "front_min": float(front_min),
            "left_min": float(left_min),
            "right_min": float(right_min),
            "safety_trigger": bool(safety_trigger),
            "collision_flag": bool(collision_flag),
            "manual_takeover_flag": bool(manual_takeover_flag),
            "latency_ms": float(latency_ms),
            "goal_heading": float(goal_heading),
        }

        if extra:
            entry.update(extra)

        self._file_handle.write(json.dumps(entry) + "\n")
        self._line_count += 1

    def close(self):
        """Close the log file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    @property
    def current_log_path(self) -> Optional[str]:
        return self._log_file

    def __del__(self):
        self.close()
