"""Hard-rule safety shield for final action filtering."""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """Configuration for the safety shield."""

    stop_threshold: float = 0.3       # front_min below this -> force stop
    side_threshold: float = 0.2       # side_min below this -> suppress lateral
    slowdown_threshold: float = 0.6   # front_min below this -> scale down vx
    min_slowdown_scale: float = 0.2   # minimum velocity scale
    max_vx: float = 0.8
    max_vy: float = 0.8
    max_omega: float = 1.0
    front_start_idx: int = 24
    front_end_idx: int = 40
    left_start_idx: int = 0
    left_end_idx: int = 23
    right_start_idx: int = 41
    right_end_idx: int = 63
    rotation_speed: float = 0.3
    prefer_rotation: bool = True


class SafetyShield:
    """Hard-rule safety layer that guarantees basic safety constraints.

    Input: action = [vx, vy, omega] and a 1D scan array.
    Output: filtered_action and trigger_info.

    Rules (applied in order):
    1. If front_min < stop_threshold: force stop (vx=vy=0), only allow slow rotation
       toward the freer side.
    2. If left_min < side_threshold: suppress vy > 0 (leftward motion).
    3. If right_min < side_threshold: suppress vy < 0 (rightward motion).
    4. If front_min < slowdown_threshold: scale down vx proportionally.
    5. Hard-clip all velocities to max limits.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.cfg = config or SafetyConfig()

    def __call__(
        self, action: np.ndarray, scan: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply safety shield to an action.

        Args:
            action: [vx, vy, omega] raw velocity command.
            scan: 1D pseudo-LiDAR scan in meters, shape (scan_dim,).

        Returns:
            filtered_action: [vx, vy, omega] after safety filtering.
            trigger_info: dict with trigger reasons and sector distances.
        """
        vx, vy, omega = float(action[0]), float(action[1]), float(action[2])

        front_min = float(np.min(scan[self.cfg.front_start_idx : self.cfg.front_end_idx + 1]))
        left_min = float(np.min(scan[self.cfg.left_start_idx : self.cfg.left_end_idx + 1]))
        right_min = float(np.min(scan[self.cfg.right_start_idx : self.cfg.right_end_idx + 1]))

        trigger_info = {
            "front_min": front_min,
            "left_min": left_min,
            "right_min": right_min,
            "stop_triggered": False,
            "side_triggered": False,
            "slowdown_triggered": False,
            "trigger_reason": "none",
        }

        # Rule 1: Emergency stop
        if front_min < self.cfg.stop_threshold:
            vx, vy = 0.0, 0.0
            if self.cfg.prefer_rotation:
                omega = self.cfg.rotation_speed if left_min > right_min else -self.cfg.rotation_speed
            trigger_info["stop_triggered"] = True
            trigger_info["trigger_reason"] = "stop"

        # Rule 2-3: Side suppression
        if left_min < self.cfg.side_threshold and vy > 0:
            vy = 0.0
            trigger_info["side_triggered"] = True
            trigger_info["trigger_reason"] = "left_side"

        if right_min < self.cfg.side_threshold and vy < 0:
            vy = 0.0
            trigger_info["side_triggered"] = True
            if trigger_info["trigger_reason"] == "left_side":
                trigger_info["trigger_reason"] = "both_sides"
            else:
                trigger_info["trigger_reason"] = "right_side"

        # Rule 4: Slowdown
        if front_min < self.cfg.slowdown_threshold and not trigger_info["stop_triggered"]:
            scale = (front_min - self.cfg.stop_threshold) / (
                self.cfg.slowdown_threshold - self.cfg.stop_threshold
            )
            scale = np.clip(scale, self.cfg.min_slowdown_scale, 1.0)
            vx *= scale
            vy *= scale
            trigger_info["slowdown_triggered"] = True
            if trigger_info["trigger_reason"] == "none":
                trigger_info["trigger_reason"] = "slowdown"

        # Rule 5: Hard velocity limits
        vx = float(np.clip(vx, -self.cfg.max_vx, self.cfg.max_vx))
        vy = float(np.clip(vy, -self.cfg.max_vy, self.cfg.max_vy))
        omega = float(np.clip(omega, -self.cfg.max_omega, self.cfg.max_omega))

        filtered_action = np.array([vx, vy, omega], dtype=np.float32)
        return filtered_action, trigger_info


def create_safety_shield_from_config(config: dict) -> SafetyShield:
    """Factory to create SafetyShield from a config dictionary."""
    cfg = config.get("safety_shield", config)
    return SafetyShield(
        SafetyConfig(
            stop_threshold=cfg.get("stop_threshold", 0.3),
            side_threshold=cfg.get("side_threshold", 0.2),
            slowdown_threshold=cfg.get("slowdown_threshold", 0.6),
            min_slowdown_scale=cfg.get("min_slowdown_scale", 0.2),
            max_vx=cfg.get("max_vx", 0.8),
            max_vy=cfg.get("max_vy", 0.8),
            max_omega=cfg.get("max_omega", 1.0),
            front_start_idx=cfg.get("front_start_idx", 24),
            front_end_idx=cfg.get("front_end_idx", 40),
            left_start_idx=cfg.get("left_start_idx", 0),
            left_end_idx=cfg.get("left_end_idx", 23),
            right_start_idx=cfg.get("right_start_idx", 41),
            right_end_idx=cfg.get("right_end_idx", 63),
            rotation_speed=cfg.get("rotation_speed", 0.3),
            prefer_rotation=cfg.get("prefer_rotation", True),
        )
    )
