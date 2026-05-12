"""Action adapter: converts policy outputs to executable velocity commands."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ActionAdapterConfig:
    """Configuration for the action adapter."""

    input_mode: str = "velocity"  # "velocity", "waypoint", "trajectory"
    max_vx: float = 0.5
    max_vy: float = 0.3
    max_omega: float = 0.5
    max_accel_vx: float = 1.0
    max_accel_vy: float = 1.0
    max_accel_omega: float = 2.0
    lp_alpha: float = 0.3          # low-pass filter alpha
    deadzone: float = 0.02         # values below this are zeroed
    emergency_stop: bool = False
    dt: float = 0.05               # control timestep in seconds


class ActionAdapter:
    """Converts raw policy outputs into executable velocity commands.

    Supports three input modes:
    - velocity: [vx, vy, omega] directly.
    - waypoint: [dx, dy, dtheta] relative displacement; converts to velocity
      using dt.
    - trajectory: [[dx, dy, dtheta], ...] sequence; takes first waypoint.

    Applies velocity limiting, acceleration limiting, low-pass filtering,
    deadzone, and emergency stop.
    """

    def __init__(self, config: Optional[ActionAdapterConfig] = None):
        self.cfg = config or ActionAdapterConfig()
        self._last_action: Optional[np.ndarray] = None

    def reset(self):
        """Reset internal state (last action for low-pass filter)."""
        self._last_action = None

    def __call__(self, policy_output: np.ndarray) -> np.ndarray:
        """Convert policy output to an executable velocity action.

        Args:
            policy_output: Raw policy output. Interpretation depends on
                           input_mode.

        Returns:
            action: [vx, vy, omega] velocity command.
        """
        if self.cfg.emergency_stop:
            self._last_action = np.zeros(3, dtype=np.float32)
            return self._last_action.copy()

        action = self._parse_input(policy_output)
        action = self._apply_accel_limit(action)
        action = self._apply_lowpass(action)
        action = self._apply_deadzone(action)
        action = self._apply_velocity_limit(action)
        return action

    def _parse_input(self, policy_output: np.ndarray) -> np.ndarray:
        """Parse policy output into [vx, vy, omega] based on input_mode."""
        arr = np.asarray(policy_output, dtype=np.float32).flatten()

        if self.cfg.input_mode == "velocity":
            if len(arr) >= 3:
                return arr[:3].copy()
            padded = np.zeros(3, dtype=np.float32)
            padded[: len(arr)] = arr
            return padded

        elif self.cfg.input_mode == "waypoint":
            # Single waypoint [dx, dy, dtheta] -> velocity
            if len(arr) >= 3:
                return arr[:3] / self.cfg.dt
            return np.zeros(3, dtype=np.float32)

        elif self.cfg.input_mode == "trajectory":
            # First waypoint of a trajectory -> velocity
            if arr.ndim >= 2:
                wp = arr[0, :3]
            elif len(arr) >= 3:
                wp = arr[:3]
            else:
                wp = np.zeros(3, dtype=np.float32)
            return wp / self.cfg.dt

        else:
            raise ValueError(f"Unknown input_mode: {self.cfg.input_mode}")

    def _apply_accel_limit(self, action: np.ndarray) -> np.ndarray:
        """Clip action based on acceleration limits from last action."""
        if self._last_action is None:
            self._last_action = action.copy()
            return action

        max_delta = np.array([
            self.cfg.max_accel_vx * self.cfg.dt,
            self.cfg.max_accel_vy * self.cfg.dt,
            self.cfg.max_accel_omega * self.cfg.dt,
        ])

        delta = action - self._last_action
        delta = np.clip(delta, -max_delta, max_delta)
        return self._last_action + delta

    def _apply_lowpass(self, action: np.ndarray) -> np.ndarray:
        """Apply first-order low-pass filter."""
        if self._last_action is None:
            self._last_action = action.copy()
            return action
        alpha = self.cfg.lp_alpha
        filtered = alpha * action + (1 - alpha) * self._last_action
        self._last_action = filtered.copy()
        return filtered

    def _apply_deadzone(self, action: np.ndarray) -> np.ndarray:
        """Zero out velocities below deadzone threshold."""
        result = action.copy()
        result[np.abs(result) < self.cfg.deadzone] = 0.0
        return result

    def _apply_velocity_limit(self, action: np.ndarray) -> np.ndarray:
        """Hard-clip velocities to configured maxima."""
        action[0] = np.clip(action[0], -self.cfg.max_vx, self.cfg.max_vx)
        action[1] = np.clip(action[1], -self.cfg.max_vy, self.cfg.max_vy)
        action[2] = np.clip(action[2], -self.cfg.max_omega, self.cfg.max_omega)
        return action


def create_action_adapter_from_config(config: dict) -> ActionAdapter:
    """Factory to create ActionAdapter from a config dictionary."""
    cfg = config.get("action_adapter", config)
    return ActionAdapter(
        ActionAdapterConfig(
            input_mode=cfg.get("input_mode", "velocity"),
            max_vx=cfg.get("max_vx", 0.5),
            max_vy=cfg.get("max_vy", 0.3),
            max_omega=cfg.get("max_omega", 0.5),
            max_accel_vx=cfg.get("max_accel_vx", 1.0),
            max_accel_vy=cfg.get("max_accel_vy", 1.0),
            max_accel_omega=cfg.get("max_accel_omega", 2.0),
            lp_alpha=cfg.get("lp_alpha", 0.3),
            deadzone=cfg.get("deadzone", 0.02),
            emergency_stop=cfg.get("emergency_stop", False),
            dt=cfg.get("dt", 0.05),
        )
    )
