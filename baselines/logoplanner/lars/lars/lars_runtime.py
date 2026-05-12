"""LARS runtime: main orchestrator that ties all components together.

Data flow:
    LoGoPlanner raw output
        -> ActionAdapter -> raw_action
        -> DepthToScan -> scan
        -> RiskResidualNet -> risk_score, residual_action
        -> if risk_score > threshold: learned_action = raw_action + scale * residual
        -> SafetyShield -> final_action
    final_action -> sent to LeKiwi
"""

import time
import os
import yaml
import numpy as np
from typing import Optional, Dict, Tuple, Any

from .depth_to_scan import DepthToScan, create_depth_to_scan_from_config
from .safety_shield import SafetyShield, create_safety_shield_from_config
from .action_adapter import ActionAdapter, create_action_adapter_from_config
from .risk_model import RiskResidualNet, create_risk_model_from_config
from .residual_adapter import ResidualAdapter
from .logger import LARSLogger


class LARSRuntime:
    """Main runtime for the Learning-based Adaptive Residual Safety Adapter.

    Usage:
        runtime = LARSRuntime.from_config("configs/lars.yaml")
        runtime.load_safety_config("configs/safety.yaml")

        # Per-step call:
        final_action, info = runtime.step(policy_output, depth_image,
                                          last_action, goal_heading)
    """

    def __init__(
        self,
        depth_to_scan: DepthToScan,
        safety_shield: SafetyShield,
        action_adapter: ActionAdapter,
        risk_model: Optional[RiskResidualNet] = None,
        risk_threshold: float = 0.5,
        residual_scale: float = 0.3,
        device: str = "cuda:0",
        enable_logging: bool = True,
        log_dir: str = "logs",
        max_log_size_mb: float = 100.0,
    ):
        self.depth_to_scan = depth_to_scan
        self.safety_shield = safety_shield
        self.action_adapter = action_adapter
        self.risk_model = risk_model
        self.risk_threshold = risk_threshold
        self.residual_scale = residual_scale
        self.device = device

        self.residual_adapter = ResidualAdapter(
            risk_model=risk_model,
            safety_shield=safety_shield,
            action_adapter=action_adapter,
            risk_threshold=risk_threshold,
            residual_scale=residual_scale,
            device=device,
        )

        self.logger = LARSLogger(
            log_dir=log_dir,
            max_log_size_mb=max_log_size_mb,
            enabled=enable_logging,
        )

        self._last_action: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, config_path: str) -> "LARSRuntime":
        """Create LARSRuntime from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        lars_cfg = config.get("lars", config)

        depth_to_scan = create_depth_to_scan_from_config(
            config.get("depth_to_scan", {})
        )
        safety_shield = create_safety_shield_from_config(
            config.get("safety_shield", {})
        )
        action_adapter = create_action_adapter_from_config(
            config.get("action_adapter", {})
        )

        # Load risk model if path provided
        model_path = lars_cfg.get("model_path", "")
        risk_model = None
        if model_path and os.path.exists(model_path):
            risk_model = create_risk_model_from_config(config)
            checkpoint = __import__("torch").load(
                model_path, map_location="cpu", weights_only=False
            )
            risk_model.load_state_dict(checkpoint, strict=False)
            risk_model.to(lars_cfg.get("device", "cuda:0"))
            risk_model.eval()

        return cls(
            depth_to_scan=depth_to_scan,
            safety_shield=safety_shield,
            action_adapter=action_adapter,
            risk_model=risk_model,
            risk_threshold=lars_cfg.get("risk_threshold", 0.5),
            residual_scale=lars_cfg.get("residual_scale", 0.3),
            device=lars_cfg.get("device", "cuda:0"),
            enable_logging=lars_cfg.get("enable_logging", True),
            log_dir=lars_cfg.get("log_dir", "logs"),
            max_log_size_mb=lars_cfg.get("max_log_size_mb", 100.0),
        )

    def load_safety_config(self, config_path: str):
        """Load or reload safety shield config from YAML."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.safety_shield = create_safety_shield_from_config(config)

    def load_model(self, model_path: str):
        """Load a trained risk model checkpoint."""
        checkpoint = __import__("torch").load(
            model_path, map_location="cpu", weights_only=False
        )
        if self.risk_model is None:
            self.risk_model = RiskResidualNet()
        self.risk_model.load_state_dict(checkpoint, strict=False)
        self.risk_model.to(self.device)
        self.risk_model.eval()
        self.residual_adapter.risk_model = self.risk_model

    def step(
        self,
        policy_output: np.ndarray,
        depth_image: np.ndarray,
        goal_heading: float = 0.0,
        collision_flag: bool = False,
        manual_takeover_flag: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute one LARS step.

        Args:
            policy_output: Raw output from LoGoPlanner.
            depth_image: 2D depth image (H, W) in configured units.
            goal_heading: Heading angle to goal in radians.
            collision_flag: External collision signal.
            manual_takeover_flag: External takeover signal.

        Returns:
            final_action: [vx, vy, omega] safe velocity command.
            info: dict with all intermediate values.
        """
        t_start = time.time()

        # Convert depth to scan
        scan_meter, scan_norm = self.depth_to_scan(depth_image)

        # Run residual adapter
        last_action = self._last_action if self._last_action is not None else np.zeros(3)
        final_action, info = self.residual_adapter(
            policy_output=policy_output,
            scan=scan_meter,
            last_action=last_action,
            goal_heading=goal_heading,
        )

        self._last_action = final_action.copy()

        latency_ms = (time.time() - t_start) * 1000.0

        # Log
        self.logger.log(
            raw_policy_output=policy_output,
            raw_action=info["raw_action"],
            learned_residual=info["learned_residual"],
            risk_score=info["risk_score"],
            learned_action=info["learned_action"],
            final_action=final_action,
            scan=scan_meter,
            front_min=info["front_min"],
            left_min=info["left_min"],
            right_min=info["right_min"],
            safety_trigger=info["safety_trigger"],
            collision_flag=collision_flag,
            manual_takeover_flag=manual_takeover_flag,
            latency_ms=latency_ms,
            goal_heading=goal_heading,
        )

        info["latency_ms"] = latency_ms
        info["scan_meter"] = scan_meter
        info["scan_norm"] = scan_norm

        return final_action, info

    def reset(self):
        """Reset internal state (action adapter history, log rotation)."""
        self.action_adapter.reset()
        self._last_action = None

    def close(self):
        """Clean up resources."""
        self.logger.close()

    def __del__(self):
        self.close()


# Mock objects for offline testing without hardware -------------------------

class MockDepthToScan(DepthToScan):
    """Mock depth-to-scan that returns synthetic scans."""

    def __call__(self, depth_image=None):
        scan = np.random.uniform(0.5, 5.0, size=self.scan_dim).astype(np.float32)
        if self.normalize:
            scan_norm = (scan - self.min_depth) / (self.max_depth - self.min_depth)
            scan_norm = np.clip(scan_norm, 0.0, 1.0)
            return scan, scan_norm
        return scan, scan


class MockPolicyOutput:
    """Mock policy output generator for offline testing."""

    def __init__(self, mode="velocity"):
        self.mode = mode

    def __call__(self):
        if self.mode == "velocity":
            return np.array([0.3, 0.0, 0.0], dtype=np.float32)
        elif self.mode == "waypoint":
            return np.array([0.015, 0.0, 0.0], dtype=np.float32)
        else:
            return np.random.randn(5, 3).astype(np.float32) * 0.01
