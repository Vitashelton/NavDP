"""Residual adapter: combines learned risk model with safety shield."""

import numpy as np
from typing import Optional, Dict, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .risk_model import RiskResidualNet
from .safety_shield import SafetyShield
from .action_adapter import ActionAdapter


class ResidualAdapter:
    """Applies learned residual correction to raw actions.

    Flow:
    1. raw_action = action_adapter(policy_output)
    2. If risk_model is loaded and risk_score > threshold:
       learned_action = raw_action + residual_scale * residual_action
       else: learned_action = raw_action
    3. final_action, trigger_info = safety_shield(learned_action, scan)
    """

    def __init__(
        self,
        risk_model: Optional[RiskResidualNet] = None,
        safety_shield: Optional[SafetyShield] = None,
        action_adapter: Optional[ActionAdapter] = None,
        risk_threshold: float = 0.5,
        residual_scale: float = 0.3,
        device: str = "cpu",
    ):
        self.risk_model = risk_model
        self.safety_shield = safety_shield or SafetyShield()
        self.action_adapter = action_adapter or ActionAdapter()
        self.risk_threshold = risk_threshold
        self.residual_scale = residual_scale
        self.device = device

    def set_device(self, device: str):
        self.device = device
        if self.risk_model is not None:
            self.risk_model.to(device)

    def __call__(
        self,
        policy_output: np.ndarray,
        scan: np.ndarray,
        last_action: Optional[np.ndarray] = None,
        goal_heading: float = 0.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Adapt raw policy output into a safe final action.

        Args:
            policy_output: Raw policy output (velocity, waypoint, or trajectory).
            scan: 1D pseudo-LiDAR scan in meters, (scan_dim,).
            last_action: Previous action [vx, vy, omega], default zeros.
            goal_heading: Heading to goal in radians.

        Returns:
            final_action: [vx, vy, omega] after all adaptations.
            info: dict with intermediate values for logging.
        """
        raw_action = self.action_adapter(policy_output)

        if last_action is None:
            last_action = np.zeros(3, dtype=np.float32)
        else:
            last_action = np.asarray(last_action, dtype=np.float32)

        # Compute sector min distances
        scan_np = np.asarray(scan, dtype=np.float32).flatten()
        n = len(scan_np)
        front_min = float(np.min(scan_np[n // 3 : 2 * n // 3]))
        left_min = float(np.min(scan_np[: n // 3]))
        right_min = float(np.min(scan_np[2 * n // 3 :]))

        min_distances = np.array([front_min, left_min, right_min], dtype=np.float32)

        # Learned residual
        risk_score = 0.0
        residual_action = np.zeros(3, dtype=np.float32)

        if self.risk_model is not None:
            self.risk_model.eval()
            with torch.no_grad():
                inputs = {
                    "scan": torch.from_numpy(scan_np).unsqueeze(0).to(self.device),
                    "raw_action": torch.from_numpy(raw_action).unsqueeze(0).to(self.device),
                    "last_action": torch.from_numpy(last_action).unsqueeze(0).to(self.device),
                    "min_distances": torch.from_numpy(min_distances).unsqueeze(0).to(self.device),
                    "goal_heading": torch.tensor([[goal_heading]], dtype=torch.float32).to(self.device),
                }
                outputs = self.risk_model(**inputs)
                risk_score = float(torch.sigmoid(outputs["risk_logit"]).cpu().numpy().item())
                residual_action = outputs["residual_action"].cpu().numpy().squeeze(0)

                if risk_score > self.risk_threshold:
                    learned_action = raw_action + self.residual_scale * residual_action
                else:
                    learned_action = raw_action.copy()
        else:
            learned_action = raw_action.copy()

        # Hard safety shield
        final_action, trigger_info = self.safety_shield(learned_action, scan_np)

        info = {
            "raw_action": raw_action,
            "learned_residual": residual_action,
            "risk_score": risk_score,
            "learned_action": learned_action,
            "final_action": final_action,
            "front_min": front_min,
            "left_min": left_min,
            "right_min": right_min,
            "safety_trigger": trigger_info["stop_triggered"] or trigger_info["side_triggered"],
            "trigger_reason": trigger_info["trigger_reason"],
        }
        return final_action, info
