"""Lightweight PyTorch model for risk assessment and residual action prediction."""

from typing import Optional, Dict, List, Any

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class RiskResidualNet(nn.Module):
        """Predicts collision risk and residual action correction.

        Input:
            scan: (batch, scan_dim) pseudo-LiDAR distances
            raw_action: (batch, 3) current raw action [vx, vy, omega]
            last_action: (batch, 3) previous action [vx, vy, omega]
            min_distances: (batch, 3) [front_min, left_min, right_min]
            goal_heading: (batch, 1) heading to goal in radians

        Output:
            risk_logit: (batch, 1) collision risk logit (pre-sigmoid)
            residual_action: (batch, 3) additive correction to raw_action
        """

        def __init__(
            self,
            scan_dim: int = 64,
            raw_action_dim: int = 3,
            last_action_dim: int = 3,
            min_distances_dim: int = 3,
            goal_heading_dim: int = 1,
            hidden_dims: List[int] = None,
            activation: str = "silu",
            dropout: float = 0.1,
            residual_output_scale: float = 0.5,
        ):
            if hidden_dims is None:
                hidden_dims = [128, 128, 64]
            super().__init__()

            input_dim = (
                scan_dim
                + raw_action_dim
                + last_action_dim
                + min_distances_dim
                + goal_heading_dim
            )

            act_fn = nn.SiLU() if activation == "silu" else nn.ReLU()

            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            self.backbone = nn.Sequential(*layers)

            self.risk_head = nn.Linear(hidden_dims[-1], 1)
            self.residual_head = nn.Linear(hidden_dims[-1], 3)

            self.residual_output_scale = residual_output_scale

        def forward(
            self,
            scan: "torch.Tensor",
            raw_action: "torch.Tensor",
            last_action: "torch.Tensor",
            min_distances: "torch.Tensor",
            goal_heading: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            """Forward pass.

            Args:
                scan: (B, scan_dim)
                raw_action: (B, 3)
                last_action: (B, 3)
                min_distances: (B, 3)
                goal_heading: (B, 1)

            Returns:
                dict with keys 'risk_logit' (B, 1) and 'residual_action' (B, 3).
            """
            x = torch.cat([scan, raw_action, last_action, min_distances, goal_heading], dim=-1)
            features = self.backbone(x)

            risk_logit = self.risk_head(features)  # (B, 1)
            residual = torch.tanh(self.residual_head(features)) * self.residual_output_scale

            return {"risk_logit": risk_logit, "residual_action": residual}

        def predict_risk(self, *args, **kwargs) -> "torch.Tensor":
            """Return risk probability (sigmoid of logit) only."""
            outputs = self.forward(*args, **kwargs)
            return torch.sigmoid(outputs["risk_logit"])

else:
    # Stub for environments without torch
    class RiskResidualNet:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RiskResidualNet requires PyTorch. Install with: pip install torch"
            )


def create_risk_model_from_config(config: dict) -> Any:
    """Factory to create RiskResidualNet from a config dictionary."""
    cfg = config.get("model", config)
    return RiskResidualNet(
        scan_dim=cfg.get("scan_dim", 64),
        raw_action_dim=cfg.get("raw_action_dim", 3),
        last_action_dim=cfg.get("last_action_dim", 3),
        min_distances_dim=cfg.get("min_distances_dim", 3),
        goal_heading_dim=cfg.get("goal_heading_dim", 1),
        hidden_dims=cfg.get("hidden_dims", [128, 128, 64]),
        activation=cfg.get("activation", "silu"),
        dropout=cfg.get("dropout", 0.1),
        residual_output_scale=cfg.get("residual_output_scale", 0.5),
    )
