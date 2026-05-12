"""Dataset builder: converts LARS logs into training samples."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import json
import os


# Time window (seconds) before collision/takeover to label as risk
RISK_WINDOW_SEC = 1.5
# Assumed timestep if not present in log
DEFAULT_DT = 0.1

# Required log fields per timestep
REQUIRED_FIELDS = [
    "timestamp", "scan", "raw_action", "safe_action", "filtered_action",
    "last_action", "front_min", "left_min", "right_min",
    "collision_flag", "manual_takeover_flag", "safety_trigger_flag",
    "success_flag", "timeout_flag",
]


def build_samples_from_logs(
    log_dir: str, dt: float = DEFAULT_DT
) -> List[Dict[str, np.ndarray]]:
    """Parse JSONL log files and construct labeled training samples.

    Each log file should contain one episode as a list of per-timestep dicts,
    or one JSON object per line (JSONL).

    Labeling rules:
    - collision_flag=True → risk_label=1 for the current frame and previous
      frames within RISK_WINDOW_SEC.
    - manual_takeover_flag=True → risk_label=1 for the current frame and
      previous frames within RISK_WINDOW_SEC.
    - safety_trigger_flag=True → risk_label=1 for current frame.
    - success_flag=True AND no safety triggers → risk_label=0.
    - Otherwise risk_label=0.

    Residual label:
    - If safe_action is available: residual_label = safe_action - raw_action
    - Else: residual_label = filtered_action - raw_action
    """
    all_episodes = []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith((".jsonl", ".json")):
            continue
        fpath = os.path.join(log_dir, fname)
        with open(fpath, "r") as f:
            content = f.read().strip()

        if not content:
            continue

        # Try JSONL (one object per line)
        try:
            lines = content.split("\n")
            episode = [json.loads(line) for line in lines if line.strip()]
            if not episode:
                # Try single JSON array
                episode = json.loads(content)
        except json.JSONDecodeError:
            continue

        if not isinstance(episode, list):
            continue

        samples = _label_episode(episode, dt)
        all_episodes.extend(samples)

    return all_episodes


def _label_episode(
    episode: List[Dict], dt: float
) -> List[Dict[str, np.ndarray]]:
    """Label a single episode's frames."""
    N = len(episode)
    if N == 0:
        return []

    risk_window_frames = max(1, int(RISK_WINDOW_SEC / dt))

    # Initialize labels
    risk_labels = np.zeros(N, dtype=np.float32)
    residual_labels = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        frame = episode[i]

        # Check if any risk event occurs at or after this frame within window
        for j in range(i, min(N, i + risk_window_frames)):
            fj = episode[j]
            if (
                fj.get("collision_flag", False)
                or fj.get("manual_takeover_flag", False)
            ):
                risk_labels[i] = 1.0
                break
            if fj.get("safety_trigger_flag", False):
                risk_labels[i] = 1.0
                break

        # Residual label
        raw_action = np.array(frame.get("raw_action", [0, 0, 0]), dtype=np.float32)
        safe_action = frame.get("safe_action", None)
        filtered_action = frame.get("filtered_action", None)

        if safe_action is not None:
            residual_labels[i] = np.array(safe_action, dtype=np.float32) - raw_action
        elif filtered_action is not None:
            residual_labels[i] = np.array(filtered_action, dtype=np.float32) - raw_action

        # Success frames with no triggers get risk_label=0 (already default)

    # Build sample dicts
    samples = []
    for i in range(N):
        frame = episode[i]
        sample = {
            "scan": np.array(frame.get("scan", np.zeros(64)), dtype=np.float32),
            "raw_action": np.array(frame.get("raw_action", [0, 0, 0]), dtype=np.float32),
            "last_action": np.array(frame.get("last_action", [0, 0, 0]), dtype=np.float32),
            "min_distances": np.array(
                [
                    frame.get("front_min", 8.0),
                    frame.get("left_min", 8.0),
                    frame.get("right_min", 8.0),
                ],
                dtype=np.float32,
            ),
            "goal_heading": np.array(
                [frame.get("goal_heading", 0.0)], dtype=np.float32
            ),
            "risk_label": risk_labels[i],
            "residual_label": residual_labels[i],
        }
        samples.append(sample)

    return samples


class LARSRiskDataset(Dataset):
    """PyTorch Dataset for training RiskResidualNet."""

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "scan": torch.from_numpy(s["scan"]),
            "raw_action": torch.from_numpy(s["raw_action"]),
            "last_action": torch.from_numpy(s["last_action"]),
            "min_distances": torch.from_numpy(s["min_distances"]),
            "goal_heading": torch.from_numpy(s["goal_heading"]),
            "risk_label": torch.tensor(s["risk_label"]),
            "residual_label": torch.from_numpy(s["residual_label"]),
        }


def save_dataset(samples: List[Dict], output_path: str):
    """Save processed samples to a .pt file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load processed samples from a .pt file."""
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected dataset format in {dataset_path}")
