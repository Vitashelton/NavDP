#!/usr/bin/env python
"""Evaluate LARS model offline on a logged dataset.

Computes metrics by replaying logged episodes through the LARS runtime
(or through the saved risk model directly) without hardware.

Usage:
    python eval_lars_offline.py --log-dir logs/ --model models/lars_best.pt
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lars.risk_model import RiskResidualNet, create_risk_model_from_config
from lars.risk_dataset import LARSRiskDataset, load_dataset
from lars.metrics import compute_aggregate_metrics, print_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation of LARS model")
    parser.add_argument("--dataset", type=str, default="logs/lars_dataset.pt",
                        help="Path to dataset .pt file")
    parser.add_argument("--model", type=str, default="models/lars_best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to training config for model architecture")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--risk-threshold", type=float, default=0.5,
                        help="Threshold for binary risk classification")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Optional: also compute episode metrics from log dir")
    return parser.parse_args()


def evaluate_model(dataset_path, model, device, batch_size, risk_threshold):
    """Evaluate model on a labeled dataset."""
    samples = load_dataset(dataset_path)
    dataset = LARSRiskDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    model.to(device)
    model.eval()

    all_risk_labels = []
    all_risk_preds = []
    all_residual_labels = []
    all_residual_preds = []

    with torch.no_grad():
        for batch in loader:
            scan = batch["scan"].to(device)
            raw_action = batch["raw_action"].to(device)
            last_action = batch["last_action"].to(device)
            min_distances = batch["min_distances"].to(device)
            goal_heading = batch["goal_heading"].to(device)

            outputs = model(scan, raw_action, last_action, min_distances, goal_heading)
            risk_prob = torch.sigmoid(outputs["risk_logit"]).cpu().numpy().flatten()
            residual = outputs["residual_action"].cpu().numpy()

            all_risk_labels.append(batch["risk_label"].numpy())
            all_risk_preds.append(risk_prob)
            all_residual_labels.append(batch["residual_label"].numpy())
            all_residual_preds.append(residual)

    risk_labels = np.concatenate(all_risk_labels)
    risk_preds = np.concatenate(all_risk_preds)
    residual_labels = np.concatenate(all_residual_labels)
    residual_preds = np.concatenate(all_residual_preds)

    # Risk classification metrics
    risk_binary = (risk_preds >= risk_threshold).astype(np.float32)

    metrics = {
        "num_samples": len(risk_labels),
        "risk_positive_ratio": float(np.mean(risk_labels)),
        "risk_auc_roc": float(roc_auc_score(risk_labels, risk_preds))
        if len(np.unique(risk_labels)) > 1 else 0.0,
        "risk_avg_precision": float(average_precision_score(risk_labels, risk_preds))
        if len(np.unique(risk_labels)) > 1 else 0.0,
        "risk_f1": float(f1_score(risk_labels, risk_binary, zero_division=0)),
        "residual_mae": float(np.mean(np.abs(residual_labels - residual_preds))),
        "residual_rmse": float(np.sqrt(np.mean((residual_labels - residual_preds) ** 2))),
        "residual_mae_per_dim": np.mean(np.abs(residual_labels - residual_preds), axis=0).tolist(),
        "residual_cosine_sim": float(
            np.mean(np.sum(residual_labels * residual_preds, axis=1) /
                    (np.linalg.norm(residual_labels, axis=1) * np.linalg.norm(residual_preds, axis=1) + 1e-8))
        ),
    }

    return metrics


def main():
    args = parse_args()

    # Resolve paths relative to lars/ directory
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = args.model if os.path.isabs(args.model) else os.path.join(base, args.model)
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(base, args.dataset)

    # Load model
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base, args.config)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = create_risk_model_from_config(config)
    else:
        model = RiskResidualNet()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: Model checkpoint not found at {model_path}, using random weights")

    device = args.device if torch.cuda.is_available() else "cpu"

    # Evaluate
    print(f"Evaluating on {dataset_path}...")
    metrics = evaluate_model(dataset_path, model, device, args.batch_size, args.risk_threshold)

    print(f"\n{'='*50}")
    print("LARS Model Evaluation Results")
    print(f"{'='*50}")
    print(f"  Dataset samples:     {metrics['num_samples']}")
    print(f"  Risk positive ratio: {metrics['risk_positive_ratio']:.4f}")
    print(f"  Risk AUC-ROC:        {metrics['risk_auc_roc']:.4f}")
    print(f"  Risk Avg Precision:  {metrics['risk_avg_precision']:.4f}")
    print(f"  Risk F1 (thresh={args.risk_threshold}): {metrics['risk_f1']:.4f}")
    print(f"  Residual MAE:        {metrics['residual_mae']:.4f}")
    print(f"  Residual RMSE:       {metrics['residual_rmse']:.4f}")
    print(f"  Residual MAE/dim:    {metrics['residual_mae_per_dim']}")
    print(f"  Residual CosSim:     {metrics['residual_cosine_sim']:.4f}")
    print(f"{'='*50}")

    # Also compute episode metrics if log dir provided
    if args.log_dir:
        print(f"\nComputing episode metrics from {args.log_dir}...")
        agg = compute_aggregate_metrics(args.log_dir)
        print_metrics(agg, prefix="Aggregate Episodes")


if __name__ == "__main__":
    main()
