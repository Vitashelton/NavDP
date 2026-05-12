#!/usr/bin/env python
"""Plot risk score distribution and ROC/PR curves from evaluation.

Usage:
    python tools/plot_risk_curve.py --dataset logs/lars_dataset.pt \
        --model models/lars_best.pt --output risk_curves.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot risk score curves")
    parser.add_argument("--dataset", type=str, default="logs/lars_dataset.pt",
                        help="Path to dataset .pt file")
    parser.add_argument("--model", type=str, default="models/lars_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to training config for model arch")
    parser.add_argument("--output", type=str, default="risk_curves.png",
                        help="Output image path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import torch
    import yaml

    from lars.risk_model import RiskResidualNet, create_risk_model_from_config
    from lars.risk_dataset import LARSRiskDataset, load_dataset

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load model
    config_path = os.path.join(base, args.config)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model = create_risk_model_from_config(config)
    else:
        model = RiskResidualNet()

    model_path = os.path.join(base, args.model)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)

    device = args.device if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load dataset
    dataset_path = os.path.join(base, args.dataset)
    samples = load_dataset(dataset_path)
    dataset = LARSRiskDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            scan = batch["scan"].to(device)
            raw_action = batch["raw_action"].to(device)
            last_action = batch["last_action"].to(device)
            min_distances = batch["min_distances"].to(device)
            goal_heading = batch["goal_heading"].to(device)

            outputs = model(scan, raw_action, last_action, min_distances, goal_heading)
            risk_prob = torch.sigmoid(outputs["risk_logit"]).cpu().numpy().flatten()
            all_labels.append(batch["risk_label"].numpy())
            all_preds.append(risk_prob)

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)

    # Compute ROC
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PR
    ax = axes[1]
    ax.plot(recall, precision, "r-", linewidth=2, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Score distribution
    ax = axes[2]
    ax.hist(preds[labels == 0], bins=50, alpha=0.5, label="Safe", density=True, color="green")
    ax.hist(preds[labels == 1], bins=50, alpha=0.5, label="Risk", density=True, color="red")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="threshold=0.5")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved to {args.output}")
    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
