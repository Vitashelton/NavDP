#!/usr/bin/env python
"""Train the LARS RiskResidualNet model.

Usage:
    python train_lars.py --config configs/train.yaml
"""

import argparse
import os
import sys
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lars.risk_model import RiskResidualNet, create_risk_model_from_config
from lars.risk_dataset import LARSRiskDataset, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train LARS RiskResidualNet")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path from config")
    parser.add_argument("--output", type=str, default=None,
                        help="Override model save path")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. cuda:0, cpu)")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(model, batch, device, lambda_residual, lambda_l2):
    scan = batch["scan"].to(device)
    raw_action = batch["raw_action"].to(device)
    last_action = batch["last_action"].to(device)
    min_distances = batch["min_distances"].to(device)
    goal_heading = batch["goal_heading"].to(device)
    risk_label = batch["risk_label"].to(device).unsqueeze(-1)
    residual_label = batch["residual_label"].to(device)

    outputs = model(scan, raw_action, last_action, min_distances, goal_heading)

    loss_risk = nn.functional.binary_cross_entropy_with_logits(
        outputs["risk_logit"], risk_label
    )
    loss_residual = nn.functional.smooth_l1_loss(
        outputs["residual_action"], residual_label
    )

    # L2 regularization
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)

    total_loss = loss_risk + lambda_residual * loss_residual + lambda_l2 * l2_loss

    return total_loss, {
        "loss_total": total_loss.item(),
        "loss_risk": loss_risk.item(),
        "loss_residual": loss_residual.item(),
        "l2_loss": l2_loss.item() if isinstance(l2_loss, float) else l2_loss.item(),
    }


def evaluate(model, dataloader, device, lambda_residual, lambda_l2):
    model.eval()
    total_loss = 0.0
    total_risk_loss = 0.0
    total_residual_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, loss_dict = compute_loss(
                model, batch, device, lambda_residual, lambda_l2
            )
            total_loss += loss_dict["loss_total"]
            total_risk_loss += loss_dict["loss_risk"]
            total_residual_loss += loss_dict["loss_residual"]
            n_batches += 1

    return {
        "loss_total": total_loss / n_batches,
        "loss_risk": total_risk_loss / n_batches,
        "loss_residual": total_residual_loss / n_batches,
    }


def main():
    args = parse_args()

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config
    )
    if not os.path.exists(config_path):
        config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    data_cfg = config["data"]
    log_cfg = config["logging"]

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(train_cfg.get("seed", 42))

    # Load dataset
    dataset_path = args.dataset or data_cfg["dataset_path"]
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_path
    )
    print(f"Loading dataset from: {dataset_path}")
    samples = load_dataset(dataset_path)
    dataset = LARSRiskDataset(samples)

    # Train/val split
    train_split = data_cfg.get("train_split", 0.85)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
    )

    # Create model
    model = create_risk_model_from_config(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    epochs = train_cfg["epochs"]
    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif train_cfg["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("step_size", 50),
            gamma=train_cfg.get("step_gamma", 0.5),
        )
    else:
        scheduler = None

    # AMP
    use_amp = train_cfg.get("use_amp", False) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    lambda_residual = train_cfg["lambda_residual"]
    lambda_l2 = train_cfg["lambda_l2"]
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)
    patience = train_cfg.get("early_stopping_patience", -1)

    save_path = args.output or train_cfg["save_path"]
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), save_path
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    curve_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        train_cfg.get("curve_csv", "logs/training_curve.csv"),
    )
    os.makedirs(os.path.dirname(curve_csv), exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    curve_rows = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_risk_loss = 0.0
        epoch_residual_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = compute_loss(
                        model, batch, device, lambda_residual, lambda_l2
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = compute_loss(
                    model, batch, device, lambda_residual, lambda_l2
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss_dict["loss_total"]
            epoch_risk_loss += loss_dict["loss_risk"]
            epoch_residual_loss += loss_dict["loss_residual"]
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        avg_train_risk = epoch_risk_loss / n_batches
        avg_train_residual = epoch_residual_loss / n_batches

        # Validation
        val_metrics = evaluate(model, val_loader, device, lambda_residual, lambda_l2)
        val_loss = val_metrics["loss_total"]

        if scheduler is not None:
            scheduler.step()

        # Logging
        if (epoch + 1) % log_cfg.get("log_interval", 1) == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} (risk: {avg_train_risk:.4f}, "
                f"res: {avg_train_residual:.4f}) | "
                f"Val Loss: {val_loss:.4f} (risk: {val_metrics['loss_risk']:.4f}, "
                f"res: {val_metrics['loss_residual']:.4f}) | "
                f"LR: {lr:.6f}"
            )

        curve_rows.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_risk_loss": avg_train_risk,
            "train_residual_loss": avg_train_residual,
            "val_loss": val_loss,
            "val_risk_loss": val_metrics["loss_risk"],
            "val_residual_loss": val_metrics["loss_residual"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save training curve
    with open(curve_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"Training curve saved to: {curve_csv}")


if __name__ == "__main__":
    main()
