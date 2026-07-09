"""
Training utilities for KG-DFI.

`train_model` reproduces the training procedure used to produce Table 2
(binary) and Table 3 (multi-class) results: Adam optimizer, gradient clipping
(max norm 1.0), ReduceLROnPlateau scheduling on validation F1, and early
stopping on validation F1 (not validation loss -- see README.md /
"Focal loss class weighting").
"""
import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from model import FocalLoss


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best_score, self.counter, self.best_state = None, 0, None

    def __call__(self, score: float, model) -> bool:
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score, self.counter = score, 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _make_criterion(config: dict, device) -> FocalLoss:
    """Build the task-appropriate FocalLoss from a flat config dict.

    binary: alpha = config['focal_alpha'] (default 0.25, fixed scalar).
    multi : alpha = config['class_weights'] (a per-class tensor produced by
            dataloader.load_all_data -- NOT a fixed scalar). If absent,
            falls back to unweighted focal loss.
    """
    task = config["task"]
    gamma = config.get("focal_gamma", 2.0 if task == "binary" else 1.0)
    if task == "binary":
        return FocalLoss(task="binary", alpha=config.get("focal_alpha", 0.25), gamma=gamma)
    class_weights = config.get("class_weights")
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return FocalLoss(task="multi", alpha=class_weights, gamma=gamma)


@torch.no_grad()
def _evaluate_loss_and_f1(model, graph, loader, criterion, task, device):
    model.eval()
    losses, all_labels, all_preds = [], [], []
    for batch in loader:
        food_ids = batch["food_id"].to(device)
        drug_ids = batch["drug_id"].to(device)
        labels = batch["label"].to(device)
        if task == "binary":
            labels = labels.float()
        out = model(graph, food_ids, drug_ids)
        losses.append(criterion(out, labels).item())
        if task == "binary":
            preds = (out >= 0.5).long().cpu().numpy()
        else:
            preds = out.argmax(dim=1).cpu().numpy()
        all_labels.extend(labels.long().cpu().numpy())
        all_preds.extend(preds)
    f1 = precision_recall_fscore_support(
        all_labels, all_preds, average="binary" if task == "binary" else "weighted",
        zero_division=0,
    )[2]
    return float(np.mean(losses)), float(f1)


def train_model(model, train_loader, val_loader, graph, config: dict, device="cpu"):
    """Train `model` and return (trained_model, history).

    `history` contains per-epoch 'train_loss', 'val_loss', and 'val_f1' lists,
    suitable for `utils.plot_training_curves`.
    """
    task = config["task"]
    criterion = _make_criterion(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get("learning_rate", 5e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.get("scheduler_factor", 0.5),
        patience=config.get("scheduler_patience", 5),
    )
    stopper = EarlyStopping(
        patience=config.get("patience", 30), mode="max",
        min_delta=config.get("min_delta", 1e-4),
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    for epoch in range(config.get("num_epochs", 200)):
        model.train()
        train_losses = []
        for batch in train_loader:
            food_ids = batch["food_id"].to(device)
            drug_ids = batch["drug_id"].to(device)
            labels = batch["label"].to(device)
            if task == "binary":
                labels = labels.float()
            optimizer.zero_grad()
            loss = criterion(model(graph, food_ids, drug_ids), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_f1 = _evaluate_loss_and_f1(model, graph, val_loader, criterion, task, device)
        scheduler.step(val_f1)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"  epoch {epoch + 1:3d} | train_loss={history['train_loss'][-1]:.4f} "
              f"| val_loss={val_loss:.4f} | val_F1={val_f1:.4f} "
              f"| patience={stopper.counter}/{stopper.patience}")

        if stopper(val_f1, model):
            print(f"  early stopping at epoch {epoch + 1} (best val F1={stopper.best_score:.4f})")
            break

    stopper.restore(model)
    return model, history


def save_checkpoint(model, save_dir: str, filename: str = "model_checkpoint.pt") -> str:
    """Save model weights to `save_dir/filename`; returns the full path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model, path: str, device="cpu"):
    """Load weights from `path` into `model` (in place) and return `model`.

    Use this to skip training and evaluate directly with a released
    checkpoint (see data/README.md). `model` must already be constructed
    with the same architecture/config used to produce the checkpoint
    (embedding_dim, kg_layers, num_heads, task, etc.) -- `create_fdi_model`
    with the `binary_config` / `multi_config` from `KG-DFI.ipynb` reproduces
    this.

    Handles both checkpoint formats present in this project's history:
      - `best_model.pt` (multi-class, Table 3): a plain `model.state_dict()`.
      - `best_model_binary.pt` (binary, Table 2): a dict with a
        'model_state_dict' key (plus training metadata such as 'val_f1' and
        'config', which are ignored here).
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
