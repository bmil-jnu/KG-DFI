"""Shared helpers: reproducibility, logging, and plotting utilities used by
the demo notebook (`KG-DFI.ipynb`)."""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (+ CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """Resolve a device string, falling back to CPU if CUDA/the requested
    device is unavailable."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[get_device] {device_name} requested but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def count_parameters(model) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_header(title: str, width: int = 65) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def print_model_info(model) -> None:
    """Print parameter counts for a KGDFI model (see model.get_model_info)."""
    from model import get_model_info
    info = get_model_info(model)
    print_header(f"Model info ({info['task']})")
    print(f"  total parameters     : {info['total_parameters']:,}")
    print(f"  trainable parameters : {info['trainable_parameters']:,}")
    print(f"  approx. size         : {info['size_mb']:.1f} MB")
    print(f"  uses cross-attention : {info['uses_attention']}")


def print_metrics(metrics: dict, task: str = "binary") -> None:
    print_header(f"Test set metrics ({task})")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue  # printed separately via plot_confusion_matrix / print_classification_report
        print(f"  {k:<20s}: {v:.4f}")


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create (if needed) and return `base_dir/experiment_name`."""
    path = os.path.join(base_dir, experiment_name)
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_curves(history: dict, save_path: str) -> str:
    """Plot train/val loss and val F1 over epochs; save to `save_path`."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_f1"], color="#378ADD", label="val F1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("weighted F1"); ax2.set_title("Validation F1")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return save_path


def plot_confusion_matrix(cm, class_names, save_path: str, figsize=(6, 5)) -> str:
    """Plot a pre-computed confusion matrix (e.g. `results['metrics']['confusion_matrix']`)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return save_path


def plot_roc_pr_curves(y_true, y_proba, save_dir: str) -> str:
    """ROC and precision-recall curves for the binary task.

    `y_proba` must be the predicted-positive-class probability (a single
    column), not the full class-probability array.
    """
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, label=f"AUROC={auc(fpr, tpr):.4f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC"); ax1.legend()

    ax2.plot(recall, precision, label=f"AUPRC={auc(recall, precision):.4f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    return path
