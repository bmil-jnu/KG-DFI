"""
Evaluation utilities: inference, metrics, bootstrap confidence intervals.

`bootstrap_ci` reproduces the analysis used to report the 95% CIs in Table 2/3
of the manuscript: 10,000 resamples (with replacement) of the held-out test
set predictions, taking the 2.5th/97.5th percentiles of each metric.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Possible", "Positive", "Negative", "Harmful"]


@torch.no_grad()
def run_inference(model, graph, loader, device):
    """Run the model over a DataLoader and collect labels/preds/probs."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        food_ids = batch["food_id"].to(device)
        drug_ids = batch["drug_id"].to(device)
        out = model(graph, food_ids, drug_ids)
        if model.task == "binary":
            probs = out.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        else:
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
        all_labels.extend(batch["label"].numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_binary_metrics(labels, preds, probs):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p, "recall": r, "f1": f,
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
    }


def compute_multiclass_metrics(labels, preds, probs, classes=(0, 1, 2, 3)):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    metrics = {
        "f1_weighted": f, "precision_weighted": p, "recall_weighted": r,
        "kappa": cohen_kappa_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
    }
    try:
        yb = label_binarize(labels, classes=list(classes))
        metrics["auroc_weighted"] = roc_auc_score(yb, probs, average="weighted", multi_class="ovr")
        metrics["auprc_weighted"] = average_precision_score(yb, probs, average="weighted")
    except ValueError:
        metrics["auroc_weighted"] = metrics["auprc_weighted"] = float("nan")
    per_class_recall = recall_score(labels, preds, labels=list(classes), average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"recall_{name}"] = per_class_recall[i]
    return metrics


def evaluate_model(model, test_loader, graph, device, task: str = "binary",
                    num_classes: int = 2) -> dict:
    """Run inference on `test_loader` and return {'metrics': ..., 'predictions': ...}.

    'metrics' includes 'confusion_matrix' (list of lists, class order
    0..num_classes-1) alongside the scalar metrics. 'predictions' contains
    'y_true', 'y_pred', and 'y_proba' as plain Python lists
    (JSON-serializable, for `save_results`).
    """
    labels, preds, probs = run_inference(model, graph, test_loader, device)
    metrics = (compute_binary_metrics(labels, preds, probs) if task == "binary"
               else compute_multiclass_metrics(labels, preds, probs, classes=range(num_classes)))
    metrics["confusion_matrix"] = confusion_matrix(
        labels, preds, labels=list(range(num_classes))).tolist()
    return {
        "metrics": metrics,
        "predictions": {"y_true": labels.tolist(), "y_pred": preds.tolist(),
                         "y_proba": probs.tolist()},
    }


def save_results(results: dict, save_dir: str, filename: str = "test_results.json") -> str:
    """Save an `evaluate_model` results dict to `save_dir/filename`; returns the path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_classification_report(y_true, y_pred, class_names, task: str = "binary") -> None:
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def bootstrap_ci(labels, preds, probs, task="multi", n_bootstrap=10_000, seed=42):
    """95% bootstrap CI for every metric, via resampling with replacement.

    Returns {metric_name: {"point": ..., "lo": ..., "hi": ...}}.
    """
    labels, preds, probs = np.asarray(labels), np.asarray(preds), np.asarray(probs)
    compute_fn = compute_binary_metrics if task == "binary" else compute_multiclass_metrics
    point = compute_fn(labels, preds, probs)

    rng = np.random.default_rng(seed)
    n = len(labels)
    boot = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        for k, v in compute_fn(labels[idx], preds[idx], probs[idx]).items():
            boot[k].append(v)

    ci = {}
    for k, vals in boot.items():
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        ci[k] = {"point": point[k], "lo": float(np.percentile(arr, 2.5)),
                 "hi": float(np.percentile(arr, 97.5))}
    return ci
