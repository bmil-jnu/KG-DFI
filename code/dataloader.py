"""
Data loading utilities for KG-DFI.

Expects `data_dir` (see `data/README.md`) to contain:
  - background_kg_graph.pt   : DGL heterogeneous graph (entities + relations,
                                direct drug-food edges removed to prevent leakage)
  - phase1_results.json      : {"num_entities": int, "num_relations": int}
  - mappings.json            : {"node_to_id": {entity_name: node_index}}
  - {train,val,test}_raw.csv : binary task -- food_entity, drug_entity, label
  - fdi_multi_class_dataset.csv : multi-class task -- food_entity, drug_entity,
                                    class_label (1-indexed: 1=Possible..4=Harmful)
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from model import FDI_Dataset

EFFECT_TO_LABEL = {"Possible": 0, "Positive": 1, "Negative": 2, "Harmful": 3}
CLASS_NAMES = ["Possible", "Positive", "Negative", "Harmful"]
FOCAL_ALPHA_POWER = 1.5  # exponent applied to balanced class weights (multi-class only)


def _load_graph(data_dir):
    graph = torch.load(os.path.join(data_dir, "background_kg_graph.pt"))
    with open(os.path.join(data_dir, "phase1_results.json")) as f:
        meta = json.load(f)
    with open(os.path.join(data_dir, "mappings.json")) as f:
        node_to_id = json.load(f)["node_to_id"]
    return graph, meta["num_entities"], meta["num_relations"], node_to_id


def _load_binary_splits(data_dir, node_to_id):
    splits = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(data_dir, f"{split}_raw.csv"))
        mask = df["food_entity"].isin(node_to_id) & df["drug_entity"].isin(node_to_id)
        df = df[mask]
        food_ids = np.array([node_to_id[e] for e in df["food_entity"]])
        drug_ids = np.array([node_to_id[e] for e in df["drug_entity"]])
        labels = df["label"].to_numpy(dtype=np.float32)
        splits[split] = FDI_Dataset(food_ids, drug_ids, labels)
    return splits["train"], splits["val"], splits["test"]


def _load_multiclass_splits(data_dir, node_to_id, csv_file, seed=42):
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    food_ids, drug_ids, labels = [], [], []
    for _, row in df.iterrows():
        fe, de, lb = row["food_entity"], row["drug_entity"], row["class_label"]
        if fe in node_to_id and de in node_to_id:
            food_ids.append(node_to_id[fe])
            drug_ids.append(node_to_id[de])
            labels.append(lb - 1)  # stored 1-indexed in the source CSV
    food_ids, drug_ids, labels = np.array(food_ids), np.array(drug_ids), np.array(labels)

    X = np.column_stack([food_ids, drug_ids])
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, labels, test_size=0.3, stratify=labels,
                                                 random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp,
                                                 random_state=seed)
    return (FDI_Dataset(X_tr[:, 0], X_tr[:, 1], y_tr),
            FDI_Dataset(X_val[:, 0], X_val[:, 1], y_val),
            FDI_Dataset(X_te[:, 0], X_te[:, 1], y_te))


def load_all_data(task: str, data_dir: str, batch_size: int = 64, num_workers: int = 4,
                   multiclass_csv: str = "fdi_multi_class_dataset.csv", seed: int = 42) -> dict:
    """Load the graph and train/val/test DataLoaders for the requested task.

    Returns a dict with keys: 'graph', 'train_loader', 'val_loader', 'test_loader',
    'mappings' ({'num_nodes', 'num_relations'}), and 'class_weights' (None for
    the binary task; a per-class balanced weight tensor for the multi-class
    task -- see model.FocalLoss docstring for why this differs from the
    binary task's fixed alpha=0.25).
    """
    assert task in ("binary", "multi")
    graph, num_entities, num_relations, node_to_id = _load_graph(data_dir)

    if task == "binary":
        train_ds, val_ds, test_ds = _load_binary_splits(data_dir, node_to_id)
        class_weights = None
    else:
        train_ds, val_ds, test_ds = _load_multiclass_splits(
            data_dir, node_to_id, multiclass_csv, seed=seed)
        cw = compute_class_weight("balanced", classes=np.unique(train_ds.labels.numpy()),
                                   y=train_ds.labels.numpy())
        class_weights = torch.FloatTensor(cw ** FOCAL_ALPHA_POWER)

    return {
        "graph": graph,
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers),
        "mappings": {"num_nodes": num_entities, "num_relations": num_relations},
        "class_weights": class_weights,
    }


def print_data_statistics(train_loader, val_loader, test_loader, task: str = "binary") -> None:
    """Print sample counts and label distribution for each split."""
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        labels = torch.cat([batch["label"] for batch in loader]).numpy()
        if task == "binary":
            dist = {"negative": int((labels == 0).sum()), "positive": int((labels == 1).sum())}
        else:
            dist = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(4)}
        print(f"  {name:<6s}: {len(labels)} samples, dist={dist}")
