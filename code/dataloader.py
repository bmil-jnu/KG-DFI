"""
KG-DFI Data Loading Utilities
Handles loading of knowledge graphs, datasets, and creating dataloaders.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import dgl

from model import FDI_Dataset


# =============================================================================
# Knowledge Graph Loading
# =============================================================================

def load_graph_and_mappings(data_dir='data/knowledge-graph/'):
    """
    Load knowledge graph and entity/relation mappings.
    
    Args:
        data_dir (str): Directory containing KG data
    
    Returns:
        tuple: (graph, mappings)
            - graph (DGLGraph): Knowledge graph
            - mappings (dict): Entity and relation mappings
    """
    # Load graph
    graph_path = os.path.join(data_dir, 'background_kg_graph.pt')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    graph = torch.load(graph_path)
    
    # Load mappings
    mappings_path = os.path.join(data_dir, 'mappings.json')
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    print(f"✅ Graph loaded: {graph.num_nodes():,} nodes, {graph.num_edges():,} edges")
    print(f"✅ Mappings loaded: {mappings['num_nodes']:,} entities, "
          f"{mappings['num_relations']:,} relations")
    
    return graph, mappings


# =============================================================================
# Dataset Loading
# =============================================================================

def load_datasets(task='binary', data_dir='data/'):
    """
    Load train, validation, and test datasets.
    
    Args:
        task (str): 'binary' or 'multi'
        data_dir (str): Base data directory
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_weights)
    """
    # Determine data subdirectory
    if task == 'binary':
        data_subdir = os.path.join(data_dir, 'binary-classification')
    elif task == 'multi':
        data_subdir = os.path.join(data_dir, 'multi-classification')
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'binary' or 'multi'")
    
    if not os.path.exists(data_subdir):
        raise FileNotFoundError(f"Data directory not found: {data_subdir}")
    
    # Load datasets
    train_dataset = torch.load(os.path.join(data_subdir, 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(data_subdir, 'val_dataset.pt'))
    test_dataset = torch.load(os.path.join(data_subdir, 'test_dataset.pt'))
    
    print(f"✅ Train: {len(train_dataset):,} samples")
    print(f"✅ Val:   {len(val_dataset):,} samples")
    print(f"✅ Test:  {len(test_dataset):,} samples")
    
    # Load class weights
    class_weights_path = os.path.join(data_subdir, 'class_weights.pt')
    class_weights = None
    if os.path.exists(class_weights_path):
        class_weights = torch.load(class_weights_path)
        print(f"✅ Class weights loaded")
    
    return train_dataset, val_dataset, test_dataset, class_weights


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloaders(task='binary', batch_size=512, num_workers=4, 
                       pin_memory=True, data_dir='data/'):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        task (str): 'binary' or 'multi'
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory for GPU
        data_dir (str): Base data directory
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_dataset, val_dataset, test_dataset, _ = load_datasets(task, data_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"✅ DataLoaders created (batch_size={batch_size})")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Complete Data Loading
# =============================================================================

def load_all_data(task='binary', batch_size=512, num_workers=4, 
                  pin_memory=True, data_dir='data/'):
    """
    Load everything: graph, mappings, and dataloaders.
    
    Args:
        task (str): 'binary' or 'multi'
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        data_dir (str): Base data directory
    
    Returns:
        dict: Dictionary containing all loaded data
            - 'graph': Knowledge graph
            - 'mappings': Entity/relation mappings
            - 'train_loader': Training dataloader
            - 'val_loader': Validation dataloader
            - 'test_loader': Test dataloader
            - 'class_weights': Class weights
    """
    print("="*70)
    print(f"🚀 Loading data for {task.upper()} classification")
    print("="*70 + "\n")
    
    # Load knowledge graph and mappings
    graph, mappings = load_graph_and_mappings(
        data_dir=os.path.join(data_dir, 'knowledge-graph')
    )
    
    print()
    
    # Load datasets and get class weights
    train_dataset, val_dataset, test_dataset, class_weights = load_datasets(
        task=task, 
        data_dir=data_dir
    )
    
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"✅ DataLoaders created (batch_size={batch_size})")
    print("\n" + "="*70 + "\n")
    
    return {
        'graph': graph,
        'mappings': mappings,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights
    }


# =============================================================================
# Data Statistics
# =============================================================================

def get_dataset_statistics(dataset, task='binary'):
    """
    Compute statistics for a dataset.
    
    Args:
        dataset (FDI_Dataset): Dataset to analyze
        task (str): 'binary' or 'multi'
    
    Returns:
        dict: Dataset statistics
    """
    labels = dataset.labels.numpy()
    
    stats = {
        'total_samples': len(dataset),
        'unique_foods': len(torch.unique(dataset.food_ids)),
        'unique_drugs': len(torch.unique(dataset.drug_ids))
    }
    
    if task == 'binary':
        stats['positive_samples'] = int(labels.sum())
        stats['negative_samples'] = int(len(labels) - labels.sum())
        stats['positive_ratio'] = float(labels.mean())
    else:
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        stats['class_distribution'] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return stats


def print_data_statistics(train_loader, val_loader, test_loader, task='binary'):
    """
    Print dataset statistics.
    
    Args:
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        test_loader (DataLoader): Test dataloader
        task (str): 'binary' or 'multi'
    """
    print("\n" + "="*70)
    print("📊 Dataset Statistics")
    print("="*70)
    
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        dataset = loader.dataset
        stats = get_dataset_statistics(dataset, task)
        
        print(f"\n{name}: {stats['total_samples']:,} samples")
        
        if task == 'binary':
            print(f"  Positive: {stats['positive_samples']:,} ({stats['positive_ratio']:.1%})")
            print(f"  Negative: {stats['negative_samples']:,} ({1-stats['positive_ratio']:.1%})")
        else:
            print(f"  Class distribution:")
            for class_id, count in stats['class_distribution'].items():
                print(f"    Class {class_id}: {count:,}")
    
    print("\n" + "="*70 + "\n")