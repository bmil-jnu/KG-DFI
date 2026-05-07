"""
KG-DFI Utility Functions
General utility functions for reproducibility, visualization, and logging.
"""

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import json
from datetime import datetime


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to {seed}")


# =============================================================================
# Device Management
# =============================================================================

def get_device(device_name='auto'):
    """
    Get torch device.
    
    Args:
        device_name (str): 'auto', 'cuda', 'cuda:0', 'cpu', etc.
    
    Returns:
        torch.device: Device to use
    """
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"✅ Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    return device


# =============================================================================
# Model Information
# =============================================================================

def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_info(model):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print(f"\n{'='*70}")
    print("Model Information")
    print(f"{'='*70}")
    print(f"Total parameters:       {params['total']:,}")
    print(f"Trainable parameters:   {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Model size (MB):        {params['total'] * 4 / (1024**2):.2f}")
    print(f"{'='*70}\n")


# =============================================================================
# Visualization - Confusion Matrix
# =============================================================================

def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# Visualization - Training Curves
# =============================================================================

def plot_training_curves(history, save_path=None, figsize=(12, 4)):
    """
    Plot training history curves.
    
    Args:
        history (dict): Training history with 'train_loss', 'val_loss', 'val_f1'
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 curve
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# Visualization - ROC and PR Curves
# =============================================================================

def plot_roc_curve(y_true, y_proba, save_path=None, figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true (np.array): True labels
        y_proba (np.array): Prediction probabilities
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_pr_curve(y_true, y_proba, save_path=None, figsize=(8, 6)):
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true (np.array): True labels
        y_proba (np.array): Prediction probabilities
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PR curve saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_pr_curves(y_true, y_proba, save_dir=None):
    """
    Plot both ROC and PR curves side by side.
    
    Args:
        y_true (np.array): True labels
        y_proba (np.array): Prediction probabilities
        save_dir (str): Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_pr_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC/PR curves saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# Visualization - Baseline Comparison
# =============================================================================

def plot_baseline_comparison(results_dict, metric='f1', save_path=None, figsize=(10, 6)):
    """
    Plot baseline model comparison.
    
    Args:
        results_dict (dict): Dictionary of {model_name: results}
        metric (str): Metric to compare
        save_path (str): Path to save figure
        figsize (tuple): Figure size
    """
    model_names = []
    scores = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            metrics = results['metrics']
            if metric in metrics:
                model_names.append(model_name)
                scores.append(metrics[metric])
            elif f'{metric}_weighted' in metrics:
                model_names.append(model_name)
                scores.append(metrics[f'{metric}_weighted'])
    
    # Create bar plot
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, scores, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(f'{metric.upper()} Score', fontsize=12)
    plt.title(f'Baseline Model Comparison ({metric.upper()})', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, max(scores) * 1.15])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# Logging
# =============================================================================

def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir (str): Base directory for experiments
        experiment_name (str): Optional experiment name
    
    Returns:
        str: Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if experiment_name:
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"✅ Experiment directory created: {exp_dir}")
    
    return exp_dir


def save_config(config, save_path):
    """
    Save configuration to JSON file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Config saved: {save_path}")


def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✅ Config loaded: {config_path}")
    return config


# =============================================================================
# Progress Printing
# =============================================================================

def print_header(text, width=70):
    """
    Print formatted header.
    
    Args:
        text (str): Header text
        width (int): Total width
    """
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def print_metrics(metrics, task='binary'):
    """
    Print metrics in formatted way.
    
    Args:
        metrics (dict): Metrics dictionary
        task (str): 'binary' or 'multi'
    """
    print(f"\n{'='*70}")
    print("Metrics:")
    print(f"{'='*70}")
    
    if task == 'binary':
        print(f"Accuracy:   {metrics.get('accuracy', 0):.4f}")
        print(f"Precision:  {metrics.get('precision', 0):.4f}")
        print(f"Recall:     {metrics.get('recall', 0):.4f}")
        print(f"F1 Score:   {metrics.get('f1', 0):.4f}")
        print(f"AUC-ROC:    {metrics.get('auc_roc', 0):.4f}")
        print(f"AUC-PR:     {metrics.get('auc_pr', 0):.4f}")
    else:
        print(f"Accuracy:       {metrics.get('accuracy', 0):.4f}")
        print(f"Kappa:          {metrics.get('kappa', 0):.4f}")
        print(f"F1 (Weighted):  {metrics.get('f1_weighted', 0):.4f}")
        print(f"F1 (Macro):     {metrics.get('f1_macro', 0):.4f}")
        if 'auc_roc_weighted' in metrics:
            print(f"AUC-ROC (W):    {metrics.get('auc_roc_weighted', 0):.4f}")
            print(f"AUC-PR (W):     {metrics.get('auc_pr_weighted', 0):.4f}")
    
    print(f"{'='*70}\n")