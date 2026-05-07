"""
KG-DFI Evaluation Utilities
Handles model evaluation, metrics computation, and result saving.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report,
    cohen_kappa_score, precision_recall_fscore_support,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import json
import os
from datetime import datetime


# =============================================================================
# Binary Classification Metrics
# =============================================================================

def compute_binary_metrics(y_true, y_pred, y_proba, threshold=0.5):
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_proba (np.array): Prediction probabilities
        threshold (float): Classification threshold
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC metrics
    metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    metrics['auc_pr'] = auc(recall_vals, precision_vals)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
    
    return metrics


# =============================================================================
# Multi-class Classification Metrics
# =============================================================================

def compute_multi_metrics(y_true, y_pred, y_proba, num_classes=4):
    """
    Compute comprehensive metrics for multi-class classification.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_proba (np.array): Prediction probabilities [n_samples, n_classes]
        num_classes (int): Number of classes
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = float(precision_w)
    metrics['recall_weighted'] = float(recall_w)
    metrics['f1_weighted'] = float(f1_w)
    
    # Macro metrics
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = float(precision_m)
    metrics['recall_macro'] = float(recall_m)
    metrics['f1_macro'] = float(f1_m)
    
    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['precision_per_class'] = precision_per.tolist()
    metrics['recall_per_class'] = recall_per.tolist()
    metrics['f1_per_class'] = f1_per.tolist()
    metrics['support_per_class'] = support_per.tolist()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # AUC metrics (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if y_true_bin.shape[1] > 1:
            auc_roc_scores = []
            auc_pr_scores = []
            
            for i in range(num_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:
                    # AUC-ROC
                    auc_roc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    auc_roc_scores.append(float(auc_roc))
                    
                    # AUC-PR
                    auc_pr = average_precision_score(y_true_bin[:, i], y_proba[:, i])
                    auc_pr_scores.append(float(auc_pr))
            
            if auc_roc_scores:
                metrics['auc_roc_per_class'] = auc_roc_scores
                metrics['auc_roc_weighted'] = float(np.average(auc_roc_scores, weights=support_per))
                metrics['auc_pr_per_class'] = auc_pr_scores
                metrics['auc_pr_weighted'] = float(np.average(auc_pr_scores, weights=support_per))
    except:
        pass
    
    return metrics


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(model, test_loader, graph, device, task='binary', num_classes=4):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        graph: Knowledge graph
        device: Device to use
        task: 'binary' or 'multi'
        num_classes: Number of classes (for multi)
    
    Returns:
        dict: Evaluation results with predictions and metrics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {task.upper()} classification model")
    print(f"{'='*70}\n")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    graph = graph.to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            food_ids = batch['food_id'].to(device)
            drug_ids = batch['drug_id'].to(device)
            labels = batch['label']
            
            # Forward pass
            outputs = model(graph, food_ids, drug_ids)
            
            # Get predictions and probabilities
            if task == 'binary':
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_probs.extend(probs)
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)
    
    # Compute metrics
    if task == 'binary':
        metrics = compute_binary_metrics(y_true, y_pred, y_proba)
    else:
        metrics = compute_multi_metrics(y_true, y_pred, y_proba, num_classes)
    
    # Print results
    print(f"\n{'='*70}")
    print("Test Results:")
    print(f"{'='*70}")
    
    if task == 'binary':
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    else:
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Kappa:         {metrics['kappa']:.4f}")
        print(f"F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"F1 (Macro):    {metrics['f1_macro']:.4f}")
        if 'auc_roc_weighted' in metrics:
            print(f"AUC-ROC (W):   {metrics['auc_roc_weighted']:.4f}")
            print(f"AUC-PR (W):    {metrics['auc_pr_weighted']:.4f}")
    
    print(f"{'='*70}\n")
    
    return {
        'metrics': metrics,
        'predictions': {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist()
        }
    }


# =============================================================================
# Results Saving
# =============================================================================

def save_results(results, save_dir, filename='test_results.json'):
    """
    Save evaluation results to JSON file.
    
    Args:
        results (dict): Evaluation results
        save_dir (str): Directory to save results
        filename (str): Filename for results
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    # Add timestamp
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved: {filepath}")


def load_results(filepath):
    """
    Load evaluation results from JSON file.
    
    Args:
        filepath (str): Path to results file
    
    Returns:
        dict: Evaluation results
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Results loaded: {filepath}")
    return results


# =============================================================================
# Classification Report
# =============================================================================

def print_classification_report(y_true, y_pred, class_names=None, task='binary'):
    """
    Print detailed classification report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Names of classes
        task (str): 'binary' or 'multi'
    """
    if class_names is None:
        if task == 'binary':
            class_names = ['Negative', 'Positive']
        else:
            class_names = ['Possible', 'Positive', 'Negative', 'Harmful']
    
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("="*70 + "\n")


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(results_dict, metric='f1'):
    """
    Compare multiple model results.
    
    Args:
        results_dict (dict): Dictionary of {model_name: results}
        metric (str): Metric to compare
    
    Returns:
        dict: Sorted comparison results
    """
    comparison = {}
    
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            metrics = results['metrics']
            if metric in metrics:
                comparison[model_name] = metrics[metric]
            elif f'{metric}_weighted' in metrics:
                comparison[model_name] = metrics[f'{metric}_weighted']
    
    # Sort by metric value (descending)
    comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n{'='*70}")
    print(f"Model Comparison ({metric.upper()}):")
    print(f"{'='*70}")
    for rank, (model_name, score) in enumerate(comparison.items(), 1):
        print(f"{rank}. {model_name:30s} {score:.4f}")
    print(f"{'='*70}\n")
    
    return comparison


def calculate_improvement(baseline_score, model_score):
    """
    Calculate percentage improvement over baseline.
    
    Args:
        baseline_score (float): Baseline score
        model_score (float): Model score
    
    Returns:
        float: Percentage improvement
    """
    if baseline_score == 0:
        return float('inf') if model_score > 0 else 0
    
    improvement = ((model_score - baseline_score) / baseline_score) * 100
    return improvement


# =============================================================================
# Statistical Significance Test
# =============================================================================

def perform_statistical_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to compare two models.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        dict: Test results
    """
    from scipy.stats import mcnemar
    
    # Create contingency table
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n00 = np.sum(~correct1 & ~correct2)  # Both wrong
    n01 = np.sum(~correct1 & correct2)   # Model 1 wrong, Model 2 correct
    n10 = np.sum(correct1 & ~correct2)   # Model 1 correct, Model 2 wrong
    n11 = np.sum(correct1 & correct2)    # Both correct
    
    # Contingency table for McNemar's test
    table = [[n11, n10], [n01, n00]]
    
    # Perform test
    result = mcnemar(table, exact=False, correction=True)
    
    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'significant': result.pvalue < 0.05,
        'table': table
    }