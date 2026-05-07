"""
KG-DFI Training Utilities
Handles model training, validation, and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

from model import FocalLoss


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' or 'max' for loss or metric
    """
    
    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def load_best_model(self, model):
        """Load the best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, graph, device, task='binary'):
    """
    Train for one epoch.
    
    Args:
        model: FDI model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        graph: Knowledge graph
        device: Device to use
        task: 'binary' or 'multi'
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch in pbar:
        food_ids = batch['food_id'].to(device)
        drug_ids = batch['drug_id'].to(device)
        labels = batch['label'].to(device)
        
        # Adjust label format for task
        if task == 'binary':
            labels = labels.float().unsqueeze(1)
        else:
            labels = labels.long()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(graph, food_ids, drug_ids)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, graph, device, task='binary'):
    """
    Validate the model.
    
    Args:
        model: FDI model
        val_loader: Validation dataloader
        criterion: Loss function
        graph: Knowledge graph
        device: Device to use
        task: 'binary' or 'multi'
    
    Returns:
        tuple: (val_loss, predictions, labels, probabilities)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            food_ids = batch['food_id'].to(device)
            drug_ids = batch['drug_id'].to(device)
            labels = batch['label'].to(device)
            
            # Adjust label format for task
            if task == 'binary':
                labels_input = labels.float().unsqueeze(1)
            else:
                labels_input = labels.long()
            
            # Forward pass
            outputs = model(graph, food_ids, drug_ids)
            loss = criterion(outputs, labels_input)
            
            total_loss += loss.item()
            num_batches += 1
            
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
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# =============================================================================
# Complete Training Loop
# =============================================================================

def train_model(model, train_loader, val_loader, graph, config, device='cuda'):
    """
    Complete training loop with early stopping.
    
    Args:
        model: FDI model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        graph: Knowledge graph
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        tuple: (best_model, history)
    """
    task = config.get('task', 'binary')
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 0.001)
    patience = config.get('patience', 15)
    
    print(f"\n{'='*70}")
    print(f"Starting training: {task.upper()} classification")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Patience: {patience}")
    print(f"{'='*70}\n")
    
    # Setup loss function
    if task == 'binary':
        alpha = config.get('focal_alpha', 0.25)
        criterion = FocalLoss(task='binary', alpha=alpha, gamma=config.get('focal_gamma', 2.0))
    else:
        class_weights = config.get('class_weights', None)
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = FocalLoss(task='multi', alpha=class_weights, gamma=config.get('focal_gamma', 2.0))
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 5),
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=config.get('min_delta', 1e-4))
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # Move graph to device
    graph = graph.to(device)
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, graph, device, task)
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, graph, device, task
        )
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        if task == 'binary':
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        else:
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        
        # Update learning rate
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val F1:     {val_f1:.4f}")
        print(f"LR:         {current_lr:.6f}")
        
        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"✅ New best F1: {best_val_f1:.4f}")
        
        # Early stopping check
        if early_stopping(val_f1, model):
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"{'='*70}\n")
    
    return model, history


# =============================================================================
# Save/Load Checkpoint
# =============================================================================

def save_checkpoint(model, optimizer, epoch, config, history, filepath):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        config: Configuration dict
        history: Training history
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded: {filepath}")
    return checkpoint


# =============================================================================
# Quick Training Function
# =============================================================================

def quick_train(task='binary', data_dir='data/', **kwargs):
    """
    Quick training with default settings.
    Convenience function for simple use cases.
    
    Args:
        task: 'binary' or 'multi'
        data_dir: Data directory
        **kwargs: Override default config
    
    Returns:
        tuple: (trained_model, history)
    """
    from dataloader import load_all_data
    from model import create_fdi_model
    
    # Default config
    config = {
        'task': task,
        'embedding_dim': 256 if task == 'binary' else 128,
        'batch_size': 512 if task == 'binary' else 64,
        'num_epochs': 300 if task == 'binary' else 200,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'patience': 50 if task == 'binary' else 30,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'kg_layers': 3 if task == 'binary' else 4,
        'use_attention': True,
        'num_classes': 4 if task == 'multi' else 2
    }
    
    # Override with kwargs
    config.update(kwargs)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_all_data(
        task=task,
        batch_size=config['batch_size'],
        num_workers=4,
        data_dir=data_dir
    )
    
    # Add class weights to config
    if data['class_weights'] is not None:
        config['class_weights'] = data['class_weights'].to(device)
    
    # Create model
    model = create_fdi_model(
        config,
        num_entities=data['mappings']['num_nodes'],
        num_relations=data['mappings']['num_relations'],
        device=device
    )
    
    # Train
    trained_model, history = train_model(
        model,
        data['train_loader'],
        data['val_loader'],
        data['graph'],
        config,
        device=device
    )
    
    return trained_model, history, data