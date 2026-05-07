"""
KG-DFI Model Definitions
Contains all model architectures and loss functions for Food-Drug Interaction prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgl
from dgl.nn.pytorch import RelGraphConv
import numpy as np


# =============================================================================
# Dataset Class
# =============================================================================

class FDI_Dataset(Dataset):
    """
    PyTorch Dataset for Food-Drug Interaction data.
    Supports both binary and multi-class classification.
    
    Args:
        food_ids (torch.Tensor): Food entity IDs
        drug_ids (torch.Tensor): Drug entity IDs
        labels (torch.Tensor): Interaction labels
        task (str): 'binary' or 'multi'
    """
    
    def __init__(self, food_ids, drug_ids, labels, task='binary'):
        self.food_ids = food_ids
        self.drug_ids = drug_ids
        self.labels = labels
        self.task = task
        
        assert len(food_ids) == len(drug_ids) == len(labels), \
            "All inputs must have the same length"
    
    def __len__(self):
        return len(self.food_ids)
    
    def __getitem__(self, idx):
        return {
            'food_id': self.food_ids[idx],
            'drug_id': self.drug_ids[idx],
            'label': self.labels[idx]
        }


# =============================================================================
# Knowledge Graph Encoder
# =============================================================================

class KnowledgeGraphEncoder(nn.Module):
    """
    RGCN-based Knowledge Graph Encoder.
    Encodes entities using Relational Graph Convolutional Networks.
    
    Args:
        num_entities (int): Number of entities in the KG
        num_relations (int): Number of relation types
        embedding_dim (int): Dimension of entity embeddings
        num_layers (int): Number of RGCN layers
        num_bases (int): Number of bases for basis decomposition
        dropout (float): Dropout rate
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim, 
                 num_layers=3, num_bases=8, dropout=0.1):
        super(KnowledgeGraphEncoder, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # RGCN layers
        self.rgcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.rgcn_layers.append(
                RelGraphConv(
                    embedding_dim, 
                    embedding_dim, 
                    num_relations,
                    num_bases=num_bases,
                    activation=F.relu if i < num_layers - 1 else None
                )
            )
            self.layer_norms.append(nn.LayerNorm(embedding_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
    
    def forward(self, graph, entity_ids=None):
        """
        Forward pass through KG encoder.
        
        Args:
            graph (DGLGraph): Knowledge graph
            entity_ids (torch.Tensor, optional): Specific entity IDs to encode.
                                                 If None, encode all entities.
        
        Returns:
            torch.Tensor: Entity embeddings
        """
        # Get initial embeddings
        if entity_ids is None:
            h = self.entity_embeddings.weight
        else:
            h = self.entity_embeddings(entity_ids)
        
        # Pass through RGCN layers with residual connections
        for i, (rgcn_layer, layer_norm, dropout) in enumerate(
            zip(self.rgcn_layers, self.layer_norms, self.dropouts)
        ):
            h_new = rgcn_layer(graph, h, graph.edata['type'])
            h_new = layer_norm(h_new)
            h_new = dropout(h_new)
            
            # Residual connection (skip connection)
            if i > 0 and h.size() == h_new.size():
                h = h + h_new
            else:
                h = h_new
        
        return h


# =============================================================================
# Cross Attention Module
# =============================================================================

class CrossAttention(nn.Module):
    """
    Multi-head Cross Attention between food and drug embeddings.
    Captures interaction patterns between food and drug entities.
    
    Args:
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, embedding_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads"
        
        # Linear projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, food_emb, drug_emb):
        """
        Apply cross attention.
        
        Args:
            food_emb (torch.Tensor): Food embeddings [batch_size, embedding_dim]
            drug_emb (torch.Tensor): Drug embeddings [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: Attended food embeddings [batch_size, embedding_dim]
            torch.Tensor: Attention weights [batch_size, num_heads, 1, 1]
        """
        batch_size = food_emb.size(0)
        
        # Multi-head attention projections
        Q = self.query(food_emb).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(drug_emb).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(drug_emb).view(batch_size, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, self.embedding_dim)
        
        # Residual connection and layer norm
        output = self.layer_norm(food_emb + attended)
        
        return output, attention_weights


# =============================================================================
# Enhanced Predictor
# =============================================================================

class EnhancedPredictor(nn.Module):
    """
    Multi-layer predictor with residual connections.
    Predicts interaction labels from concatenated food-drug embeddings.
    
    Args:
        embedding_dim (int): Dimension of input embeddings
        output_dim (int): Output dimension (1 for binary, num_classes for multi)
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of prediction layers
        dropout (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, embedding_dim, output_dim=1, hidden_dim=64, 
                 num_layers=4, dropout=0.2, use_batch_norm=True):
        super(EnhancedPredictor, self).__init__()
        
        input_dim = embedding_dim * 2  # Concatenated food + drug
        
        # Define layer dimensions
        dims = [input_dim] + [hidden_dim * (2 ** max(0, i - 1)) 
                              for i in range(num_layers)] + [output_dim]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer_list = [nn.Linear(dims[i], dims[i + 1])]
            
            # Add batch norm and activation for hidden layers
            if i < len(dims) - 2:
                if use_batch_norm:
                    layer_list.append(nn.BatchNorm1d(dims[i + 1]))
                layer_list.append(nn.ReLU())
                layer_list.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_list))
        
        # Residual connection
        self.residual = nn.Linear(input_dim, dims[-2])
    
    def forward(self, food_emb, drug_emb):
        """
        Predict interaction from embeddings.
        
        Args:
            food_emb (torch.Tensor): Food embeddings [batch_size, embedding_dim]
            drug_emb (torch.Tensor): Drug embeddings [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: Predictions [batch_size, output_dim]
        """
        # Concatenate embeddings
        x = torch.cat([food_emb, drug_emb], dim=1)
        x_input = x
        
        # Pass through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
        
        # Add residual connection before final layer
        if x.size(1) == self.residual(x_input).size(1):
            x = x + self.residual(x_input)
        
        # Final prediction layer
        output = self.layers[-1](x)
        
        return output


# =============================================================================
# FDI Model (Main Model)
# =============================================================================

class FDIModel(nn.Module):
    """
    Complete Food-Drug Interaction prediction model.
    Integrates KG encoder, attention, and predictor.
    
    Args:
        num_entities (int): Number of entities in KG
        num_relations (int): Number of relation types
        embedding_dim (int): Dimension of embeddings
        task (str): 'binary' or 'multi'
        num_classes (int): Number of classes (for multi-class)
        use_attention (bool): Whether to use cross attention
        **kwargs: Additional arguments for sub-modules
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim,
                 task='binary', num_classes=4, use_attention=True, **kwargs):
        super(FDIModel, self).__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Determine output dimension
        output_dim = 1 if task == 'binary' else num_classes
        
        # Knowledge Graph Encoder
        self.kg_encoder = KnowledgeGraphEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_layers=kwargs.get('kg_layers', 3),
            num_bases=kwargs.get('kg_bases', 8),
            dropout=kwargs.get('kg_dropout', 0.1)
        )
        
        # Cross Attention (optional)
        if use_attention:
            self.attention = CrossAttention(
                embedding_dim=embedding_dim,
                num_heads=kwargs.get('num_heads', 4)
            )
        
        # Predictor
        self.predictor = EnhancedPredictor(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dim=kwargs.get('pred_hidden_dim', 64),
            num_layers=kwargs.get('pred_layers', 4),
            dropout=kwargs.get('pred_dropout', 0.2),
            use_batch_norm=kwargs.get('use_batch_norm', True)
        )
    
    def forward(self, graph, food_ids, drug_ids):
        """
        Forward pass.
        
        Args:
            graph (DGLGraph): Knowledge graph
            food_ids (torch.Tensor): Food entity IDs [batch_size]
            drug_ids (torch.Tensor): Drug entity IDs [batch_size]
        
        Returns:
            torch.Tensor: Predictions [batch_size, output_dim]
        """
        # Encode all entities
        entity_embeddings = self.kg_encoder(graph)
        
        # Get food and drug embeddings
        food_emb = entity_embeddings[food_ids]
        drug_emb = entity_embeddings[drug_ids]
        
        # Apply attention if enabled
        if self.use_attention:
            food_emb, _ = self.attention(food_emb, drug_emb)
        
        # Predict interaction
        logits = self.predictor(food_emb, drug_emb)
        
        # Apply activation based on task
        if self.task == 'binary':
            return torch.sigmoid(logits)
        else:
            return logits  # Return logits for CrossEntropyLoss


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Supports both binary and multi-class classification.
    
    Args:
        task (str): 'binary' or 'multi'
        alpha (float or torch.Tensor): Weighting factor
        gamma (float): Focusing parameter
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, task='binary', alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.task = task
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels
        
        Returns:
            torch.Tensor: Loss value
        """
        if self.task == 'binary':
            # Binary focal loss
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            p_t = torch.where(targets == 1, inputs, 1 - inputs)
            
            if self.alpha is not None:
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = 1.0
            
            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            focal_loss = focal_weight * bce_loss
        
        else:
            # Multi-class focal loss
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, 
                                     reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# Model Initialization Helper
# =============================================================================

def create_fdi_model(config, num_entities, num_relations, device='cuda'):
    """
    Factory function to create FDI model from config.
    
    Args:
        config (dict): Configuration dictionary
        num_entities (int): Number of entities
        num_relations (int): Number of relations
        device (str): Device to place model on
    
    Returns:
        FDIModel: Initialized model
    """
    model = FDIModel(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config.get('embedding_dim', 256),
        task=config.get('task', 'binary'),
        num_classes=config.get('num_classes', 4),
        use_attention=config.get('use_attention', True),
        kg_layers=config.get('kg_layers', 3),
        kg_bases=config.get('kg_bases', 8),
        kg_dropout=config.get('kg_dropout', 0.1),
        pred_hidden_dim=config.get('pred_hidden_dim', 64),
        pred_layers=config.get('pred_layers', 4),
        pred_dropout=config.get('pred_dropout', 0.2),
        use_batch_norm=config.get('use_batch_norm', True),
        num_heads=config.get('num_heads', 4)
    ).to(device)
    
    return model


# =============================================================================
# Model Information
# =============================================================================

def get_model_info(model):
    """
    Get model information including parameter count.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        dict: Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    return info