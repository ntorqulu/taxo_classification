import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from src.models.architectures.base_model import BaseModel
from src.constants.taxonomy_labels import TAXONOMY_LABELS, TAXONOMY_LEVELS


class HierarchicalModel(BaseModel):
    """
    Hierarchical model for multi-level taxonomy classification.
    
    This model uses a shared feature extractor with multiple output heads,
    one for each taxonomic level (kingdom, phylum, class, order).
    """
    
    def __init__(self,
                input_size: int,
                shared_hidden_sizes: List[int] = [512, 256],
                level_specific_sizes: Optional[Dict[str, List[int]]] = None,
                num_classes_per_level: Optional[Dict[str, int]] = None,
                target_levels: Optional[List[str]] = None,
                dropout: float = 0.3,
                name: str = "HierarchicalModel"):
        """
        Initialize hierarchical model.
        
        Args:
            input_size: Size of input features
            shared_hidden_sizes: List of hidden layer sizes for shared feature extractor
            level_specific_sizes: Dict mapping taxonomic level to specific hidden layer sizes
            num_classes_per_level: Dict mapping taxonomic level to number of classes
            target_levels: List of taxonomic levels to train on (if None, uses all available levels from num_classes_per_level)
            dropout: Dropout probability
            name: Model name
        """
        super().__init__(name=name)
        self.input_size = input_size
        self.shared_hidden_sizes = shared_hidden_sizes
        self.dropout = dropout
        
        # Use target_levels if provided, otherwise use all available levels from num_classes_per_level
        if target_levels is not None:
            self.target_levels = target_levels
        elif num_classes_per_level is not None:
            self.target_levels = list(num_classes_per_level.keys())
        else:
            # Fallback to basic levels if nothing is provided
            self.target_levels = ['kingdom_name', 'phylum_name', 'class_name', 'order_name']
        
        # Default level-specific sizes if not provided
        if level_specific_sizes is None:
            level_specific_sizes = {
                'kingdom_name': [128, 64],
                'phylum_name': [128, 64], 
                'class_name': [128, 64],
                'order_name': [128, 64],
                'family_name': [128, 64],
                'genus_name': [128, 64],
                'species_name': [128, 64]
            }
        self.level_specific_sizes = level_specific_sizes
        
        # Build shared feature extractor
        self.shared_layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in shared_hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Build level-specific output heads
        self.level_heads = nn.ModuleDict()
        for level in self.target_levels:
            # Use provided num_classes_per_level or fall back to hardcoded values
            if num_classes_per_level and level in num_classes_per_level:
                num_classes = num_classes_per_level[level]
            elif level in TAXONOMY_LABELS:
                num_classes = len(TAXONOMY_LABELS[level])
            else:
                # Skip levels that don't have class information
                continue
            
            head_layers = nn.ModuleList()
            
            # Level-specific hidden layers
            prev_size = shared_hidden_sizes[-1]
            for hidden_size in level_specific_sizes.get(level, [128, 64]):
                head_layers.append(nn.Linear(prev_size, hidden_size))
                prev_size = hidden_size
            
            # Output layer
            head_layers.append(nn.Linear(prev_size, num_classes))
            
            self.level_heads[level] = head_layers
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical model.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Dictionary mapping taxonomic levels to their logits
        """
        # Shared feature extraction
        shared_features = x
        for layer in self.shared_layers:
            shared_features = layer(shared_features)
            shared_features = F.relu(shared_features)
            shared_features = self.dropout_layer(shared_features)
        
        # Level-specific predictions
        outputs = {}
        for level, head_layers in self.level_heads.items():
            level_features = shared_features
            
            # Apply level-specific layers
            for i, layer in enumerate(head_layers[:-1]):  # All except last layer
                level_features = layer(level_features)
                level_features = F.relu(level_features)
                level_features = self.dropout_layer(level_features)
            
            # Final output layer (no activation, no dropout)
            outputs[level] = head_layers[-1](level_features)
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'shared_hidden_sizes': self.shared_hidden_sizes,
            'level_specific_sizes': self.level_specific_sizes,
            'dropout': self.dropout
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'HierarchicalModel':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            shared_hidden_sizes=config['shared_hidden_sizes'],
            level_specific_sizes=config.get('level_specific_sizes'),
            num_classes_per_level=config.get('num_classes_per_level'),
            target_levels=config.get('target_levels'),
            dropout=config.get('dropout', 0.3),
            name=config.get('name', 'HierarchicalModel')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class HierarchicalLoss(nn.Module):
    """
    Loss function for hierarchical classification.
    
    Combines losses from all taxonomic levels with optional weighting.
    """
    
    def __init__(self, 
                level_weights: Optional[Dict[str, float]] = None,
                target_levels: Optional[List[str]] = None,
                loss_type: str = 'cross_entropy',
                focal_alpha: float = 1.0,
                focal_gamma: float = 2.0):
        """
        Initialize hierarchical loss.
        
        Args:
            level_weights: Dictionary mapping taxonomic levels to their loss weights
            target_levels: List of taxonomic levels to include in loss calculation
            loss_type: Type of loss function ('cross_entropy' or 'focal')
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        # Use target_levels if provided, otherwise use basic levels
        if target_levels is not None:
            self.target_levels = target_levels
        else:
            self.target_levels = ['kingdom_name', 'phylum_name', 'class_name', 'order_name']
        
        # Default equal weights if not provided
        if level_weights is None:
            level_weights = {level: 1.0 for level in self.target_levels}
        self.level_weights = level_weights
        
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            # Focal loss will be implemented in forward method
            self.criterion = None
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model
            targets: Target labels
            
        Returns:
            Focal loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute hierarchical loss.
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        
        for level in self.target_levels:
            if level in predictions and level in targets:
                if self.loss_type == 'cross_entropy':
                    level_loss = self.criterion(predictions[level], targets[level])
                elif self.loss_type == 'focal':
                    level_loss = self._focal_loss(predictions[level], targets[level])
                else:
                    raise ValueError(f"Unsupported loss type: {self.loss_type}")
                
                total_loss += self.level_weights[level] * level_loss
        
        return total_loss


class HierarchicalAccuracy:
    """
    Accuracy metrics for hierarchical classification.
    """
    
    @staticmethod
    def compute_accuracy(predictions: Dict[str, torch.Tensor], 
                        targets: Dict[str, torch.Tensor],
                        target_levels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute accuracy for each taxonomic level.
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            target_levels: List of taxonomic levels to compute accuracy for
            
        Returns:
            Dictionary mapping taxonomic levels to their accuracies
        """
        if target_levels is None:
            target_levels = ['kingdom_name', 'phylum_name', 'class_name', 'order_name']
            
        accuracies = {}
        
        for level in target_levels:
            if level in predictions and level in targets:
                pred = predictions[level].argmax(dim=1)
                correct = pred.eq(targets[level]).sum().item()
                total = targets[level].size(0)
                accuracies[level] = correct / total if total > 0 else 0.0
        
        return accuracies
    
    @staticmethod
    def compute_hierarchical_accuracy(predictions: Dict[str, torch.Tensor], 
                                    targets: Dict[str, torch.Tensor],
                                    target_levels: Optional[List[str]] = None) -> float:
        """
        Compute hierarchical accuracy (all levels correct).
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            target_levels: List of taxonomic levels to include in hierarchical accuracy
            
        Returns:
            Hierarchical accuracy (fraction of samples where all levels are correct)
        """
        if not predictions or not targets:
            return 0.0
        
        if target_levels is None:
            target_levels = ['kingdom_name', 'phylum_name', 'class_name', 'order_name']
        
        # Get the first level to determine batch size
        first_level = list(predictions.keys())[0]
        batch_size = predictions[first_level].size(0)
        
        # Check if all levels are correct for each sample
        all_correct = torch.ones(batch_size, dtype=torch.bool, device=predictions[first_level].device)
        
        for level in target_levels:
            if level in predictions and level in targets:
                pred = predictions[level].argmax(dim=1)
                level_correct = pred.eq(targets[level])
                all_correct = all_correct & level_correct
        
        return all_correct.float().mean().item() 