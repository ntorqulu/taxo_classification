import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from src.models.architectures.base_model import BaseModel
from src.constants.taxonomy_labels import TAXONOMY_LABELS, TAXONOMY_LEVELS


class CascadeHierarchicalModel(BaseModel):
    """
    Cascade hierarchical model where predictions flow from higher to lower levels.
    
    Each level's prediction is used as additional input for the next level,
    creating a cascade effect that leverages hierarchical relationships.
    """
    
    def __init__(self,
                input_size: int,
                shared_hidden_sizes: List[int] = [512, 256],
                level_specific_sizes: Optional[Dict[str, List[int]]] = None,
                num_classes_per_level: Optional[Dict[str, int]] = None,
                target_levels: Optional[List[str]] = None,
                dropout: float = 0.3,
                use_confidence_weighting: bool = True,
                name: str = "CascadeHierarchicalModel"):
        super().__init__(name=name)
        self.input_size = input_size
        self.shared_hidden_sizes = shared_hidden_sizes
        self.dropout = dropout
        self.use_confidence_weighting = use_confidence_weighting
        
        # Use target_levels if provided, otherwise use basic levels
        if target_levels is not None:
            self.target_levels = target_levels
        else:
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
        
        # Build cascade components
        self.cascade_components = nn.ModuleDict()
        self.cascade_order = self.target_levels  # Use target_levels as cascade order
        
        for i, level in enumerate(self.cascade_order):
            # Use provided num_classes_per_level or fall back to hardcoded values
            if num_classes_per_level and level in num_classes_per_level:
                num_classes = num_classes_per_level[level]
            elif level in TAXONOMY_LABELS:
                num_classes = len(TAXONOMY_LABELS[level])
            else:
                # Skip levels that don't have class information
                continue
            
            # Determine input size for this level
            if i == 0:
                # First level only uses shared features
                level_input_size = shared_hidden_sizes[-1]
            else:
                # Subsequent levels use shared features + parent probabilities
                parent_level = self.cascade_order[i-1]
                if num_classes_per_level and parent_level in num_classes_per_level:
                    parent_classes = num_classes_per_level[parent_level]
                elif parent_level in TAXONOMY_LABELS:
                    parent_classes = len(TAXONOMY_LABELS[parent_level])
                else:
                    parent_classes = 0
                level_input_size = shared_hidden_sizes[-1] + parent_classes
            
            # Build level-specific layers
            level_layers = nn.ModuleList()
            prev_size = level_input_size
            
            for hidden_size in level_specific_sizes.get(level, [128, 64]):
                level_layers.append(nn.Linear(prev_size, hidden_size))
                prev_size = hidden_size
            
            # Output layer
            level_layers.append(nn.Linear(prev_size, num_classes))
            
            # Store the layers
            self.cascade_components[level] = level_layers
            
            # Add confidence weighting layer if enabled and not first level
            if self.use_confidence_weighting and i > 0:
                self.cascade_components[f'{level}_confidence'] = nn.Linear(1, 1)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the cascade hierarchical model.
        
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
        
        # Cascade predictions
        outputs = {}
        current_features = shared_features
        
        for i, level in enumerate(self.cascade_order):
            # Skip levels that don't have output heads
            if level not in self.cascade_components:
                continue
                
            # Determine input for this level
            if i == 0:
                # First level uses only shared features
                level_input = current_features
            else:
                # Subsequent levels use shared features + parent probabilities
                parent_level = self.cascade_order[i-1]
                if parent_level in outputs:
                    parent_output = outputs[parent_level]
                    parent_probs = F.softmax(parent_output, dim=1)
                    
                    # Apply confidence weighting if enabled
                    if self.use_confidence_weighting and f'{level}_confidence' in self.cascade_components:
                        parent_confidence = torch.max(parent_probs, dim=1, keepdim=True)[0]
                        confidence_layer = self.cascade_components[f'{level}_confidence']
                        confidence_weight = confidence_layer(parent_confidence)
                        # Apply sigmoid to ensure positive weights
                        confidence_weight = torch.sigmoid(confidence_weight)
                        parent_probs = parent_probs * confidence_weight
                    
                    # Concatenate shared features with parent probabilities
                    level_input = torch.cat([current_features, parent_probs], dim=1)
                else:
                    # If parent level is not available, use only shared features
                    level_input = current_features
            
            # Apply level-specific layers
            level_layers = self.cascade_components[level]
            for j, layer in enumerate(level_layers[:-1]):
                level_input = layer(level_input)
                level_input = F.relu(level_input)
                level_input = self.dropout_layer(level_input)
            
            # Final output layer (no activation, no dropout)
            level_output = level_layers[-1](level_input)
            outputs[level] = level_output
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'shared_hidden_sizes': self.shared_hidden_sizes,
            'level_specific_sizes': self.level_specific_sizes,
            'dropout': self.dropout,
            'use_confidence_weighting': self.use_confidence_weighting
        })
        return config

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'CascadeHierarchicalModel':
        """
        Load a cascade hierarchical model from a checkpoint.
        
        Args:
            path: Path to checkpoint file
            map_location: Device to load model to
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            shared_hidden_sizes=config['shared_hidden_sizes'],
            level_specific_sizes=config.get('level_specific_sizes'),
            num_classes_per_level=config.get('num_classes_per_level'),
            target_levels=config.get('target_levels'),
            dropout=config.get('dropout', 0.3),
            use_confidence_weighting=config.get('use_confidence_weighting', True),
            name=config.get('name', 'CascadeHierarchicalModel')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class CascadeLoss(nn.Module):
    """
    Loss function for cascade hierarchical classification.
    
    Combines classification loss with cascade consistency and confidence regularization.
    """
    
    def __init__(self, 
                level_weights: Optional[Dict[str, float]] = None,
                target_levels: Optional[List[str]] = None,
                cascade_weight: float = 0.1,
                confidence_weight: float = 0.05):
        """
        Initialize cascade loss.
        
        Args:
            level_weights: Dictionary mapping taxonomic levels to their loss weights
            target_levels: List of taxonomic levels to include in loss calculation
            cascade_weight: Weight for cascade consistency loss
            confidence_weight: Weight for confidence regularization
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
        
        self.cascade_weight = cascade_weight
        self.confidence_weight = confidence_weight
        
        # Classification loss
        self.classification_criterion = nn.CrossEntropyLoss()
    
    def _cascade_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cascade consistency loss.
        
        Encourages child predictions to be more confident when parent predictions are confident.
        """
        consistency_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for i in range(1, len(self.target_levels)):
            child_level = self.target_levels[i]
            parent_level = self.target_levels[i-1]
            
            if child_level in predictions and parent_level in predictions:
                parent_probs = F.softmax(predictions[parent_level], dim=1)
                child_probs = F.softmax(predictions[child_level], dim=1)
                
                # Parent confidence (max probability)
                parent_confidence = torch.max(parent_probs, dim=1)[0]
                
                # Child entropy (lower entropy = higher confidence)
                child_entropy = -torch.sum(child_probs * torch.log(child_probs + 1e-8), dim=1)
                
                # Encourage lower child entropy when parent is confident
                consistency_loss = consistency_loss + torch.mean(parent_confidence * child_entropy)
        
        return consistency_loss
    
    def _confidence_regularization(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute confidence regularization loss.
        
        Encourages models to make confident predictions.
        """
        confidence_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for level in self.target_levels:
            if level in predictions:
                probs = F.softmax(predictions[level], dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                # Penalize low confidence (encourage high max probability)
                confidence_loss = confidence_loss + torch.mean(1.0 - max_probs)
        
        return confidence_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cascade loss.
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            
        Returns:
            Combined loss tensor
        """
        # Classification loss
        classification_loss = 0.0
        for level in self.target_levels:
            if level in predictions and level in targets:
                level_loss = self.classification_criterion(predictions[level], targets[level])
                classification_loss += self.level_weights[level] * level_loss
        
        # Cascade consistency loss
        cascade_loss = self._cascade_consistency_loss(predictions)
        
        # Confidence regularization
        confidence_loss = self._confidence_regularization(predictions)
        
        # Combine all losses
        total_loss = (classification_loss + 
                     self.cascade_weight * cascade_loss + 
                     self.confidence_weight * confidence_loss)
        
        return total_loss


class CascadeAccuracy:
    """
    Accuracy metrics for cascade hierarchical classification.
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