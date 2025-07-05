import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from src.models.architectures.base_model import BaseModel
from src.constants.taxonomy_labels import TAXONOMY_LABELS, TAXONOMY_LEVELS


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer for hierarchical taxonomy classification.
    
    This layer performs message passing between nodes in the taxonomic hierarchy.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize graph convolution layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Learnable bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Normalize adjacency matrix
        adj_norm = self._normalize_adjacency(adj)
        
        # Graph convolution: H = σ(D^(-1/2) * A * D^(-1/2) * X * W + b)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj_norm, support)
        
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix using symmetric normalization.
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        
        return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)


class AttentionLayer(nn.Module):
    """
    Attention layer for weighted aggregation of node features.
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize attention layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Attention weights
        self.attention = nn.Linear(in_features * 2, 1)
        
        # Feature transformation
        self.transform = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Compute attention scores for all pairs
        attention_scores = torch.zeros(num_nodes, num_nodes, device=x.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:  # Only compute attention for connected nodes
                    # Concatenate features of nodes i and j
                    concat_features = torch.cat([x[i], x[j]], dim=0)
                    attention_scores[i, j] = self.attention(concat_features)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted aggregation
        weighted_features = torch.mm(attention_weights, x)
        
        # Transform features
        output = self.transform(weighted_features)
        
        return output


class GNNHierarchicalModel(BaseModel):
    """
    Graph Neural Network model for hierarchical taxonomy classification.
    
    This model represents taxonomic levels as nodes in a graph and uses
    graph convolutions to capture hierarchical relationships.
    """
    
    def __init__(self,
                input_size: int,
                hidden_sizes: List[int] = [256, 128],
                gnn_layers: int = 2,
                use_attention: bool = True,
                dropout: float = 0.3,
                name: str = "GNNHierarchicalModel"):
        """
        Initialize GNN hierarchical model.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            gnn_layers: Number of GNN layers
            use_attention: Whether to use attention mechanism
            dropout: Dropout probability
            name: Model name
        """
        super().__init__(name=name)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.gnn_layers = gnn_layers
        self.use_attention = use_attention
        self.dropout = dropout
        
        # Build feature extractor
        self.feature_extractor = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.feature_extractor.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Build GNN layers
        self.gnn_layers_list = nn.ModuleList()
        gnn_input_size = hidden_sizes[-1]
        
        for _ in range(gnn_layers):
            if use_attention:
                self.gnn_layers_list.append(AttentionLayer(gnn_input_size, gnn_input_size))
            else:
                self.gnn_layers_list.append(GraphConvolution(gnn_input_size, gnn_input_size))
        
        # Build output heads for each taxonomic level
        self.output_heads = nn.ModuleDict()
        for level in TAXONOMY_LEVELS:
            if level in TAXONOMY_LABELS:
                num_classes = len(TAXONOMY_LABELS[level])
                self.output_heads[level] = nn.Linear(gnn_input_size, num_classes)
        
        # Create hierarchical adjacency matrix
        self.register_buffer('adjacency_matrix', self._create_hierarchical_adjacency())
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def _create_hierarchical_adjacency(self) -> torch.Tensor:
        """
        Create adjacency matrix representing hierarchical relationships.
        
        Returns:
            Adjacency matrix [num_levels, num_levels]
        """
        num_levels = len(TAXONOMY_LEVELS)
        adj = torch.zeros(num_levels, num_levels)
        
        # Define hierarchical relationships
        hierarchy = {
            'kingdom_name': ['phylum_name'],
            'phylum_name': ['class_name'],
            'class_name': ['order_name'],
            'order_name': []
        }
        
        # Build adjacency matrix
        for i, level in enumerate(TAXONOMY_LEVELS):
            if level in hierarchy:
                for child in hierarchy[level]:
                    if child in TAXONOMY_LEVELS:
                        j = TAXONOMY_LEVELS.index(child)
                        adj[i, j] = 1.0  # Parent to child
                        adj[j, i] = 1.0  # Child to parent (bidirectional)
        
        # Add self-loops
        adj += torch.eye(num_levels)
        
        return adj
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN hierarchical model.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Dictionary mapping taxonomic levels to their logits
        """
        batch_size = x.size(0)
        num_levels = len(TAXONOMY_LEVELS)
        
        # Feature extraction
        features = x
        for layer in self.feature_extractor:
            features = layer(features)
            features = F.relu(features)
            features = self.dropout_layer(features)
        
        # Create node features for each taxonomic level
        # Each level gets the same shared features initially
        node_features = features.unsqueeze(1).expand(-1, num_levels, -1)  # [batch_size, num_levels, hidden_size]
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers_list:
            # Process each sample in the batch
            updated_features = []
            for i in range(batch_size):
                sample_features = node_features[i]  # [num_levels, hidden_size]
                
                if self.use_attention:
                    updated_sample = gnn_layer(sample_features, self.adjacency_matrix)
                else:
                    updated_sample = gnn_layer(sample_features, self.adjacency_matrix)
                    updated_sample = F.relu(updated_sample)
                
                updated_features.append(updated_sample)
            
            node_features = torch.stack(updated_features, dim=0)  # [batch_size, num_levels, hidden_size]
            node_features = self.dropout_layer(node_features)
        
        # Generate predictions for each level
        outputs = {}
        for i, level in enumerate(TAXONOMY_LEVELS):
            if level in self.output_heads:
                level_features = node_features[:, i, :]  # [batch_size, hidden_size]
                outputs[level] = self.output_heads[level](level_features)
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'gnn_layers': self.gnn_layers,
            'use_attention': self.use_attention,
            'dropout': self.dropout
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'GNNHierarchicalModel':
        """
        Load a GNN hierarchical model from a checkpoint.
        
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
            hidden_sizes=config['hidden_sizes'],
            gnn_layers=config.get('gnn_layers', 2),
            use_attention=config.get('use_attention', True),
            dropout=config.get('dropout', 0.3),
            name=config.get('name', 'GNNHierarchicalModel')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class GNHLoss(nn.Module):
    """
    Loss function for GNN hierarchical classification.
    
    Combines classification loss with graph structure regularization.
    """
    
    def __init__(self, 
                level_weights: Optional[Dict[str, float]] = None,
                graph_weight: float = 0.1,
                consistency_weight: float = 0.05):
        """
        Initialize GNN loss.
        
        Args:
            level_weights: Dictionary mapping taxonomic levels to their loss weights
            graph_weight: Weight for graph structure regularization
            consistency_weight: Weight for hierarchical consistency
        """
        super().__init__()
        
        # Default equal weights if not provided
        if level_weights is None:
            level_weights = {level: 1.0 for level in TAXONOMY_LEVELS}
        self.level_weights = level_weights
        self.graph_weight = graph_weight
        self.consistency_weight = consistency_weight
        
        # Classification loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Hierarchical order
        self.hierarchy_order = ['kingdom_name', 'phylum_name', 'class_name', 'order_name']
    
    def _graph_structure_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute graph structure regularization loss.
        
        Encourages predictions to respect hierarchical relationships.
        """
        structure_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for i in range(len(self.hierarchy_order) - 1):
            parent_level = self.hierarchy_order[i]
            child_level = self.hierarchy_order[i + 1]
            
            if parent_level in predictions and child_level in predictions:
                parent_probs = F.softmax(predictions[parent_level], dim=1)
                child_probs = F.softmax(predictions[child_level], dim=1)
                
                # Encourage child predictions to be more specific when parent is confident
                parent_confidence = torch.max(parent_probs, dim=1)[0]
                child_entropy = -torch.sum(child_probs * torch.log(child_probs + 1e-8), dim=1)
                
                structure_loss = structure_loss + torch.mean(parent_confidence * child_entropy)
        
        return structure_loss
    
    def _hierarchical_consistency_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute hierarchical consistency loss.
        
        Ensures that predictions are consistent across levels.
        """
        consistency_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        for level in TAXONOMY_LEVELS:
            if level in predictions:
                probs = F.softmax(predictions[level], dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                # Encourage confident predictions
                consistency_loss = consistency_loss + torch.mean(1.0 - max_probs)
        
        return consistency_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute GNN loss.
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            
        Returns:
            Combined loss tensor
        """
        # Classification loss
        classification_loss = 0.0
        for level in TAXONOMY_LEVELS:
            if level in predictions and level in targets:
                level_loss = self.criterion(predictions[level], targets[level])
                classification_loss += self.level_weights[level] * level_loss
        
        # Graph structure loss
        graph_loss = self._graph_structure_loss(predictions)
        
        # Hierarchical consistency loss
        consistency_loss = self._hierarchical_consistency_loss(predictions)
        
        # Combine all losses
        total_loss = (classification_loss + 
                     self.graph_weight * graph_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss


class GNNAccuracy:
    """
    Accuracy metrics for GNN hierarchical classification.
    """
    
    @staticmethod
    def compute_accuracy(predictions: Dict[str, torch.Tensor], 
                        targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute accuracy for each taxonomic level.
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            
        Returns:
            Dictionary mapping taxonomic levels to their accuracies
        """
        accuracies = {}
        
        for level in TAXONOMY_LEVELS:
            if level in predictions and level in targets:
                pred = predictions[level].argmax(dim=1)
                correct = pred.eq(targets[level]).sum().item()
                total = targets[level].size(0)
                accuracies[level] = correct / total if total > 0 else 0.0
            else:
                accuracies[level] = 0.0
        
        return accuracies
    
    @staticmethod
    def compute_hierarchical_accuracy(predictions: Dict[str, torch.Tensor], 
                                    targets: Dict[str, torch.Tensor]) -> float:
        """
        Compute hierarchical accuracy (all levels correct).
        
        Args:
            predictions: Dictionary of predictions for each taxonomic level
            targets: Dictionary of targets for each taxonomic level
            
        Returns:
            Hierarchical accuracy (fraction of samples where all levels are correct)
        """
        if not predictions or not targets:
            return 0.0
        
        # Get predictions and targets for all levels
        all_correct = torch.ones(targets[list(targets.keys())[0]].size(0), dtype=torch.bool, device=targets[list(targets.keys())[0]].device)
        
        for level in TAXONOMY_LEVELS:
            if level in predictions and level in targets:
                pred = predictions[level].argmax(dim=1)
                level_correct = pred.eq(targets[level])
                all_correct = all_correct & level_correct
        
        return all_correct.float().mean().item() 