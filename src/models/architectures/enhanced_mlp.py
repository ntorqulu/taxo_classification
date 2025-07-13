import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from models.architectures.base_model import BaseModel

class EnhancedMLP(BaseModel):
    """Enhanced MLP with configurable layers and batch normalization."""
    
    def __init__(self, 
                input_size: int, 
                hidden_sizes: List[int],
                output_size: int, 
                dropout: float = 0.2,
                use_batch_norm: bool = True,
                name: str = "EnhancedMLP"):
        """
        Initialize enhanced MLP model.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of classes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            name: Model name
        """
        super().__init__(name=name)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build network
        layers = []
        prev_size = input_size
        
        for i, size in enumerate(hidden_sizes):
            # Add linear layer
            layers.append(nn.Linear(prev_size, size))
            
            # Add batch norm if requested
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
                
            # Add activation
            layers.append(nn.ReLU())
            
            # Add dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 4-row encoding input - flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]
        
        # Handle input size mismatch dynamically
        if x.size(1) != self.input_size:
            # Get the actual input size
            actual_input_size = x.size(1)
            
            # Rebuild the first layer with correct input size
            first_layer_output_size = self.hidden_sizes[0]
            new_first_layer = nn.Linear(actual_input_size, first_layer_output_size)
            new_first_layer.to(x.device)
            
            # Rebuild the entire model with correct input size
            layers = []
            layers.append(new_first_layer)
            
            # Add batch norm if requested
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(first_layer_output_size))
                
            # Add activation
            layers.append(nn.ReLU())
            
            # Add dropout
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            # Add remaining hidden layers
            prev_size = first_layer_output_size
            for i, size in enumerate(self.hidden_sizes[1:], 1):
                layers.append(nn.Linear(prev_size, size))
                
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(size))
                    
                layers.append(nn.ReLU())
                
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                    
                prev_size = size
            
            # Add output layer
            layers.append(nn.Linear(prev_size, self.output_size))
            
            # Replace the model
            self.model = nn.Sequential(*layers)
            self.input_size = actual_input_size
            
            print(f"Enhanced MLP: Adjusted input size from {self.input_size} to {actual_input_size}")
        
        return self.model(x)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm
        })
        # Add class_names if present
        if hasattr(self, 'class_names'):
            config['class_names'] = self.class_names
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'EnhancedMLP':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size'],
            dropout=config.get('dropout', 0.2),
            use_batch_norm=config.get('use_batch_norm', True),
            name=config.get('name', 'EnhancedMLP')
        )
        
        # Restore class_names if present
        if 'class_names' in config:
            model.class_names = config['class_names']
        model.load_state_dict(checkpoint['model_state_dict'])
        return model