import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from models.architectures.base_model import BaseModel

class CNNModel(BaseModel):
    """CNN model for sequence-based taxonomy classification."""
    
    def __init__(self,
                input_size: int,
                output_size: int,
                kernel_sizes: List[int] = [3, 5, 7],
                num_filters: List[int] = [64, 128, 256],
                fc_sizes: List[int] = [512, 256],
                dropout: float = 0.3,
                name: str = "CNN"):
        """
        Initialize CNN model.
        
        Args:
            input_size: Size of input features
            output_size: Number of classes
            kernel_sizes: List of kernel sizes for conv layers
            num_filters: List of filter counts for conv layers
            fc_sizes: List of fully connected layer sizes
            dropout: Dropout probability
            name: Model name
        """
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.fc_sizes = fc_sizes
        self.dropout = dropout
        
        assert len(kernel_sizes) == len(num_filters), "Must provide same number of kernel sizes and filters"
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_channels = 1  # Initial input has 1 channel
        
        for i, (k, f) in enumerate(zip(kernel_sizes, num_filters)):
            conv = nn.Conv1d(in_channels, f, kernel_size=k, padding=k//2)
            self.conv_layers.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(f))
            in_channels = f
        
        # Calculate size after convolutions
        # For 3 conv layers with max pooling, feature map is reduced by factor of 2^3
        conv_output_size = (input_size // (2 ** len(kernel_sizes))) * num_filters[-1]
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = conv_output_size
        
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            prev_size = fc_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension [batch_size, features] -> [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # Apply convolutions
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.max_pool1d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
        # Apply output layer
        x = self.output_layer(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'output_size': self.output_size,
            'kernel_sizes': self.kernel_sizes,
            'num_filters': self.num_filters,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'CNNModel':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            output_size=config['output_size'],
            kernel_sizes=config['kernel_sizes'],
            num_filters=config['num_filters'],
            fc_sizes=config['fc_sizes'],
            dropout=config.get('dropout', 0.3),
            name=config.get('name', 'CNN')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model