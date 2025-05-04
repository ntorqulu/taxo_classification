import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from .base_model import BaseModel

class CNNClassifier(BaseModel):
    """
    CNN-based sequence classifier for taxonomic classification.
    
    This model uses 1D convolutions to extract features from sequences,
    followed by max pooling and fully connected layers for classification.
    It works with various input formats:
    - One-hot encoded sequences
    - K-mer frequency vectors
    - Numerical encodings
    """
    
    def __init__(self, 
                 input_size: int,
                 num_classes: int,
                 conv_channels: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 fc_sizes: List[int] = [512, 256],
                 dropout: float = 0.5,
                 input_channels: int = 1,
                 input_length: Optional[int] = None):
        """
        Initialize the CNN classifier.
        
        Parameters:
        -----------
        input_size : int
            Size of input features or sequence length for one-hot encoding
        num_classes : int
            Number of classes to predict
        conv_channels : List[int]
            Number of channels for each convolutional layer
        kernel_sizes : List[int]
            Kernel sizes for convolutional layers
        fc_sizes : List[int]
            Sizes of fully connected layers
        dropout : float
            Dropout probability
        input_channels : int
            Number of input channels (1 for vectors, multiple for one-hot encoding)
        input_length : int, optional
            Length of input sequences (required for one-hot encoding)
        """
        super().__init__(name="cnn_classifier", num_classes=num_classes)
        
        self.input_size = input_size
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.fc_sizes = fc_sizes
        self.dropout_rate = dropout
        self.input_channels = input_channels
        self.input_length = input_length or input_size
        
        # Create parallel convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_sequence = []
            in_channels = input_channels
            
            for out_channels in conv_channels:
                # Add padding to maintain sequence length
                padding = kernel_size // 2
                conv_sequence.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
                conv_sequence.append(nn.BatchNorm1d(out_channels))
                conv_sequence.append(nn.ReLU())
                in_channels = out_channels
            
            self.conv_layers.append(nn.Sequential(*conv_sequence))
        
        # Calculate output size after convolutions and pooling
        # Each parallel path produces output_channels[-1] features
        conv_output_size = conv_channels[-1] * len(kernel_sizes)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = conv_output_size
        
        for out_features in fc_sizes:
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append(nn.BatchNorm1d(out_features))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            in_features = out_features
        
        # Final classification layer
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CNN classifier.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, features) or (batch_size, channels, seq_length)
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary with 'logits' and 'features' keys
        """
        batch_size = x.size(0)
        
        # Reshape input if necessary
        if x.dim() == 2:
            # Convert from (batch_size, features) to (batch_size, channels, length)
            if self.input_channels == 1:
                x = x.unsqueeze(1)  # Add channel dimension
            else:
                # Reshape to (batch_size, channels, length) for one-hot encoding
                x = x.view(batch_size, self.input_channels, -1)
        
        # Apply each convolutional path
        conv_outputs = []
        for conv_path in self.conv_layers:
            # Each path processes the input independently
            out = conv_path(x)
            # Global max pooling
            out = F.adaptive_max_pool1d(out, 1).squeeze(-1)
            conv_outputs.append(out)
        
        # Concatenate outputs from different convolutional paths
        features = torch.cat(conv_outputs, dim=1)
        
        # Apply fully connected layers
        for i in range(0, len(self.fc_layers), 4):
            fc_block = self.fc_layers[i:i+4]
            for layer in fc_block:
                features = layer(features)
        
        # Final classification
        logits = self.classifier(features)
        
        return {'logits': logits, 'features': features}
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'fc_sizes': self.fc_sizes,
            'dropout': self.dropout_rate,
            'input_channels': self.input_channels,
            'input_length': self.input_length
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'CNNClassifier':
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            conv_channels=config['conv_channels'],
            kernel_sizes=config['kernel_sizes'],
            fc_sizes=config['fc_sizes'],
            dropout=config['dropout'],
            input_channels=config['input_channels'],
            input_length=config['input_length']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model