import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from models.architectures.base_model import BaseModel

class BasicTaxoModel(BaseModel):
    """Simple MLP model for taxonomy classification."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, name: str = "BasicMLP"):
        """
        Initialize the basic model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Number of classes
            name: Model name
        """
        super().__init__(name=name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'BasicTaxoModel':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            name=config.get('name', 'BasicMLP')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model