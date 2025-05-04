import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple

class BaseModel(nn.Module, ABC):
    """Abstract base class for all taxonomy classification models."""
    
    def __init__(self, name: str, num_classes: int):
        """
        Initialize the base model.
        
        Parameters:
        -----------
        name : str
            Model name
        num_classes : int
            Number of classes to predict
        """
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing at least 'logits' key with model outputs
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make class predictions.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Class predictions (indices)
        """
        with torch.no_grad():
            outputs = self(x)
            return torch.argmax(outputs['logits'], dim=1)
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.name,
            'num_classes': self.num_classes,
            'model_config': self.get_config()
        }, path)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Model configuration dictionary
        """
        return {'name': self.name, 'num_classes': self.num_classes}
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'BaseModel':
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        map_location : str, optional
            Device to map the model to
            
        Returns:
        --------
        BaseModel
            Loaded model
        """
        # This is a placeholder - subclasses should implement this
        raise NotImplementedError("Subclasses must implement load method")