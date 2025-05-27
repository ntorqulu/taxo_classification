import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

class BaseModel(nn.Module, ABC):
    """Base class for all taxonomy classification models."""
    
    def __init__(self, name: str):
        """
        Initialize base model.
        
        Args:
            name: Unique name for the model
        """
        super().__init__()
        self.name = name
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
             extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer to save state
            extra_info: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.name,
            'model_config': self.get_config()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        if extra_info is not None:
            checkpoint.update(extra_info)
            
        torch.save(checkpoint, path)
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving/loading.
        
        Returns:
            Dictionary with model configuration
        """
        return {'name': self.name}
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'BaseModel':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            map_location: Device to load to
            
        Returns:
            Loaded model instance
        """
        raise NotImplementedError("Subclasses must implement load method")