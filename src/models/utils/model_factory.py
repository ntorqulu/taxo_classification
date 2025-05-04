from typing import Dict, Any, Optional, Union, List, Tuple
from ..architectures.cnn_classifier import CNNClassifier
from ..architectures.base_model import BaseModel

def create_model(model_type: str, model_params: Dict[str, Any]) -> BaseModel:
    """
    Create a model based on type and parameters.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    model_params : Dict[str, Any]
        Parameters for the model
        
    Returns:
    --------
    BaseModel
        Created model
    """
    if model_type.lower() == 'cnn':
        return CNNClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_taxonomy_classifier(
    num_classes: int,
    input_size: int,
    model_type: str = 'cnn',
    input_channels: int = 1,
    **kwargs
) -> BaseModel:
    """
    Create a taxonomy classifier with sensible defaults.
    
    Parameters:
    -----------
    num_classes : int
        Number of taxonomic classes to predict
    input_size : int
        Size of input features
    model_type : str
        Type of model to create
    input_channels : int
        Number of input channels (1 for vectors, multiple for one-hot encoding)
    **kwargs : Any
        Additional parameters to pass to the model
        
    Returns:
    --------
    BaseModel
        Created model
    """
    if model_type.lower() == 'cnn':
        # Default parameters for CNN
        params = {
            'input_size': input_size,
            'num_classes': num_classes,
            'conv_channels': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'fc_sizes': [512, 256],
            'dropout': 0.5,
            'input_channels': input_channels
        }
        
        # Override defaults with provided kwargs
        params.update(kwargs)
        
        return CNNClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")