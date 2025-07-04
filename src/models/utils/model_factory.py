from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from models.architectures.base_model import BaseModel
from models.architectures.basic_model import BasicTaxoModel
from models.architectures.cnn_model import CNNModel
from models.architectures.enhanced_mlp import EnhancedMLP
from models.architectures.nanni2024 import nanni_att, nanni_cnn1, nanni_cnn2
from models.architectures.cascade_hierarchical_model import CascadeHierarchicalModel
from models.architectures.hierarchical_model import HierarchicalModel
from models.architectures.gnn_hierarchical_model import GNNHierarchicalModel
from models.architectures.bert_model import BERTTaxoModel


def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory function to create model instances.

    Args:
        model_type: Type of model ('basic', 'enhanced_mlp', 'cnn')
        **kwargs: Model parameters

    Returns:
        Instantiated model
    """
    if model_type == "basic":
        # Get required parameters
        input_size = kwargs.get("input_size")
        output_size = kwargs.get("output_size")
        
        if input_size is None:
            raise ValueError("Missing required parameter: input_size")
        if output_size is None:
            raise ValueError("Missing required parameter: output_size")

        # For hidden_size, use a default if None
        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None:
            hidden_size = input_size // 2

        return BasicTaxoModel(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, name=kwargs.get("name", "BasicMLP")
        )

    elif model_type == "enhanced_mlp":
        required_params = ["input_size", "output_size"]
        _check_required_params(required_params, kwargs)
        return EnhancedMLP(
            input_size=kwargs["input_size"],
            hidden_sizes=kwargs.get("hidden_sizes", [256, 128]),
            output_size=kwargs["output_size"],
            dropout=kwargs.get("dropout", 0.2),
            use_batch_norm=kwargs.get("use_batch_norm", True),
            name=kwargs.get("name", "EnhancedMLP"),
        )

    elif model_type == "cnn":
        required_params = ["input_size", "output_size"]
        _check_required_params(required_params, kwargs)
        return CNNModel(
            input_size=kwargs["input_size"],
            output_size=kwargs["output_size"],
            kernel_sizes=kwargs.get("kernel_sizes", [3, 5, 7]),
            num_filters=kwargs.get("num_filters", [64, 128, 256]),
            fc_sizes=kwargs.get("fc_sizes", [512, 256]),
            dropout=kwargs.get("dropout", 0.3),
            name=kwargs.get("name", "CNN"),
        )

    elif model_type == "nanni_cnn1":
        required_params = ["output_size"]
        _check_required_params(required_params, kwargs)
        return nanni_cnn1(
            sequence_length=kwargs.get("sequence_length", 313),
            hidden_size=kwargs.get("hidden_size", 8),
            output_size=kwargs["output_size"],
            name=kwargs.get("name", "nanni_cnn1"),
        )

    elif model_type == "nanni_cnn2":
        required_params = ["output_size"]
        _check_required_params(required_params, kwargs)
        return nanni_cnn2(
            sequence_length=kwargs.get("sequence_length", 313),
            hidden_size=kwargs.get("hidden_size", 1024),
            output_size=kwargs["output_size"],
            name=kwargs.get("name", "nanni_cnn2"),
        )

    elif model_type == "nanni_att":
        required_params = ["output_size"]
        _check_required_params(required_params, kwargs)
        return nanni_att(
            sequence_length=kwargs.get("sequence_length", 313),
            output_size=kwargs["output_size"],
            num_heads=kwargs.get("num_heads", 8),
            embed_dim=kwargs.get("embed_dim", 64),
            hidden_size=kwargs.get("hidden_size", 100),
            batch_size=kwargs.get("batch_size", 30),
            name=kwargs.get("name", "nanni_att"),
        )

    elif model_type == "hierarchical":
        input_size = kwargs.get("input_size")
        if input_size is None:
            raise ValueError("Missing required parameter: input_size")
        return HierarchicalModel(
            input_size=input_size,
            shared_hidden_sizes=kwargs.get("shared_hidden_sizes", [512, 256]),
            level_specific_sizes=kwargs.get("level_specific_sizes", {
                'kingdom_name': [128, 64],
                'phylum_name': [128, 64],
                'class_name': [128, 64],
                'order_name': [128, 64]
            }),
            dropout=kwargs.get("dropout", 0.3),
            name=kwargs.get("name", "HierarchicalModel"),
        )

    elif model_type == "cascade_hierarchical":
        input_size = kwargs.get("input_size")
        if input_size is None:
            raise ValueError("Missing required parameter: input_size")
        return CascadeHierarchicalModel(
            input_size=input_size,
            shared_hidden_sizes=kwargs.get("shared_hidden_sizes", [512, 256]),
            level_specific_sizes=kwargs.get("level_specific_sizes", {
                'kingdom_name': [128, 64],
                'phylum_name': [128, 64],
                'class_name': [128, 64],
                'order_name': [128, 64]
            }),
            dropout=kwargs.get("dropout", 0.3),
            use_confidence_weighting=kwargs.get("use_confidence_weighting", True),
            name=kwargs.get("name", "CascadeHierarchicalModel"),
        )

    elif model_type == "gnn_hierarchical":
        input_size = kwargs.get("input_size")
        if input_size is None:
            raise ValueError("Missing required parameter: input_size")
        return GNNHierarchicalModel(
            input_size=input_size,
            hidden_sizes=kwargs.get("hidden_sizes", [256, 128]),
            gnn_layers=kwargs.get("gnn_layers", 2),
            use_attention=kwargs.get("use_attention", True),
            dropout=kwargs.get("dropout", 0.3),
            name=kwargs.get("name", "GNNHierarchicalModel"),
        )

    elif model_type == "bert":
        required_params = ["output_size"]
        _check_required_params(required_params, kwargs)
        return BERTTaxoModel(
            vocab_size=kwargs.get("vocab_size", 5),
            max_length=kwargs.get("max_length", 512),
            hidden_size=kwargs.get("hidden_size", 256),
            num_layers=kwargs.get("num_layers", 6),
            num_heads=kwargs.get("num_heads", 8),
            dropout=kwargs.get("dropout", 0.3),
            output_size=kwargs["output_size"],
            classifier_hidden_size=kwargs.get("classifier_hidden_size", 256),
            name=kwargs.get("name", "BERTTaxoModel"),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _check_required_params(required: List[str], provided: Dict[str, Any]):
    """Check that all required parameters are provided."""
    missing = [p for p in required if p not in provided]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")


def load_model(checkpoint_path: str, map_location: Optional[str] = None) -> BaseModel:
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to load model to

    Returns:
        Loaded model instance
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_name = checkpoint.get("model_name", "")

    if "BasicMLP" in model_name:
        return BasicTaxoModel.load(checkpoint_path, map_location)
    elif "EnhancedMLP" in model_name:
        return EnhancedMLP.load(checkpoint_path, map_location)
    elif "CNN" in model_name:
        return CNNModel.load(checkpoint_path, map_location)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_name}")
