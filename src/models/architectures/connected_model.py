import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.utils import info
from typing import Dict, Any, List, Optional, Tuple
from models.architectures.base_model import BaseModel
from constants.taxonomy_labels import TAXONOMY_LABELS, TAXONOMY_LEVELS
from models.architectures.nanni2024 import nanni_cnn2, nanni_cnn1


class ConnectedModel(BaseModel):
    r"""
    Iterative Fixed Input
        x = net1(x0)
        x1 = net2(concat(x, x0))
        x2 = net3(concat(x1, x0))
        x3 = net4(concat(x2, x0))

    DenseNet
        x1 = net1(x0)
        x2 = net2(concat(x0, x1))
        x3 = net3(concat(x0, x1, x2))

    ResNet
        x1 = net1(x0) + x0
        x2 = net2(x1) + x1
        x3 = net3(x2) + x2

    Recurrent refinement
        x = net(x0)
        x1 = net(concat(x, x0))
        x2 = net(concat(x1, x0))
        x3 = net(concat(x2, x0))


    NO Cascade:
        x1 = net1(x0)
        x2 = net2(x1)
        x3 = net3(x2)

    """

    MODELS: list[str] = ["iterative", "densenet", "resnet", "recurrent"]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 models: list[nn.Module],
                 dropout: float = 0.5,
                 connected_type: str  = 'iterative',
                 connected_models: dict[str, str] = None):
        super().__init__(name=f"ConnectedModel_{connected_type}")

        # Validate parameters

        if connected_type not in self.MODELS:
            raise ValueError("Unknown connected type {connected_type}")
        if not hidden_size:
            raise ValueError("Hidden size must be specified")
        if not models:
            raise ValueError("Models must be specified")
        if len(models) < 2:
            raise ValueError("We need at least two models to connect them")
        for i, model in enumerate(models):
            if not hasattr(model, 'output_size'):
                raise ValueError(f"The model {i} has not output_size attribute.")
        if connected_type == "recurrent" and len(list(set(connected_models.values()))) == 1:
            raise ValueError("Only one recurrent model is allowed")

        info(f"Creating ConnectedModel of type {connected_type} with {len(models)} models. {input_size=}, {output_size=}")

        self.input_size = input_size
        self.output_size = output_size
        self.models = models
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.connected_models = connected_models
        self.connected_type = connected_type

        self.layers = nn.ModuleList()
        self.projections_mlp1 = nn.ModuleList()
        self.projections_relu= nn.ModuleList()
        self.projections_mlp2 = nn.ModuleList()
        self.linear_additions = nn.ModuleList()
        self.linear_additions_batch_norm = nn.ModuleList()
        self.linear_additions_relu = nn.ModuleList()

        # Initialize the layers

        acc_output_size = 0
        for idx, m in enumerate(models):
            self.layers.append(m)
            if idx < len(models) - 1:
                output_size = m.output_size
                acc_output_size += output_size
                connection_output_size = 4 * input_size
                if self.connected_type in ('iterative', 'recurrent'):
                    self.projections_mlp1.append(nn.Linear(4 * input_size + output_size, self.hidden_size))
                    self.projections_relu.append(nn.ReLU())
                    self.projections_mlp2.append(nn.Linear(self.hidden_size, connection_output_size))
                elif self.connected_type == "densenet":
                    self.projections_mlp1.append(nn.Linear((4 * input_size) + acc_output_size, self.hidden_size))
                    self.projections_relu.append(nn.ReLU())
                    self.projections_mlp2.append(nn.Linear(self.hidden_size, connection_output_size))
                elif self.connected_type == "resnet":
                    self.linear_additions.append(nn.Linear(output_size, 4 * input_size))
                    self.linear_additions_batch_norm.append(nn.BatchNorm1d(4 * input_size))
                    self.linear_additions_relu.append(nn.ReLU())
                    self.projections_mlp1.append(nn.Linear(4 * input_size, self.hidden_size))
                    self.projections_relu.append(nn.ReLU())
                    self.projections_mlp2.append(nn.Linear(self.hidden_size, connection_output_size))
                else:
                    raise ValueError(f"Unknown connected type {self.connected_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save the origina input
        x0 = x

        # Initialize concatenation for models that need it
        x0_cat = None

        #
        for idx, model in enumerate(self.models):
            current_input = x
            x = model(x)

            if idx < len(self.models) - 1:
                if self.connected_type in ('iterative', 'recurrent'):
                    # Do the concatenation
                    # x0.size() -> [bs, 1, 4, 313]
                    x0_flat = x0.view(x0.size(0), -1)
                    # x0_flat.size() -> [bs, 4*313] = [bs, 1252]
                    x_combined = torch.cat([x0_flat, x], dim=1)
                    # x_combined.size() -> [bs, 1252+3] [bs, 1255]

                    # Projections
                    x = self.projections_mlp1[idx](x_combined)
                    x = self.projections_relu[idx](x)
                    x = self.projections_mlp2[idx](x)
                elif self.connected_type == "densenet":
                    # Do the concatenation
                    if x0_cat is None:
                        # x0.size() -> [100, 1, 4, 313]
                        x0_flat = x0.view(x0.size(0), -1)
                    else:
                        x0_flat = x0_cat.view(x0_cat.size(0), -1)
                        # x0_flat.size() -> [50, 1252]
                    x0_cat = torch.cat([x0_flat, x], dim=1)
                    # x_combined.size() -> [bs, 1252+3] [bs, 1255]

                    # Projections
                    x = self.projections_mlp1[idx](x0_cat)
                    x = self.projections_relu[idx](x)
                    x = self.projections_mlp2[idx](x)
                elif self.connected_type == "resnet":
                    # Do the additino

                    x_linear_for_add = self.linear_additions[idx](x)
                    x_linear_for_add = self.linear_additions_batch_norm[idx](x_linear_for_add)
                    x_addition = x_linear_for_add + current_input.view(current_input.size()[0], -1)
                    x = self.linear_additions_relu[idx](x_addition)

                    # Projections
                    x = self.projections_mlp1[idx](x)
                    x = self.projections_relu[idx](x)
                    x = self.projections_mlp2[idx](x)
                else:
                    raise ValueError(f"Unknown connected type {self.connected_type}")

                # Change the dimensions to the original one
                x = x.view_as(x0)
        return x


    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'name': self.name,
            'connected_type': self.connected_type,
            'connected_models': self.connected_models
        })
        return config
