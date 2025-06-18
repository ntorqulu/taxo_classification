from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from networkx import config

from models.architectures.base_model import BaseModel


class nanni_cnn1(BaseModel):
    # - Convolution2d(3, 16, ‘Padding’, ‘same’): The size of the convolutional kernel/filter is
    # 3 × 3. The number of filters is 16. ‘Padding’, ‘same’ means the padding is set so that
    # the spatial dimensions of the input and output feature maps are the same.
    #      -- modification to adapt to the 4X320 input size:
    # - Batch normalization: The output of the previous layer is normalized, thus helping
    # with training stability and convergence.
    # - Dropout: This CNN introduces dropout, a regularization technique to randomly set a
    # fraction of input units to zero during training. Dropout helps prevent overfitting. The
    # dropout rate is 0.5.
    # - Relu: A Rectified Linear Unit (ReLU) activation layer.
    # - Fully connected(8): The number of neurons in this fully connected layer is 8.
    # - Fully connected: The number of neurons in this layer is equal to the number of
    # classes in the classification task. This layer produces the final output scores before
    # applying softmax.
    # - Softmax: The softmax activation function is applied to the output, converting logits
    # into probabilities.
    def __init__(self, sequence_length: int, hidden_size: int, output_size: int, name: str = "nanni_cnn1"):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 4 * sequence_length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = f.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = f.softmax(x, dim=1)
        # softmax is applied to the output layer
        # to convert logits into probabilities
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"sequence_length": self.sequence_length, "hidden_size": self.hidden_size, "output_size": self.output_size}
        )
        return config

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "nanni_cnn1":
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["model_config"]

        model = cls(
            sequence_length=config["sequence_length"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"],
            name=config.get("name", "nanni_cnn1"),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class nanni_cnn2(BaseModel):
    # - Convolution2d(5, 16, ‘Padding’, ‘same’): The size of the convolutional kernel/filter is
    # 5 × 5. The number of filters is 16. ‘Padding’, ‘same’ means the padding is set so that
    # the spatial dimensions of the input and output feature maps are the same.
    # - Relu: Rectified Linear Unit activation layer.
    # - Convolution2d(5, 36, ‘Padding’, ‘same’): CNN2 has another convolutional layer with
    # size 5 × 5. The number of filters is 36.
    # - Relu: Another ReLU activation layer.
    # - Max pooling2d(2): This is a max pooling layer with a 2 × 2 pool size. Max pooling
    # helps reduce spatial dimensions.
    # - Dropout(0.2): CNN2 also has a dropout layer with a dropout rate 0.2.
    # - Relu: Another ReLU activation layer.
    # - Fully connected(1024/reduce). A fully connected layer with 1024/reduce output
    # neurons. The value of reduce is related to the dataset. We set it to ‘1’ and increase the
    # value if and when encountering a GPU memory problem.
    # - Relu: ReLU activation layer.
    # - Fully connectedLayer(1024/reduce). Another fully connected layer/reducer with 1024
    # output neurons.
    # - Relu: Another ReLU activation layer.
    # - Fully connected(1024/reduce). Yet another fully connected layer with 1024/reduce
    # output neurons.
    # - Relu: Another ReLU activation layer.
    # - Fully connected(numClasses): A fully connected layer with the number of neurons
    # equal to the number of classes, as is typical of a CNN output layer.
    # - Softmax: The softmax activation layer normalizes the output into a probability distri-
    # bution over the classes.

    def __init__(self, sequence_length: int, output_size: int, hidden_size: int = 1024, name: str = "nanni_cnn2"):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 36, kernel_size=5, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(36 * 4 / 2 * (sequence_length // 2), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = f.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)
        x = f.softmax(x, dim=1)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"sequence_length": self.sequence_length, "hidden_size": self.hidden_size, "output_size": self.output_size}
        )
        return config

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "nanni_cnn2":
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["model_config"]

        model = cls(
            sequence_length=config["sequence_length"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"],
            name=config.get("name", "nanni_cnn2"),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"sequence_length": self.sequence_length, "hidden_size": self.hidden_size, "output_size": self.output_size}
        )
        return config

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "nanni_cnn2":
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["model_config"]

        model = cls(
            sequence_length=config["sequence_length"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"],
            name=config.get("name", "nanni_cnn2"),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class nanni_att(BaseModel):
    # - flattenConverts the multi-dimensional input into a 1D
    #   vector by flattening the spatial dimensions.
    # - selfAttentionLayer(8,64): A layer that applies self-attention, which allows the network
    #   to focus on different parts of the input. Parameters: Number of attention heads = 8.
    #   Size of the projection = 64. -> MultiheadAttention(emb_dim = mida output flatten layer 4*n, num_heads = 8, dropout = 0,)
    # - bilstmLayer(100):Bidirectional Long Short-Term Memory layer; a recurrent layer that
    #   can process sequences in both forward and backward directions. Each BiLSTM cell
    #   has 100 hidden units.
    # - batchNormalizationLayer: It improves model convergence and stabilizes the training
    #   process by standardizing the inputs to each layer.
    # - fullyConnectedLayer(numClasses): A fully connected layer that maps the output from
    #   the BiLSTM layer to the number of classes in the classification task.
    # - Softmax: The softmax activation layer normalizes the output into a probability distri-
    # bution over the classes.

    def __init__(
        self,
        sequence_length: int,
        output_size=int,
        num_heads: int = 8,
        embed_dim: int = 64,
        hidden_size: int = 100,
        batch_size: int = 30,
        name: str = "nanni_att",
    ):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.flatten = nn.Flatten()
        self.input_projection = nn.Linear(4, self.embed_dim)
        # self.input_projection = nn.Linear(self.sequence_length * 4, self.embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )
        self.bilstm = nn.LSTM(
            input_size=embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(num_features=2 * self.hidden_size)  # 100*2 por bidireccional
        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        x = self.input_projection(x)
        x, attn_weights = self.self_attention(x, x, x)
        x, _ = self.bilstm(x)
        x = x.permute(1, 2, 0)
        x = self.batch_norm(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "output_size": self.output_size,
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
                "hidden_size": self.hidden_size,
                "batch_size": self.batch_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "nanni_att":
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["model_config"]

        model = cls(
            sequence_length=config["sequence_length"],
            output_size=config["output_size"],
            num_heads=config["num_heads"],
            embed_dim=config["embed_dim"],
            hidden_size=config["hidden_size"],
            batch_size=config["batch_size"],
            name=config["name"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


#     2025-06-18 14:49:51 INFO     Starting training for 50 epochs
# 2025-06-18 14:49:52 INFO     Train Epoch: 1 [0/357188] Loss: 2.775197, Acc: 3.33%
