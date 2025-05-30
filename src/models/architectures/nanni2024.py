import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Dict, Any, List, Optional, Tuple
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
    def __init__(self,
                 sequence_length: int,
                 hidden_size: int,
                 output_size: int,
                 name: str = "nanni_cnn1"):
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16  * 4 * sequence_length, self.hidden_size)
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
        config.update({
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'nanni_cnn1':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            name=config.get('name', 'nanni_cnn1')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


