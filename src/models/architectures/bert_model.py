import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional
from models.architectures.base_model import BaseModel

# ONLY IMPLEMENT FOR 4-ROW ENCODING

class BERTTaxoModel(BaseModel):
    """BERT-based model for taxonomy classification using 4-row encoding."""
    
    def __init__(self, 
                 vocab_size: int = 4,  # A, T, G, C
                 max_length: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 output_size: Optional[int] = None,
                 classifier_hidden_size: int = 128,
                 name: str = "BERTTaxoModel"):
        """Initialize the BERT-based taxonomy model."""
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_size = output_size
        self.classifier_hidden_size = classifier_hidden_size
        
        # Character to ID mapping for DNA
        self.char_to_id = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.id_to_char = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Fixed positional encodings
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_length, hidden_size)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)
        
        # Layer normalization
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, classifier_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size // 2, output_size) if output_size else nn.Identity()
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the BERT model."""
        if x is None or x.nelement() == 0:
            raise ValueError("Empty input tensor provided")
            
        # Handle extra dimension if present
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # Validate input format
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError(f"Expected 4-row encoding with shape [batch_size, 4, seq_len], got: {x.shape}")
        
        return self._forward_4row(x)
    
    def _forward_4row(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 4-row encoding."""
        batch_size, num_channels, seq_len = x.shape
        
        # Convert 4-row matrix directly to token IDs using argmax
        input_ids = torch.argmax(x, dim=1)  # [batch_size, seq_len]
        
        # Create attention mask (assuming all positions are valid)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
        
        # Truncate if sequence is too long
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Embedding layer
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        embeddings = embeddings + pos_enc
        
        # Apply layer normalization
        embeddings = self.embed_norm(embeddings)
        
        # Apply dropout
        embeddings = self.dropout_layer(embeddings)
        
        # Create padding mask for transformer
        padding_mask = ~attention_mask  # Invert attention mask
        
        try:
            # Pass through transformer
            transformer_output = self.transformer(embeddings, src_key_padding_mask=padding_mask)
            
            # Apply final layer normalization
            transformer_output = self.final_norm(transformer_output)
            
            # Global average pooling (mean over sequence length)
            masked_output = transformer_output * attention_mask.unsqueeze(-1)
            features = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            # Classification
            logits = self.classifier(features)
            return logits
            
        except Exception as e:
            raise RuntimeError(f"Error in transformer processing: {str(e)}. Input shape: {x.shape}") from e
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'output_size': self.output_size,
            'classifier_hidden_size': self.classifier_hidden_size,
        })
        return config
    
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'BERTTaxoModel':
        try:
            checkpoint = torch.load(path, map_location=map_location)
            config = checkpoint['model_config']
            
            required_keys = ['vocab_size', 'max_length', 'hidden_size', 
                            'num_layers', 'num_heads', 'dropout']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration key: {key}")
            
            model = cls(
                vocab_size=config['vocab_size'],
                max_length=config['max_length'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout'],
                output_size=config.get('output_size'),
                classifier_hidden_size=config.get('classifier_hidden_size', 128),
                name=config.get('name', 'BERTTaxoModel')
            )
            
            # Load the state dict, ignoring kmer_classifier related keys
            state_dict = checkpoint['model_state_dict']
            # Filter out kmer_classifier keys if present
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith('kmer_classifier')}
            
            model.load_state_dict(filtered_state_dict, strict=False)
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {str(e)}") from e