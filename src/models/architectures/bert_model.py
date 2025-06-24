import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List, Optional, Tuple
from models.architectures.base_model import BaseModel

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for better sequence position understanding."""
    
    def __init__(self, hidden_size: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create sinusoidal position encoding
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with masking support."""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Single projection for query, key, value (efficient implementation)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        # Single projection for Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        
        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        return output

class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization (more stable training)."""
    
    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Pre-layer normalization (BarcodeBERT approach for stable training)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        
        # Position-wise feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),  # BERT uses GELU activation
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm attention
        attn_input = self.attn_norm(x)
        attn_output = self.attention(attn_input, mask)
        x = x + self.dropout1(attn_output)
        
        # Pre-norm feed-forward
        ff_input = self.ff_norm(x)
        ff_output = self.ff(ff_input)
        x = x + self.dropout2(ff_output)
        
        return x

class NucleotideEmbedding(nn.Module):
    """DNA-specific embedding layer with nucleotide and position information."""
    
    def __init__(self, hidden_size, max_length, dropout=0.1, use_sinusoidal=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_sinusoidal = use_sinusoidal
        
        # Nucleotide embedding (from 4-channel to hidden_size)
        self.nucleotide_embedding = nn.Linear(4, hidden_size)
        
        # Position embedding
        if use_sinusoidal:
            self.position_embedding = PositionalEncoding(hidden_size, max_length, dropout)
        else:
            self.position_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
            nn.init.normal_(self.position_embedding, mean=0, std=0.02)
            self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """
        Process input sequence to embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, 4, seq_len) or (batch_size, seq_len, 4)
        """
        # Make sure input is of shape (batch_size, seq_len, 4)
        if x.size(1) == 4 and x.size(2) != 4:
            x = x.permute(0, 2, 1)  # (batch_size, seq_len, 4)
        
        # Get nucleotide embeddings
        embeddings = self.nucleotide_embedding(x)
        
        # Add position embeddings
        if self.use_sinusoidal:
            embeddings = self.position_embedding(embeddings)
        else:
            # Ensure we don't exceed maximum length
            seq_len = min(embeddings.size(1), self.max_length)
            embeddings = embeddings[:, :seq_len] + self.position_embedding[:, :seq_len]
            embeddings = self.dropout(embeddings)
        
        # Layer normalization
        embeddings = self.layer_norm(embeddings)
        
        return embeddings

class SequencePooler(nn.Module):
    """Sequence pooling with multiple options (inspired by BarcodeBERT)."""
    
    def __init__(self, hidden_size, pooling_type='mean'):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling_type = pooling_type
        
        if pooling_type == 'cls':
            # [CLS] token approach for sequence classification
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.Tanh()
        elif pooling_type == 'attention':
            # Attention-based pooling
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(self, sequence_output):
        """
        Pool sequence outputs into a single vector.
        
        Args:
            sequence_output: Transformer outputs of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Pooled representation of shape (batch_size, hidden_size)
        """
        if self.pooling_type == 'cls':
            # Use first token ([CLS] equivalent)
            first_token = sequence_output[:, 0]
            pooled_output = self.activation(self.dense(first_token))
            
        elif self.pooling_type == 'mean':
            # Mean pooling (effective for DNA sequences)
            pooled_output = torch.mean(sequence_output, dim=1)
            
        elif self.pooling_type == 'max':
            # Max pooling (captures salient features)
            pooled_output, _ = torch.max(sequence_output, dim=1)
            
        elif self.pooling_type == 'attention':
            # Attention pooling (weighted average)
            weights = self.attention(sequence_output)
            pooled_output = torch.sum(weights * sequence_output, dim=1)
            
        return pooled_output

class BertDNAModel(BaseModel):
    """BERT-like model for DNA barcode sequence classification inspired by BarcodeBERT."""
    
    def __init__(self, 
                sequence_length: int,
                output_size: int,
                hidden_size: int = 256,
                num_layers: int = 4,
                num_heads: int = 8,
                ff_dim: int = 1024,
                dropout: int = 0.1,
                pooling_type: str = 'mean',
                use_sinusoidal_pos: bool = True,
                name: str = "BertDNAModel"):
        """
        Initialize BertDNAModel model.
        
        Args:
            sequence_length: Maximum sequence length
            output_size: Number of output classes
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Size of feed-forward hidden layer
            dropout: Dropout probability
            pooling_type: Type of pooling to use ('mean', 'max', 'cls', 'attention')
            use_sinusoidal_pos: Whether to use sinusoidal positional encoding
            name: Model name
        """
        super().__init__(name=name)
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.use_sinusoidal_pos = use_sinusoidal_pos
        
        # Embedding layer
        self.embedding = NucleotideEmbedding(
            hidden_size=hidden_size,
            max_length=sequence_length,
            dropout=dropout,
            use_sinusoidal=use_sinusoidal_pos
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Sequence pooler
        self.pooler = SequencePooler(hidden_size, pooling_type)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through BarcodeTransformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 4, seq_len)
               or other formats that will be properly handled
        
        Returns:
            Classification logits of shape (batch_size, output_size)
        """
        # Handle different input shapes
        if x.dim() == 4 and x.size(1) == 1:
            # Input is (batch_size, 1, 4, seq_len)
            x = x.squeeze(1)  # (batch_size, 4, seq_len)
        
        # Generate embeddings
        embeddings = self.embedding(x)
        
        # Process through transformer blocks
        hidden_states = embeddings
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states)
        
        # Pool sequence representation
        pooled_output = self.pooler(hidden_states)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'pooling_type': self.pooling_type,
            'use_sinusoidal_pos': self.use_sinusoidal_pos
        })
        return config
    
    @classmethod
    def load(cls, path, map_location=None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            sequence_length=config['sequence_length'],
            output_size=config['output_size'],
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            ff_dim=config.get('ff_dim', 1024),
            dropout=config.get('dropout', 0.1),
            pooling_type=config.get('pooling_type', 'mean'),
            use_sinusoidal_pos=config.get('use_sinusoidal_pos', True),
            name=config.get('name', 'BarcodeTransformer')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model