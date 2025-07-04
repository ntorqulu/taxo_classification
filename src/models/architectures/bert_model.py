import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
from models.architectures.base_model import BaseModel


class BERTTaxoModel(BaseModel):
    """BERT-based model for taxonomy classification that handles multiple encoding types."""
    
    def __init__(self, 
                 vocab_size: int = 5,  # A, T, G, C, N
                 max_length: int = 512,
                 hidden_size: int = 128,  # Reduced for better training
                 num_layers: int = 3,     # Reduced for better training
                 num_heads: int = 4,      # Reduced for better training
                 dropout: float = 0.2,    # Reduced for better training
                 output_size: Optional[int] = None,
                 classifier_hidden_size: int = 128,  # Reduced for better training
                 name: str = "BERTTaxoModel"):
        """
        Initialize the BERT-based taxonomy model.
        
        Args:
            vocab_size: Size of vocabulary (5 for DNA: A, T, G, C, N)
            max_length: Maximum sequence length
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_size: Number of classes for classification
            classifier_hidden_size: Size of hidden layer in classifier
            name: Model name
        """
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
        self.char_to_id = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.id_to_char = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, hidden_size))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        
        # Flag to track which encoding type we're using
        self.encoding_type = None
        
    def _tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to token IDs."""
        tokens = []
        for char in sequence:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.char_to_id['N'])  # Unknown character
        return torch.tensor(tokens, dtype=torch.long)
    
    def _tokenize_batch(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of sequences."""
        batch_tokens = []
        batch_masks = []
        
        for sequence in sequences:
            # Tokenize sequence
            tokens = self._tokenize_sequence(sequence)
            
            # Pad or truncate to max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                mask = torch.ones(self.max_length, dtype=torch.bool)
            else:
                # Pad with N token (ID 4)
                padding_length = self.max_length - len(tokens)
                tokens = torch.cat([tokens, torch.full((padding_length,), 4, dtype=torch.long)])
                mask = torch.cat([torch.ones(len(tokens) - padding_length, dtype=torch.bool),
                                torch.zeros(padding_length, dtype=torch.bool)])
            
            batch_tokens.append(tokens)
            batch_masks.append(mask)
        
        return torch.stack(batch_tokens), torch.stack(batch_masks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BERT model.
        
        Args:
            x: Input tensor - can be:
               - 4-row encoding: [batch_size, 4, sequence_length] or [batch_size, 1, 4, sequence_length]
               - One-hot encoding: [batch_size, 4, sequence_length] or [batch_size, 1, 4, sequence_length]
               - K-mer encoding: [batch_size, features]
        Returns:
            Classification logits
        """
        # Handle extra dimension if present
        if x.dim() == 4 and x.shape[1] == 1:
            # Input shape: [batch_size, 1, 4, sequence_length] -> squeeze to [batch_size, 4, sequence_length]
            x = x.squeeze(1)
        
        # Determine encoding type based on input shape
        if x.dim() == 3 and x.shape[1] == 4:
            # 4-row or one-hot encoding: [batch_size, 4, sequence_length]
            self.encoding_type = "4row_or_onehot"
            return self._forward_4row_or_onehot(x)
        elif x.dim() == 2:
            # K-mer encoding: [batch_size, features]
            self.encoding_type = "kmer"
            return self._forward_kmer(x)
        else:
            raise ValueError(f"Unsupported input format: {x.shape}")
    
    def _forward_4row_or_onehot(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 4-row or one-hot encoding."""
        # Input shape: [batch_size, 4, sequence_length]
        batch_size, num_channels, seq_len = x.shape
        
        # Convert to character sequences
        sequences = self._convert_4row_to_sequences(x)
        
        # Tokenize sequences
        input_ids, attention_mask = self._tokenize_batch(sequences)
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
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
        
        # Apply dropout
        embeddings = self.dropout_layer(embeddings)
        
        # Create padding mask for transformer
        padding_mask = ~attention_mask  # Invert attention mask
        
        # Pass through transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
        # Global average pooling (mean over sequence length)
        masked_output = transformer_output * attention_mask.unsqueeze(-1)
        features = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def _forward_kmer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for k-mer encoding."""
        # Input shape: [batch_size, features]
        # For k-mer encoding, we'll use a simple MLP approach
        
        # Create MLP classifier if it doesn't exist
        if not hasattr(self, 'kmer_classifier'):
            input_size = x.shape[1]
            self.kmer_classifier = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, self.output_size) if self.output_size else nn.Identity()
            ).to(x.device)
        
        return self.kmer_classifier(x)
    
    def _convert_4row_to_sequences(self, matrix: torch.Tensor) -> list[str]:
        """Convert 4-row matrix encoding to DNA sequences."""
        # Matrix shape: [batch_size, 4, sequence_length]
        # Each position has 4 values representing A, T, G, C
        batch_size, _, seq_len = matrix.shape
        sequences = []
        
        for i in range(batch_size):
            sequence = ""
            for j in range(seq_len):
                # Get the nucleotide probabilities for this position
                probs = matrix[i, :, j]  # [4] - probabilities for A, T, G, C
                
                # Find the most likely nucleotide
                nucleotide_idx = int(torch.argmax(probs).item())
                nucleotides = ['A', 'T', 'G', 'C']
                nucleotide = nucleotides[nucleotide_idx]
                sequence += nucleotide
            
            sequences.append(sequence)
        
        return sequences
    
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
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        
        model = cls(
            vocab_size=config['vocab_size'],
            max_length=config['max_length'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            output_size=config['output_size'],
            classifier_hidden_size=config['classifier_hidden_size'],
            name=config.get('name', 'BERTTaxoModel')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 