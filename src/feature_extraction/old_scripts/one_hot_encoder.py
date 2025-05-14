import numpy as np
from typing import List, Dict, Optional
from .base_encoder import BaseSequenceEncoder

class OneHotEncoder(BaseSequenceEncoder):
    """
    One-hot encoding for DNA/RNA sequences.
    
    This encoder represents each nucleotide as a binary vector where one position is 1 
    and all others are 0. For example, A might be [1,0,0,0], C might be [0,1,0,0], etc.
    
    The standard encoding uses:
    - A, C, G, T/U as the main bases
    - N for unknown bases (or other ambiguous characters)
    """
    
    def __init__(self, max_length: Optional[int] = None, padding: bool = True):
        """
        Initialize the one-hot encoder.
        
        Parameters:
        -----------
        max_length : int, optional
            Maximum sequence length to encode. If None, uses the length of the longest sequence.
        padding : bool
            Whether to pad sequences to the same length
        """
        super().__init__(name="one_hot", max_length=max_length)
        self.padding = padding
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.vocab_size = len(self.vocab)
        
    def fit(self, sequences: List[str]) -> 'OneHotEncoder':
        """
        Fit the encoder. For one-hot encoding this just determines max_length if not already set.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : OneHotEncoder
            Returns self for method chaining
        """
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        return self
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequences to one-hot encoding.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
            
        Returns:
        --------
        np.ndarray
            One-hot encoded sequences of shape (n_sequences, max_length, vocab_size)
        """
        sequences = self._validate_sequences(sequences)
        
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        
        # Initialize the output array
        encoded = np.zeros((len(sequences), self.max_length, self.vocab_size), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq[:self.max_length]):
                # Get the index for this base, defaulting to 'N' for unknown bases
                base_idx = self.vocab.get(base, self.vocab['N'])
                encoded[i, j, base_idx] = 1.0
                
        return encoded
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
        --------
        List[str]
            Feature names
        """
        feature_names = []
        bases = {v: k for k, v in self.vocab.items()}
        
        for pos in range(self.max_length):
            for base_idx in range(self.vocab_size):
                feature_names.append(f"pos_{pos+1}_{bases[base_idx]}")
                
        return feature_names