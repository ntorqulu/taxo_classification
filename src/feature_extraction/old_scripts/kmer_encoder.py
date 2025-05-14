import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
from itertools import product
from .base_encoder import BaseSequenceEncoder

class KmerEncoder(BaseSequenceEncoder):
    """
    K-mer encoding for DNA/RNA sequences.
    
    This encoder represents sequences as k-mer frequency vectors.
    It supports three modes:
    - 'count': Raw counts of each k-mer
    - 'frequency': Normalized counts (frequency)
    - 'binary': Presence (1) or absence (0) of each k-mer
    """
    
    def __init__(self, k: int = 3, mode: str = 'frequency', stride: int = 1, 
                 normalize: bool = True, max_features: Optional[int] = None):
        """
        Initialize the k-mer encoder.
        
        Parameters:
        -----------
        k : int
            Size of k-mers (subsequence length)
        mode : str
            Encoding mode - 'count', 'frequency', or 'binary'
        stride : int
            Step size when extracting k-mers (1 = all possible k-mers)
        normalize : bool
            Whether to normalize the feature vectors (only for 'count' mode)
        max_features : int, optional
            Maximum number of features to keep (most frequent k-mers)
        """
        super().__init__(name=f"kmer_{k}_{mode}", max_length=None)
        self.k = k
        self.mode = mode
        self.stride = stride
        self.normalize = normalize
        self.max_features = max_features
        self.kmers = None
        self.kmer_to_idx = None
        
    def fit(self, sequences: List[str]) -> 'KmerEncoder':
        """
        Fit the encoder by determining the vocabulary of k-mers.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : KmerEncoder
            Returns self for method chaining
        """
        sequences = self._validate_sequences(sequences)
        
        # Extract all k-mers from all sequences
        all_kmers = []
        for seq in sequences:
            kmers = self._extract_kmers(seq)
            all_kmers.extend(kmers)
        
        # Count k-mer occurrences
        kmer_counts = Counter(all_kmers)
        
        # If max_features is set, keep only the most frequent k-mers
        if self.max_features is not None and len(kmer_counts) > self.max_features:
            self.kmers = [kmer for kmer, _ in kmer_counts.most_common(self.max_features)]
        else:
            self.kmers = sorted(kmer_counts.keys())
        
        # Create mapping from k-mer to index
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        
        return self
    
    def _extract_kmers(self, sequence: str) -> List[str]:
        """
        Extract all k-mers from a sequence.
        
        Parameters:
        -----------
        sequence : str
            Input sequence
            
        Returns:
        --------
        List[str]
            List of k-mers
        """
        if len(sequence) < self.k:
            return []
        
        kmers = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i+self.k]
            if 'N' not in kmer:  # Skip k-mers with unknown bases
                kmers.append(kmer)
                
        return kmers
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequences to k-mer encoding.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
            
        Returns:
        --------
        np.ndarray
            K-mer encoded sequences of shape (n_sequences, n_features)
        """
        if self.kmers is None:
            raise ValueError("Encoder must be fitted before transform")
            
        sequences = self._validate_sequences(sequences)
        n_features = len(self.kmers)
        encoded = np.zeros((len(sequences), n_features), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            kmers = self._extract_kmers(seq)
            
            if self.mode == 'binary':
                # Binary: 1 if k-mer is present, 0 otherwise
                for kmer in set(kmers):
                    if kmer in self.kmer_to_idx:
                        encoded[i, self.kmer_to_idx[kmer]] = 1.0
                        
            else:
                # Count: raw count of each k-mer
                kmer_counts = Counter(kmers)
                for kmer, count in kmer_counts.items():
                    if kmer in self.kmer_to_idx:
                        encoded[i, self.kmer_to_idx[kmer]] = count
                
                # Frequency: normalize by sequence length
                if self.mode == 'frequency' or self.normalize:
                    # Avoid division by zero
                    seq_len = max(1, len(seq) - self.k + 1)
                    encoded[i, :] /= seq_len
                    
        return encoded
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
        --------
        List[str]
            Feature names (k-mers)
        """
        if self.kmers is None:
            return []
        return self.kmers
    
    @classmethod
    def create_all_kmers(cls, k: int, alphabet: str = 'ACGT') -> List[str]:
        """
        Generate all possible k-mers from an alphabet.
        
        Parameters:
        -----------
        k : int
            Length of k-mers
        alphabet : str
            Characters to use for generating k-mers
            
        Returns:
        --------
        List[str]
            All possible k-mers
        """
        return [''.join(p) for p in product(alphabet, repeat=k)]