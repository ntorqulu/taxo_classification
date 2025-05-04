import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Any
from Bio import SeqIO
from Bio.Seq import Seq

def load_sequences(file_path: str, file_format: str = 'fasta') -> List[str]:
    """
    Load sequences from a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the sequence file
    file_format : str
        Format of the sequence file (e.g., 'fasta', 'fastq')
        
    Returns:
    --------
    List[str]
        List of sequences
    """
    sequences = []
    for record in SeqIO.parse(file_path, file_format):
        sequences.append(str(record.seq).upper())
    return sequences

def reverse_complement(sequence: str) -> str:
    """
    Get the reverse complement of a DNA sequence.
    
    Parameters:
    -----------
    sequence : str
        DNA sequence
        
    Returns:
    --------
    str
        Reverse complement sequence
    """
    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement_map.get(base, 'N') for base in reversed(sequence.upper()))

def sliding_window(sequence: str, window_size: int, stride: int = 1) -> List[str]:
    """
    Generate subsequences using a sliding window.
    
    Parameters:
    -----------
    sequence : str
        Input sequence
    window_size : int
        Size of the sliding window
    stride : int
        Step size for sliding the window
        
    Returns:
    --------
    List[str]
        List of subsequences
    """
    if len(sequence) < window_size:
        return [sequence]
        
    subsequences = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        subsequences.append(sequence[i:i+window_size])
    return subsequences

def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize feature vectors.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature vectors to normalize
    method : str
        Normalization method: 'minmax', 'zscore', or 'l2'
        
    Returns:
    --------
    np.ndarray
        Normalized feature vectors
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals[range_vals == 0] = 1
        return (features - min_vals) / range_vals
        
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean_vals = features.mean(axis=0)
        std_vals = features.std(axis=0)
        # Avoid division by zero
        std_vals[std_vals == 0] = 1
        return (features - mean_vals) / std_vals
        
    elif method == 'l2':
        # L2 normalization (unit length)
        norms = np.sqrt((features ** 2).sum(axis=1, keepdims=True))
        # Avoid division by zero
        norms[norms == 0] = 1
        return features / norms
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")