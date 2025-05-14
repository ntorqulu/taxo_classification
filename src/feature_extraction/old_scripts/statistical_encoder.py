import numpy as np
from typing import List, Dict, Optional, Union
from collections import Counter
from .base_encoder import BaseSequenceEncoder

class StatisticalEncoder(BaseSequenceEncoder):
    """
    Statistical features encoder for DNA/RNA sequences.
    
    This encoder extracts various statistical features from sequences:
    - Nucleotide composition (%)
    - Dinucleotide composition (%)
    - GC content
    - Sequence length
    - Shannon entropy
    - Z-curves (cumulative nucleotide distributions)
    """
    
    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialize the statistical encoder.
        
        Parameters:
        -----------
        features : List[str], optional
            List of features to extract. If None, extracts all features.
            Options: 'length', 'gc_content', 'nucleotide_freq', 'dinucleotide_freq', 
                   'entropy', 'z_curve'
        """
        super().__init__(name="statistical", max_length=None)
        
        self.available_features = [
            'length', 'gc_content', 'nucleotide_freq', 
            'dinucleotide_freq', 'entropy', 'z_curve'
        ]
        
        self.features = features or self.available_features
        
        # Validate features
        for feature in self.features:
            if feature not in self.available_features:
                raise ValueError(f"Unknown feature: {feature}. "
                               f"Use one of: {', '.join(self.available_features)}")
    
    def fit(self, sequences: List[str]) -> 'StatisticalEncoder':
        """
        Fit the encoder. For statistical encoding, this is a no-op.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : StatisticalEncoder
            Returns self for method chaining
        """
        return self
    
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform sequences to statistical features.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
            
        Returns:
        --------
        np.ndarray
            Statistical features of shape (n_sequences, n_features)
        """
        sequences = self._validate_sequences(sequences)
        feature_vectors = []
        
        for seq in sequences:
            # Collect features for this sequence
            seq_features = []
            
            if 'length' in self.features:
                seq_features.append(len(seq))
                
            if 'gc_content' in self.features:
                gc_content = self._calculate_gc_content(seq)
                seq_features.append(gc_content)
                
            if 'nucleotide_freq' in self.features:
                nuc_freqs = self._calculate_nucleotide_frequencies(seq)
                seq_features.extend(nuc_freqs)
                
            if 'dinucleotide_freq' in self.features:
                dinuc_freqs = self._calculate_dinucleotide_frequencies(seq)
                seq_features.extend(dinuc_freqs)
                
            if 'entropy' in self.features:
                entropy = self._calculate_shannon_entropy(seq)
                seq_features.append(entropy)
                
            if 'z_curve' in self.features:
                z_curve = self._calculate_z_curve(seq)
                seq_features.extend(z_curve)
                
            feature_vectors.append(seq_features)
            
        return np.array(feature_vectors, dtype=np.float32)
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if not sequence:
            return 0.0
            
        return (sequence.count('G') + sequence.count('C')) / len(sequence)
    
    def _calculate_nucleotide_frequencies(self, sequence: str) -> List[float]:
        """Calculate frequencies of each nucleotide."""
        if not sequence:
            return [0.0, 0.0, 0.0, 0.0]
            
        length = len(sequence)
        return [
            sequence.count('A') / length,
            sequence.count('C') / length,
            sequence.count('G') / length,
            sequence.count('T') / length
        ]
    
    def _calculate_dinucleotide_frequencies(self, sequence: str) -> List[float]:
        """Calculate frequencies of each dinucleotide."""
        if len(sequence) < 2:
            return [0.0] * 16
            
        # Generate all possible dinucleotides
        nucleotides = ['A', 'C', 'G', 'T']
        dinucleotides = [a + b for a in nucleotides for b in nucleotides]
        
        # Count dinucleotides
        dinuc_counts = Counter()
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            if 'N' not in dinuc:
                dinuc_counts[dinuc] += 1
                
        # Calculate frequencies
        total = sum(dinuc_counts.values()) or 1  # Avoid division by zero
        return [dinuc_counts.get(dinuc, 0) / total for dinuc in dinucleotides]
    
    def _calculate_shannon_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of a sequence."""
        if not sequence:
            return 0.0
            
        # Count nucleotides
        counts = Counter(sequence)
        length = len(sequence)
        
        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            p = count / length
            entropy -= p * np.log2(p)
            
        return entropy
    
    def _calculate_z_curve(self, sequence: str) -> List[float]:
        """Calculate Z-curve representation of a sequence."""
        if not sequence:
            return [0.0, 0.0, 0.0]
            
        # Count cumulative occurrences
        a_count = g_count = c_count = t_count = 0
        
        for base in sequence:
            if base == 'A':
                a_count += 1
            elif base == 'G':
                g_count += 1
            elif base == 'C':
                c_count += 1
            elif base == 'T' or base == 'U':
                t_count += 1
                
        # Calculate Z-curve coordinates
        x = (a_count + g_count) - (c_count + t_count)  # Purine vs Pyrimidine
        y = (a_count + c_count) - (g_count + t_count)  # Amino vs Keto
        z = (a_count + t_count) - (g_count + c_count)  # Weak vs Strong H-bonds
        
        # Normalize by sequence length
        length = len(sequence)
        return [x / length, y / length, z / length]
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
        --------
        List[str]
            Feature names
        """
        feature_names = []
        
        if 'length' in self.features:
            feature_names.append('length')
            
        if 'gc_content' in self.features:
            feature_names.append('gc_content')
            
        if 'nucleotide_freq' in self.features:
            feature_names.extend(['freq_A', 'freq_C', 'freq_G', 'freq_T'])
            
        if 'dinucleotide_freq' in self.features:
            nucleotides = ['A', 'C', 'G', 'T']
            dinucleotides = [a + b for a in nucleotides for b in nucleotides]
            feature_names.extend([f'freq_{dinuc}' for dinuc in dinucleotides])
            
        if 'entropy' in self.features:
            feature_names.append('shannon_entropy')
            
        if 'z_curve' in self.features:
            feature_names.extend(['z_curve_x', 'z_curve_y', 'z_curve_z'])
            
        return feature_names