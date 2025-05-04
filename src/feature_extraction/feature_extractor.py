import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Any
from .base_encoder import BaseSequenceEncoder
from .one_hot_encoder import OneHotEncoder
from .kmer_encoder import KmerEncoder
from .numerical_encoder import NumericalEncoder
from .statistical_encoder import StatisticalEncoder
from .utils import normalize_features

class FeatureExtractor:
    """
    Unified interface for extracting features from DNA/RNA sequences.
    Combines multiple encoding methods and provides a single interface.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.encoders = {}
        
    def add_encoder(self, encoder: BaseSequenceEncoder, name: Optional[str] = None) -> 'FeatureExtractor':
        """
        Add an encoder to the feature extractor.
        
        Parameters:
        -----------
        encoder : BaseSequenceEncoder
            Encoder to add
        name : str, optional
            Name to assign to the encoder. If None, uses encoder.name.
            
        Returns:
        --------
        self : FeatureExtractor
            Returns self for method chaining
        """
        encoder_name = name or encoder.name
        self.encoders[encoder_name] = encoder
        return self
    
    def fit(self, sequences: List[str]) -> 'FeatureExtractor':
        """
        Fit all encoders on the sequences.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : FeatureExtractor
            Returns self for method chaining
        """
        for encoder in self.encoders.values():
            encoder.fit(sequences)
        return self
    
    def transform(self, sequences: List[str], encoders: Optional[List[str]] = None, 
                 concatenate: bool = True, normalize: bool = False) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Transform sequences using specified encoders.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
        encoders : List[str], optional
            Names of encoders to use. If None, uses all available encoders.
        concatenate : bool
            If True, concatenates all encoded features into a single array.
            If False, returns a dictionary of encoded features for each encoder.
        normalize : bool
            Whether to normalize the feature vectors
            
        Returns:
        --------
        np.ndarray or Dict[str, np.ndarray]
            Encoded features
        """
        if encoders is None:
            encoders_to_use = list(self.encoders.keys())
        else:
            # Validate encoder names
            for encoder_name in encoders:
                if encoder_name not in self.encoders:
                    raise ValueError(f"Unknown encoder: {encoder_name}")
            encoders_to_use = encoders
        
        # Transform sequences with each encoder
        encoded = {}
        for encoder_name in encoders_to_use:
            encoder = self.encoders[encoder_name]
            features = encoder.transform(sequences)
            
            # Handle one-hot encoding's 3D output when concatenating
            if concatenate and isinstance(features, np.ndarray) and features.ndim == 3:
                # Reshape to 2D: (n_samples, n_positions * n_features)
                n_samples, n_positions, n_features = features.shape
                features = features.reshape(n_samples, n_positions * n_features)
                
            encoded[encoder_name] = features
        
        if not concatenate:
            return encoded
        
        # Concatenate all encodings
        features_list = []
        for encoder_name in encoders_to_use:
            features = encoded[encoder_name]
            
            # Skip non-concatenatable features (e.g., dictionaries)
            if not isinstance(features, np.ndarray):
                continue
                
            # Ensure 2D shape
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            features_list.append(features)
        
        concatenated = np.hstack(features_list)
        
        if normalize:
            concatenated = normalize_features(concatenated, method='minmax')
            
        return concatenated
    
    def get_feature_names(self, encoders: Optional[List[str]] = None) -> List[str]:
        """
        Get feature names for specified encoders.
        
        Parameters:
        -----------
        encoders : List[str], optional
            Names of encoders to use. If None, uses all available encoders.
            
        Returns:
        --------
        List[str]
            Feature names
        """
        if encoders is None:
            encoders_to_use = list(self.encoders.keys())
        else:
            encoders_to_use = encoders
        
        feature_names = []
        for encoder_name in encoders_to_use:
            encoder = self.encoders[encoder_name]
            names = encoder.get_feature_names()
            
            # Prefix feature names with encoder name to avoid conflicts
            prefixed_names = [f"{encoder_name}_{name}" for name in names]
            feature_names.extend(prefixed_names)
            
        return feature_names