from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any, Optional

class BaseSequenceEncoder(ABC):
    """Abstract base class for all sequence encoders."""
    
    def __init__(self, name: str, max_length: Optional[int] = None):
        """
        Initialize the encoder.
        
        Parameters:
        -----------
        name : str
            Name of the encoder
        max_length : int, optional
            Maximum sequence length to encode. If None, uses full sequence.
        """
        self.name = name
        self.max_length = max_length
        
    @abstractmethod
    def fit(self, sequences: List[str]) -> 'BaseSequenceEncoder':
        """
        Fit the encoder on a list of sequences (if necessary).
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : BaseSequenceEncoder
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Transform a list of sequences into encoded form.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
            
        Returns:
        --------
        np.ndarray
            Encoded sequences
        """
        pass
    
    def fit_transform(self, sequences: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit and encode
            
        Returns:
        --------
        np.ndarray
            Encoded sequences
        """
        return self.fit(sequences).transform(sequences)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
        --------
        List[str]
            Feature names
        """
        return []
    
    def _validate_sequences(self, sequences: List[str]) -> List[str]:
        """
        Validate and preprocess sequences.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to validate
            
        Returns:
        --------
        List[str]
            Validated and preprocessed sequences
        """
        # Convert to uppercase
        processed = [seq.upper() for seq in sequences]
        
        # Truncate if max_length is specified
        if self.max_length is not None:
            processed = [seq[:self.max_length] for seq in processed]
            
        return processed