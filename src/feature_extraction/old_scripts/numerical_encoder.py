import numpy as np
from typing import List, Dict, Optional, Union
from .base_encoder import BaseSequenceEncoder

class NumericalEncoder(BaseSequenceEncoder):
    """
    Numerical encoding for DNA/RNA sequences.
    
    This encoder represents each nucleotide as a number:
    - Standard encoding: A=1, C=2, G=3, T=4, N=0
    - Binary encoding: A=00, C=01, G=10, T=11, N=0000
    - Physicochemical encoding: Based on physical and chemical properties
    """
    
    ENCODING_SCHEMES = {
        'standard': {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4, 'N': 0},
        'purine_pyrimidine': {'A': 1, 'G': 1, 'C': 0, 'T': 0, 'U': 0, 'N': 0.5},
        'weak_strong': {'A': 0, 'T': 0, 'U': 0, 'G': 1, 'C': 1, 'N': 0.5},
        'amino_keto': {'A': 0, 'C': 0, 'T': 1, 'U': 1, 'G': 1, 'N': 0.5},
        'gcontent': {'G': 1, 'C': 1, 'A': 0, 'T': 0, 'U': 0, 'N': 0.5}
    }
    
    def __init__(self, max_length: Optional[int] = None, padding: bool = True, 
                 encoding_type: str = 'standard', return_dict: bool = False):
        """
        Initialize the numerical encoder.
        
        Parameters:
        -----------
        max_length : int, optional
            Maximum sequence length to encode. If None, uses full sequence.
        padding : bool
            Whether to pad sequences to the same length
        encoding_type : str
            Type of numerical encoding to use (see ENCODING_SCHEMES)
        return_dict : bool
            If True, transform returns a dictionary of different encoding schemes
        """
        super().__init__(name=f"numerical_{encoding_type}", max_length=max_length)
        self.padding = padding
        self.encoding_type = encoding_type
        self.return_dict = return_dict
        
        # Validate encoding type
        if encoding_type not in self.ENCODING_SCHEMES and encoding_type != 'all':
            raise ValueError(f"Unknown encoding type: {encoding_type}. "
                             f"Use one of: {', '.join(self.ENCODING_SCHEMES.keys())} or 'all'")
        
    def fit(self, sequences: List[str]) -> 'NumericalEncoder':
        """
        Fit the encoder. For numerical encoding this just determines max_length if not already set.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to fit on
            
        Returns:
        --------
        self : NumericalEncoder
            Returns self for method chaining
        """
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        return self
    
    def transform(self, sequences: List[str]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transform sequences to numerical encoding.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
            
        Returns:
        --------
        np.ndarray or Dict[str, np.ndarray]
            Numerically encoded sequences of shape (n_sequences, max_length)
            or dictionary of different encoding schemes if return_dict is True
        """
        sequences = self._validate_sequences(sequences)
        
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        
        if self.encoding_type == 'all' or self.return_dict:
            # Return all encoding schemes
            result = {}
            for scheme_name, scheme in self.ENCODING_SCHEMES.items():
                encoded = self._encode_with_scheme(sequences, scheme)
                result[scheme_name] = encoded
            return result
        else:
            # Return single encoding scheme
            scheme = self.ENCODING_SCHEMES[self.encoding_type]
            return self._encode_with_scheme(sequences, scheme)
    
    def _encode_with_scheme(self, sequences: List[str], scheme: Dict[str, int]) -> np.ndarray:
        """
        Encode sequences using a specific encoding scheme.
        
        Parameters:
        -----------
        sequences : List[str]
            List of sequences to encode
        scheme : Dict[str, int]
            Encoding scheme mapping nucleotides to numerical values
            
        Returns:
        --------
        np.ndarray
            Encoded sequences
        """
        encoded = np.zeros((len(sequences), self.max_length), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq[:self.max_length]):
                encoded[i, j] = scheme.get(base, scheme.get('N', 0))
                
        return encoded
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
        --------
        List[str]
            Feature names
        """
        if self.encoding_type == 'all' or self.return_dict:
            # Return empty list for dictionary output
            return []
        
        return [f"pos_{i+1}" for i in range(self.max_length)]