from pathlib import Path
import torch
import pandas as pd
from typing import Final, Dict, Any, Optional
from torch.utils.data import Dataset

from src.dataset.cached_dataframe import CachedDataFrame
from src.dataset.parquet_builder import ParquetBuilder
from src.dataset.utils import info
from src.feature_extraction.main import SequenceCoder
from src.constants.taxonomy_labels import TAXONOMY_LABELS, TAXONOMY_LEVELS


class HierarchicalDataset(Dataset):
    """
    Dataset for hierarchical taxonomy classification.
    
    This dataset provides all taxonomic levels as targets simultaneously,
    enabling training of a hierarchical model that predicts all levels at once.
    """
    
    SEQUENCE_CHAR_DIFFERENT_VALUES = 4
    SEQUENCE_LENGTH = 300

    def __init__(self,
                 parquets_path: Path,
                 filters: Dict[str, str | list[str]] = None,
                 k: int = None,
                 bits: int = None,
                 seq_len_filter: int | None = None,
                 include_sequence: bool = False):
        """
        Initialize hierarchical dataset.
        
        Args:
            parquets_path: Path of the parquets to be loaded
            filters: Dictionary of column_names with the values to filter by
            k: k-mer size. If k is specified, bits must be None
            bits: bits for bit encoding. If bits is specified, k must be None
            seq_len_filter: Filter sequences with the exact length
            include_sequence: Whether to include raw sequence in output
        """
        super().__init__()

        if not filters:
            filters = {}
        elif any(r not in TAXONOMY_LABELS for r in filters.keys()):
            raise ValueError(f"Unrecognized filter keys: {filters.keys()}")
        if k is None and bits is None:
            raise ValueError(f"Must specify k or bits")
        if k is not None and k not in ParquetBuilder.KMERS_SIZES:
            raise ValueError(f"Values allowed for k: {ParquetBuilder.KMERS_SIZES}")
        if bits is not None and bits not in SequenceCoder().bit_mapping and bits != 0:
            raise ValueError(f"Values allowed for bits: 0, {SequenceCoder().bit_mapping.keys()}")
        if (k is not None) == (bits is not None):
            raise ValueError(f"You only can specify k and bits, not both: {k=} {bits=}")
        if any(not isinstance(f, str) for f in filters.values()):
            raise NotImplementedError(f"Only strings are allowed as filter values")
        if seq_len_filter is not None and seq_len_filter <= 0:
            raise ValueError(f"seq_len_filter must be positive: {seq_len_filter}")

        self.parquet_path: Path = parquets_path
        self.filters: Dict[str, str] = filters
        self.k: Optional[int] = k
        self.bits: Optional[int] = bits
        self.seq_len_filter: int = seq_len_filter
        self.include_sequence: bool = include_sequence

        # DataFrame with the data, but without the encodings
        self._df: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(self.parquet_path)

        # DataFrame with the encodings
        self._df_encoding: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(
            parquets_path=self.parquet_path,
            k=self.k,
            bits=self.bits
        )

        # Indexes of the final rows after applying filters
        self._filter_indexes: Optional[list[int]] = self._init_filter_indexes()

        # Dictionary with the mapping between label strings and their assigned IDs for each level
        self._labels_ids: Dict[str, Dict[str, int]] = self._init_labels_ids()

    def _init_filter_indexes(self) -> Optional[list[int]]:
        """
        Initialize the list of indexes for the instances that match the filters.
        
        Returns:
            The list of indexes, or None if no filters applied
        """
        if not self.filters and self.seq_len_filter is None:
            return None

        mask = pd.Series(True, index=self._df.index)

        # Update the mask applying the filters
        for column_name, value in self.filters.items():
            if isinstance(value, str):
                mask &= self._df[column_name] == value
            elif isinstance(value, list):
                mask &= self._df[column_name].isin(value)
            else:
                raise NotImplementedError(f"Value types not implemented: {value}")

        # Update the mask with the maximum sequence length filter
        if self.seq_len_filter is not None:
            mask &= self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME].str.len() == self.seq_len_filter

        # Return a list with the indexes after applying the filters
        indexes = self._df[mask].index.tolist()
        return indexes

    def _init_labels_ids(self) -> Dict[str, Dict[str, int]]:
        """
        Assign IDs to each label value for each taxonomic level.
        
        Returns:
            Dictionary mapping taxonomic levels to their label-to-id mappings
        """
        labels_ids = {}
        
        for level in TAXONOMY_LEVELS:
            if level not in TAXONOMY_LABELS:
                continue
                
            # Get the label values depending on whether filters are applied or not
            if self._filter_indexes is None:
                label_values = self._df[level]
            else:
                label_values = self._df.loc[self._filter_indexes, level]

            # Obtain the unique values
            unique_values: list[str] = label_values.unique().tolist()

            # Sort the values to ensure that the IDs remain consistent
            unique_values = sorted(unique_values)

            # Assign the IDs
            level_labels_ids = {l[1]: l[0] for l in enumerate(unique_values)}
            labels_ids[level] = level_labels_ids
            
            info(f"Level {level}: {len(level_labels_ids)} labels available")

        return labels_ids

    @property
    def num_labels_per_level(self) -> Dict[str, int]:
        """
        Get the number of unique labels for each taxonomic level.
        
        Returns:
            Dictionary mapping taxonomic levels to their label counts
        """
        return {level: len(ids) for level, ids in self._labels_ids.items()}

    @property
    def labels_names_per_level(self) -> Dict[str, list[str]]:
        """
        Get the names of labels for each taxonomic level.
        
        Returns:
            Dictionary mapping taxonomic levels to their label names
        """
        return {level: list(ids.keys()) for level, ids in self._labels_ids.items()}

    @property
    def labels_ids_per_level(self) -> Dict[str, list[int]]:
        """
        Get the list of assigned label identifiers for each taxonomic level.
        
        Returns:
            Dictionary mapping taxonomic levels to their label IDs
        """
        return {level: list(ids.values()) for level, ids in self._labels_ids.items()}

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of rows in the dataset
        """
        if self._filter_indexes:
            return len(self._filter_indexes)
        return len(self._df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'features': Input features tensor
                - 'targets': Dictionary mapping taxonomic levels to their target tensors
                - 'sequence': Raw sequence (if include_sequence=True)
        """
        # Get the actual row index
        if self._filter_indexes is not None:
            actual_idx = self._filter_indexes[idx]
        else:
            actual_idx = idx

        # Get features
        features = self._get_features(actual_idx)
        
        # Get targets for all levels
        targets = {}
        for level in TAXONOMY_LEVELS:
            if level in self._labels_ids:
                target_value = self._get_column_value(actual_idx, level)
                target_id = self._labels_ids[level][target_value]
                targets[level] = torch.tensor(target_id, dtype=torch.long)
        
        result = {
            'features': features,
            'targets': targets
        }
        
        # Include sequence if requested
        if self.include_sequence:
            sequence = self._get_column_value(actual_idx, CachedDataFrame.SEQUENCE_COLUMN_NAME)
            result['sequence'] = sequence
            
        return result

    def _get_features(self, idx: int) -> torch.Tensor:
        """
        Get features for a given index.
        
        Args:
            idx: Row index
            
        Returns:
            Features tensor
        """
        # Get the encoding column name
        if self.k is not None:
            encoding_column = f"kmer_{self.k}"
        elif self.bits is not None:
            if self.bits == 0:
                encoding_column = "4rowmatrix"
            else:
                encoding_column = f"bits_{self.bits}"
        else:
            raise ValueError("Neither k nor bits specified")
        
        # Get the encoding
        encoding = self._df_encoding.loc[idx, encoding_column]
        
        # Convert to tensor
        if isinstance(encoding, str):
            # Handle string representation of list/array
            import ast
            encoding = ast.literal_eval(encoding)
        
        features = torch.tensor(encoding, dtype=torch.float32)
        return features

    def _get_column_value(self, idx: int, column_name: str) -> Any:
        """
        Get a column value for a given index.
        
        Args:
            idx: Row index
            column_name: Name of the column
            
        Returns:
            Column value
        """
        return self._df.loc[idx, column_name]

    def get_label(self, idx: int, level: str) -> str:
        """
        Get the label string for a given index and taxonomic level.
        
        Args:
            idx: Sample index
            level: Taxonomic level
            
        Returns:
            Label string
        """
        actual_idx = self._filter_indexes[idx] if self._filter_indexes is not None else idx
        return self._get_column_value(actual_idx, level)

    def get_label_id(self, idx: int, level: str) -> int:
        """
        Get the label ID for a given index and taxonomic level.
        
        Args:
            idx: Sample index
            level: Taxonomic level
            
        Returns:
            Label ID
        """
        label_value = self.get_label(idx, level)
        return self._labels_ids[level][label_value]

    def get_sequence(self, idx: int) -> str:
        """
        Get the sequence for a given index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sequence string
        """
        actual_idx = self._filter_indexes[idx] if self._filter_indexes is not None else idx
        return self._get_column_value(actual_idx, CachedDataFrame.SEQUENCE_COLUMN_NAME)

    @property
    def min_sequence_len(self) -> int:
        """
        Get the minimum sequence length in the dataset.
        
        Returns:
            Minimum sequence length
        """
        if self._filter_indexes is not None:
            sequences = self._df.loc[self._filter_indexes, CachedDataFrame.SEQUENCE_COLUMN_NAME]
        else:
            sequences = self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME]
        return sequences.str.len().min()

    @property
    def max_sequence_len(self) -> int:
        """
        Get the maximum sequence length in the dataset.
        
        Returns:
            Maximum sequence length
        """
        if self._filter_indexes is not None:
            sequences = self._df.loc[self._filter_indexes, CachedDataFrame.SEQUENCE_COLUMN_NAME]
        else:
            sequences = self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME]
        return sequences.str.len().max()

    @property
    def data_length(self) -> int:
        """
        Get the total number of rows in the original dataset.
        
        Returns:
            Total number of rows
        """
        return len(self._df) 