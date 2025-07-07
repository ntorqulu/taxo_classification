from pathlib import Path

import torch
import pandas as pd
from typing import Final
from torch.utils.data import Dataset

from dataset.cached_dataframe import CachedDataFrame
from dataset.parquet_builder import ParquetBuilder
from feature_extraction.main import SequenceCoder
from dataset.utils import info

class TaxoDataset(Dataset):
    # Discared columns seqID,taxID,scientific_name

    SEQUENCE_CHAR_DIFFERENT_VALUES = 4 # No commit
    SEQUENCE_LENGTH = 300 # No commit
    LABEL_ID_COLUMN_NAME = 'label_id'

    def __init__(self,
                 parquets_path: Path,
                 label_column_name: str,
                 k: int = None,
                 bits: int = None,
                 value_filters: dict[str, str | list[str]] = None,
                 min_cardinality_filters: dict[str, int] = None,
                 seq_len_filter: int | None = None,
                 ):
        """
        Parameters
        ----------
        parquets_path: Path
            Path of the parquets to be loaded

        label_column_name: str
            Name of the column that contains the labels.

        filters: dict[str, str | list[str]]
            Dictionary of column_names with the values of the column to filter by. The filter
            values can be strings or lists of strings.

        k: int
            k-mer size. If k is specified, bits must be None.

        bits: int
            bits for bit encoding. If bits is specified, k must be None.

        seq_len_filter: int | None
            Filter sequences with the exact length.
        """
        super().__init__()

        # Set default values

        if not value_filters:
            value_filters = {}

        if not min_cardinality_filters:
            min_cardinality_filters = {}

        # Validate k parameter

        if k is None and bits is None:
            raise ValueError(f"Must specify k or bits")

        if k is not None and k not in ParquetBuilder.KMERS_SIZES:
            raise ValueError(f"Values allowed for k: {ParquetBuilder.KMERS_SIZES}")

        # Validate bits parameter

        if bits is not None and bits not in SequenceCoder().bit_mapping and bits != 0:
            raise ValueError(f"Values allowed for bits: 0, {SequenceCoder().bit_mapping.keys()}")

        if (k is not None) == (bits is not None):
            raise ValueError(f"You only can specify k and bits, not both: {k=} {bits=}")

        # Validate value_filters parameter

        if any(not isinstance(f, str) and not isinstance(f, list) for f in value_filters.values()):
            raise ValueError(f"Only strings or list of strings are allowed as filter values")

        for v in value_filters.values():
            if isinstance(v, list) and any(not isinstance(vv, str) for vv in v):
                raise ValueError(f"Only strings are allowed as filter values in lists")

        # Validate min_cardinality_filters parameter

        if any(not isinstance(f, int) for f in min_cardinality_filters.values()):
            raise ValueError(f"Only int are allowed as filter values")

        if any(f <= 0 for f in min_cardinality_filters.values()):
            raise ValueError(f"Only positive values are allowed as filter values")

        # Validate seq_len_filter

        if seq_len_filter is not None and seq_len_filter <= 0:
            raise ValueError(f"seq_len_filter must be positive: {seq_len_filter}")


        self.parquet_path: Path = parquets_path
        self.k: int | None = k
        self.bits: int | None = bits
        self.seq_len_filter: int = seq_len_filter

        # DataFrame with the data, but without the encodings.
        self._df: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(self.parquet_path)

        # DataFrame with the encodings
        self._df_encoding: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(parquets_path=self.parquet_path,
                                                                                k=self.k,
                                                                                bits=self.bits)

        level_column_names: Final[list[str]] = CachedDataFrame.get_level_column_names()
        if any(r not in level_column_names for r in value_filters.keys()):
            raise ValueError(f"Unrecognized filter keys in filters: {value_filters.keys()}")
        if any(r not in level_column_names for r in min_cardinality_filters.keys()):
            raise ValueError(f"Unrecognized filter keys in level_column_names: {min_cardinality_filters.keys()}")
        if label_column_name not in level_column_names:
            raise ValueError(f"Unrecognized label column name: {label_column_name}")

        self.value_filters: dict[str, str] = value_filters
        self.min_cardinality_filters: dict[str, int] = min_cardinality_filters
        self.label_column_name: str = label_column_name

        # Indexes of the final rows after applying both the filter and max_len_filter, if specified.
        # It is None if no index is applied, meaning all rows are included.
        self._filter_indexes: list[int] | None = self._init_filter_indexes()

        # Dictionary with the mapping between label strings and their assigned IDs.
        self._label_ids_by_name: dict[str, int] = self._init_label_ids()
        info(f"There are {len(self.label_ids)} label values in '{self.label_column_name}': "
             f"{', '.join(self.label_values)}")

        self._label_cardinality_by_name: dict[str, int] = CachedDataFrame.get_column_cardinality(self.label_column_name)
        #info("Distinct values for the labels:")
        #[info(f"  {value}: {cardinality}") for value, cardinality in self._label_cardinality_by_name.items()]

    def _init_filter_indexes(self) -> list[int] | None:
        """
        Initializes the list of indexes for the instances that match the filters.

        Returns
        -------
        The list of indexes, or None if no filters applyied
        """

        if not self.value_filters and not self.min_cardinality_filters and not self.seq_len_filter:
            return None

        mask = pd.Series(True, index=self._df.index)

        # Update the mask applying the filters
        for columm_name, value in self.value_filters.items():
            if isinstance(value, str):
                mask &= self._df[columm_name] == value
            elif isinstance(value, list):
                mask &= self._df[columm_name].isin(value)
            else:
                raise NotImplementedError(f"Value types not implemented: {value}")

        # Update the mask with the maxiumn sequence length fitler
        if self.seq_len_filter is not None:
            mask &= self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME].str.len() == self.seq_len_filter

        # Update the mask applying the filter_min_cardinalities
        # mask_fixed = mask.copy()  # Use this if you don't want to update the mask in place
        for column_name, min_cardinality in self.min_cardinality_filters.items():
            cardinalities = self._df[mask][column_name].value_counts()
            mask &= self._df[column_name].map(cardinalities).astype(int) >= min_cardinality

        # Return a list with the indexes after applying the filters
        indexes = self._df[mask].index.tolist()
        return indexes

    def _init_label_ids(self) -> dict[str, int]:
        """
        Assigns an id to each label value.

        -------
        Returns a dictionary with the mapping between label strings and their assigned IDs.
        """

        # Gets the label values depending on whether filters are applied or not.
        if self._filter_indexes is None:
            label_values = self._df[self.label_column_name]
        else:
            label_values = self._df.loc[self._filter_indexes, self.label_column_name]

        # Obtain the unique valus
        unique_values: list[str] = label_values.unique().tolist()

        # We sort the values to ensure that the IDs remain consistent when the labels are the same.
        unique_values = sorted(unique_values)

        # Assign the ids
        label_ids = {l[1]: l[0] for l in enumerate(unique_values)}

        return label_ids


    @property
    def num_labels(self) -> int:
        """
        Gets the number of unique labels.

        Returns
        -------
        int
            The count of unique labels.
        """
        return len(self._label_ids_by_name)

    @property
    def label_values(self) -> list[str]:
        """
        Property to retrieves the values of labels

        Returns
        -------
        list of str
            A list containing the values of all labels.

        """
        return list(self._label_ids_by_name.keys())

    @property
    def label_ids(self) -> list[int]:
        """
        Property to retrieve the list of assigned label identifiers.

        Returns
        -------
        list of str
            A list containing all label identifiers present in the `_labels_ids`
            dictionary.
        """
        return list(self._label_ids_by_name.values())

    def __len__(self) -> int:
        """
        The length of the dataset depends on whether filters are applied or not.

        Returns
        -------
        Number of rows in the dataset
        """
        if self._filter_indexes:
            l = len(self._filter_indexes)
        else:
            l = len(self._df)
        return l

    @property
    def data_length(self) -> int:
        return len(self._df_encoding.iloc[0, 0]) # TODO: Sempre és la mateixa mida?

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an item at the specified index and returns the corresponding
        tensor data and label.

        Parameters
        ----------
        idx : int
            The index of the item to be retrieved.

        Returns
        -------
        tuple of torch.Tensor
            A tuple where the first element is a tensor of the encoding data at
            the specified index, and the second element is a tensor containing
            the label associated with the data.

        Raises
        ------
        IndexError
            If the provided index is negative or exceeds the maximum allowed
            index of the dataset.
        """

        if idx < 0:
            raise IndexError(f"Index {idx} is negative")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is higher than the maximum number of rows ({len(self)})")

        if self._filter_indexes:
            idx = self._filter_indexes[idx]

        # Check if this is 4-row encoding by looking at available columns
        encoding_columns = self._df_encoding.columns
        
        if any(col.startswith('4row_') for col in encoding_columns):
            # This is 4-row encoding - get all 4 rows
            row1 = self._df_encoding.iloc[idx]["4row_1"]
            row2 = self._df_encoding.iloc[idx]["4row_2"] 
            row3 = self._df_encoding.iloc[idx]["4row_3"]
            row4 = self._df_encoding.iloc[idx]["4row_4"]
            
            # Parse strings if needed and convert to tensors
            rows = []
            for row in [row1, row2, row3, row4]:
                if isinstance(row, str):
                    import ast
                    row = ast.literal_eval(row)
                rows.append(torch.tensor(row, dtype=torch.float32))
            
            # Stack to create [4, 313] tensor
            encoding = torch.stack(rows, dim=0)
            
        else:
            # Handle other encoding types (k-mer, bit, etc.)
            encoding = self._df_encoding.iloc[idx, 0]
            if isinstance(encoding, str):
                import ast
                encoding = ast.literal_eval(encoding)
            encoding = torch.tensor(encoding, dtype=torch.float32)

        # Get label
        label_row = self._df.iloc[idx]
        label = label_row[self.label_column_name]
        label = self._label_ids_by_name[label]
        label = torch.tensor([label], dtype=torch.long).view(-1)

        return encoding, label


    def _get_column_value(self, idx:int, column_name: str):
        if idx < 0:
            raise IndexError(f"Index {idx} is negative")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is higher than the maximum number of rows ({len(self)})")

        if self._filter_indexes:
            idx = self._filter_indexes[idx]

        value = self._df.loc[idx, column_name]
        return value

    def get_label_value(self, idx: int) -> str:
        label = self._get_column_value(idx, self.label_column_name)
        assert isinstance(label, str), label
        return label

    def get_label_id(self, idx: int) -> int:
        label = self.get_label_value(idx)
        label_id  = self._label_ids_by_name[label]
        return label_id

    def get_sequence(self, idx: int) -> str:
        sequence = self._get_column_value(idx, CachedDataFrame.SEQUENCE_COLUMN_NAME)
        assert isinstance(sequence, str), sequence
        return sequence

    @property
    def min_sequence_len(self) -> int:
        """
        Gets the minimum sequence length.

        Returns
        -------
        The minimum sequence length.
        """
        if self._filter_indexes:
            min_len = min(len(self._df.loc[idx, CachedDataFrame.SEQUENCE_COLUMN_NAME]) for idx in self._filter_indexes)
        else:
            min_len = self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME].astype(str).str.len().min()

        return min_len

    @property
    def max_sequence_len(self) -> int:
        """
        Gets the maximum sequence length.

        Returns
        -------
        The maximum sequence length.
        """
        if self._filter_indexes:
            max_len = max(len(self._df.loc[idx, CachedDataFrame.SEQUENCE_COLUMN_NAME]) for idx in self._filter_indexes)
        else:
            max_len = self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME].astype(str).str.len().max()

        return max_len
