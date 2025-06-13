import os
import torch
import pandas as pd
from typing import Final
from torch.utils.data import Dataset

from dataset.cached_dataframe import CachedDataFrame
from dataset.parquet_builder import ParquetBuilder
from feature_extraction.main import SequenceCoder
from constants.taxonomy_labels import TAXONOMY_LABELS


class TaxoDataset(Dataset):
    # Discared columns seqID,taxID,scientific_name

    SEQUENCE_CHAR_DIFFERENT_VALUES = 4 # No commit
    SEQUENCE_LENGTH = 300 # No commit
    LABEL_ID_COLUMN_NAME = 'label_id'
    DEFAULT_MAX_SEQUENCE_LEN = 9999999

    def __init__(self,
                 taxo_path: str,
                 label_column_name: str,
                 filters: dict[str, str | list[str]] = None,
                 k: int = None,
                 bits: int = None,
                 max_len_filter: int = DEFAULT_MAX_SEQUENCE_LEN,
                 ):
        """
        Parameters
        ----------
        taxo_path: str
            Path of the dataset to be loaded

        label_column_name: str
            Name of the column that contains the labels.

        filters: dict[str, str | list[str]]
            Dictionary of column_names with the values of the column to filter by. The filter
            values can be strings or lists of strings.

        k: int
            k-mer size. If k is specified, bits must be None.

        bits: int
            bits for bit encoding. If bits is specified, k must be None.

        max_len_filter: int
            Filter sequences that are longer than this value.
        """
        super().__init__()

        if not filters:
            filters = {}
        elif any(r not in TAXONOMY_LABELS for r in filters.keys()):
            raise ValueError(f"Unrecognized filter keys: {filters.keys()}")
        if label_column_name not in TAXONOMY_LABELS:
            raise ValueError(f"Unrecognized label column name: {label_column_name}")
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
        if max_len_filter < 0:
            raise ValueError(f"max_sequence_len must be positive: {max_len_filter}")

        self.taxo_path: str = taxo_path
        self.filters: dict[str, str] = filters
        self.label_column_name: str = label_column_name
        self.k: int | None = k
        self.bits: int | None = bits
        self.max_len_filter: int = max_len_filter

        # DataFrame with the data, but without the encodings.
        self._df: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(self.taxo_path)

        # DataFrame with the encodings
        self._df_encoding: Final[pd.DataFrame] = CachedDataFrame.get_data_frame(self.taxo_path, k=self.k, bits=self.bits)

        # Indexes of the final rows after applying both the filter and max_len_filter, if specified.
        # It is None if no index is applied, meaning all rows are included.
        self._filter_indexes: list[int] | None = self._init_filter_indexes()

        # Dictionary with the mapping between label strings and their assigned IDs.
        self._labels_ids: dict[str, int] = self._init_labels_ids()

    def _init_filter_indexes(self) -> list[int] | None:
        """
        Initializes the list of indexes of the values for the filters
        Initializes the list of indexes for the instances that match the filters.

        Returns
        -------
        The list of indexes, or None if no filters applyied
        """

        if not self.filters and self.max_len_filter == TaxoDataset.DEFAULT_MAX_SEQUENCE_LEN:
            return None

        mask = pd.Series(True, index=self._df.index)

        # Update the mask applying the filters
        for columm_name, value in self.filters.items():
            if isinstance(value, str):
                mask &= self._df[columm_name] == value
            elif isinstance(value, list):
                mask &= self._df[columm_name].isin(value)
            else:
                raise NotImplementedError(f"Value types not implemented: {value}")

        # Undate the mask with the maxiumn sequence length fitler
        if self.max_len_filter < TaxoDataset.DEFAULT_MAX_SEQUENCE_LEN:
            mask &= self._df[CachedDataFrame.SEQUENCE_COLUMN_NAME].str.len() <= self.max_len_filter

        # Return a list with the indexes after applying the filters
        indexes = self._df[mask].index.tolist()
        return indexes

    def _init_labels_ids(self) -> dict[str, int]:
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
        # info(f"There is {len(label_ids)} labels available.")

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
        return len(self._labels_ids)

    @property
    def labels_names(self) -> list[str]:
        """
        Property to retrieves the names of labels

        Returns
        -------
        list of str
            A list containing the names of all labels.

        """
        return list(self._labels_ids.keys())

    @property
    def labels_ids(self) -> list[int]:
        """
        Property to retrieve the list of assigned label identifiers.

        Returns
        -------
        list of str
            A list containing all label identifiers present in the `_labels_ids`
            dictionary.
        """
        return list(self._labels_ids.values())

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

        # Get the row index for the whole subset
        if self._filter_indexes:
            idx = self._filter_indexes[idx]

        # Gete the encoding value to return
        encoding = self._df_encoding.iloc[idx, 0]
        encoding = torch.tensor(encoding, dtype=torch.float32)

        # Get the label id as a tensor
        label_row = self._df.iloc[idx]
        label = label_row[self.label_column_name]
        label = self._labels_ids[label]
        label = torch.tensor([label], dtype=torch.long).view(-1)

        return encoding, label
