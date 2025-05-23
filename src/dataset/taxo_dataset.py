import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from dataset.cached_dataframe import CachedDataFrame
from dataset.utils import info
from dataset.parquet_builder import ParquetBuilder
from feature_extraction.main import SequenceCoder


class TaxoDataset(Dataset):
    FILTERS_COLUMN_NAMES: list[str] = [
        'kingdom_name',
        'phylum_name',
        'class_name',
        'order_name'
    ]
    # Discared columns seqID,taxID,scientific_name

    SEQUENCE_COLUMN_NAME = 'sequence'
    LABEL_ID_COLUMN_NAME = 'label_id'

    def __init__(self,
                 taxo_path: str,
                 label_column_name: str,
                 filters: dict[str, str] = None,
                 k: int = None,
                 bits: int = None,
                 ):
        super().__init__()

        if not filters:
            filters = {}
        elif any(r not in TaxoDataset.FILTERS_COLUMN_NAMES for r in filters.keys()):
            raise ValueError(f"Unrecognized filter keys: {filters.keys()}")
        if label_column_name not in self.FILTERS_COLUMN_NAMES:
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

        self.taxo_path: str = taxo_path
        self.filters: dict[str, str] = filters
        self.label_column_name: str = label_column_name
        self.k: int | None = k
        self.bits: int | None = bits
        self.df: pd.DataFrame = CachedDataFrame.get_data_frame(self.taxo_path)
        self.df_encoding: pd.DataFrame = CachedDataFrame.get_data_frame(self.taxo_path, k=self.k, bits=self.bits)
        self.indexes: list[int] | None = self._init_indexes()
        self.labels_ids: dict[str, int] = self._init_labels_ids()

    def _init_indexes(self) -> list[int] | None:
        """
        Initializes the list of indexes of the values for the filters
        Returns
        -------
        The list of indexes or None if no filters
        """
        if not self.filters:
            return None

        missing = set(self.filters.keys()) - set(self.df.columns) # TODO: AIxò cal?
        if missing:
            raise ValueError(f"Unrecognized filter key(s): {', '.join(missing)} (column(s) do not exist)")

        mask = pd.Series(True, index=self.df.index)
        for colum_name, values in self.filters.items():
            mask &= self.df[colum_name].isin([values])
        indexes = self.df[mask].index.tolist()
        return indexes

    def _init_labels_ids(self) -> dict[str, int]:
        """
        Returns
        -------
        Returns an id for each label value of teh  current self.label_column_name
        """
        label_values: pd.Series
        if self.indexes is None:
            label_values = self.df[self.label_column_name]
        else:
            label_values = self.df.loc[self.indexes, self.label_column_name]

        unique_values: list[str] = label_values.unique().tolist()
        label_ids = {l[1]: l[0] for l in enumerate(unique_values)}
        info(f"There is {len(label_ids)} labels available.")

        return label_ids

    @property
    def num_labels(self) -> int:
        return len(self.labels_ids)

    @property
    def data_length(self) -> int:
        return len(self.df_encoding.iloc[0, 0]) # TODO: Sempre és la mateixa mida?

    def __len__(self) -> int:
        if self.indexes:
            l = len(self.indexes)
        else:
            l = len(self.df)
        return l

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            raise IndexError(f"Index {idx} is negative")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is higher than the maximum number of rows ({len(self)})")

        if self.indexes:
            idx = self.indexes[idx]

        label_row = self.df.iloc[idx]
        label = label_row[self.label_column_name]
        label = self.labels_ids[label]
        label = torch.tensor([label], dtype=torch.long).view(-1)

        encoding = self.df_encoding.iloc[idx, 0]
        encoding = torch.tensor(encoding, dtype=torch.float32)

        return encoding, label
