import os
import random
import time
import torch
import pandas as pd
from torch import Tensor
from typing import Callable, Literal
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from .util import info, warn

class TaxoDataset(Dataset):
    RANKS_COLUMN_NAMES: list[str] = [
        'superkingdom_name',
        'kingdom_name',
        'phylum_name',
        'class_name',
        'order_name',
        'family_name',
        'genus_name',
        'species_name'
    ]
    # Discared columns seqID,taxID,scientific_name

    # We cannot use parameters to set this value because we can have different instances of TaxoDataset
    # using the cached DataFrame and the percentatge of the distribution has to be the same for all the
    # instances
    TRAIN_PCT = 0.8
    EVAL_PCT = 0.1
    TEST_PCT = 0.1
    assert TRAIN_PCT + EVAL_PCT + TEST_PCT == 1.0

    _cached_df: pd.DataFrame = None
    _cached_path: str = None
    _cached_file_size: int = 0
    _cached_memory_usage: int = 0

    @classmethod
    def _is_df_cached(cls, taxo_path: str) -> bool:
        if cls._cached_df is None:
            return False
        if cls._cached_path != taxo_path:
            return False
        file_size = os.path.getsize(taxo_path)
        if cls._cached_file_size != file_size:
            return False
        return True

    def __init__(self,
                 taxo_path: str,
                 # TODO: sequence_encoder: BaseSequenceEncoder,
                 split: Literal['all', 'train', 'eval', 'test'],
                 max_rows: int | float = 1.,
                 ranks: dict[str, str] = None,
                 sequence_column_name: str = "sequence",
                 random_seed: int = random.randint(0, 100000),
                 use_cache: bool = True,
                 ):
        """
        taxo_path: Path to a CSV file with the taxonomy. It has to have the a sequence_column_name column and a "label"
                   column. Optionally, it can have other columns with the taxonomic ranks (depending on the ranks argument).
        split: Split to load. It can be "all", "train", "eval" or "test".
        max_rows: Maximum number of rows to load. If the value is int, it is used as the number of rows to load. If the
                  number is float, it is used as a percentage of the total number of rows. Default is 1. as 100%
                  This value reffers to all the splits together: split, eval and train.
        ranks: A dictionary with the names of the taxonomic ranks and the names of the columns that contain the values.
               The dictionary keys has to be one of the RANKS_COLUMN_NAMES. If the dictionary is empty, all the
               taxonomic are loaded.
        sequence_column_name: Sequence column name. Default is "sequence".
        use_cache: If True, first try to use a memory cache of the DataFrame. If the cache is not valid, the dataset is
                    loaded from a cached file, if not found, it is loaded from the original file and saved in a cached .
                    memory and file for future use.
                   If False, the dataset is loaded from the original file.
        """
        super().__init__()

        if not os.path.isfile(taxo_path):
            raise FileNotFoundError(taxo_path)
        if not split:
            raise ValueError("split has to be a non-empty string")
        if split not in ('all', 'train', 'eval', 'test'):
            raise ValueError(f"Unrecognized split: {split}")
        if max_rows <= 0:
            raise ValueError("max_rows has to be a positive number.")
        if isinstance(max_rows, float) and max_rows > 1.:
            raise ValueError("If max_rows is a float, it has to be <= 1.0 (100%)")
        if not ranks:
            ranks = {}
        elif any(k not in TaxoDataset.RANKS_COLUMN_NAMES for k in ranks.keys()):
            raise ValueError(f"Unrecognized ranks keys: {ranks.keys()}")

        self.taxo_path = taxo_path
        self.ranks = ranks
        self.sequence_column_name = sequence_column_name
        self.label_column_name = TaxoDataset.RANKS_COLUMN_NAMES[0] # TODO:
        self.use_cache = use_cache
        self.random_seed = random_seed
        self.split = split
        self.df: pd.DataFrame = self._init_df()
        info(self.df.columns)
        self.max_rows: int = self._init_max_rows(max_rows)
        if self.max_rows == len(self.df):
            self.row_indexes = []
            self.start_index, self.end_index = TaxoDataset.split_indexes(length=len(self.df), split=self.split)
        else:
            self.row_indexes = self._init_row_indexes()
            self.start_index, self.end_index = None, None
        # TODO: self.sequence_encoder = sequence_encoder

    def _init_df(self) -> pd.DataFrame:
        if self.use_cache and TaxoDataset._is_df_cached(self.taxo_path):
            df = TaxoDataset._cached_df
            info(f"Using cached contents of {TaxoDataset._cached_path }")
            return df

        pickle_path = TaxoDataset.get_pickle_path(self.taxo_path)
        t0 = time.time()
        if self.use_cache and os.path.exists(pickle_path):
            info(f"Loading {pickle_path}")
            df = pd.read_pickle(pickle_path)
            save_parquet = False
        else:
            info(f"Loading {self.taxo_path} ")
            t0 = time.time()
            df = pd.read_csv(
                self.taxo_path,
                low_memory=False,
                usecols = [self.sequence_column_name] + TaxoDataset.RANKS_COLUMN_NAMES
            ) # low_memory=False to avoid warning about mixed types
            save_parquet = self.use_cache

        seconds = time.time() - t0
        info(f"Loaded {len(df):,} rows in {seconds:.1f} seconds ")
        if save_parquet:
            info(f"Saving {pickle_path}")
            df.to_pickle(pickle_path)
            info(f"Saved")
        TaxoDataset._cached_df = df
        TaxoDataset._cached_path = self.taxo_path
        TaxoDataset._cached_file_size = os.path.getsize(self.taxo_path)

        return df

    def _init_max_rows(self, max_rows: int | float) -> int:
        if isinstance(max_rows, float):
            pct_rows = max_rows
            return int(len(self.df) * pct_rows)
        elif isinstance(max_rows, int):
            if len(self.df) < max_rows:
                warn(f"{max_rows=} is higher than the number of total rows ({len(self.df)})."
                     f" Adjusting to {len(self.df)} rows.")
                return len(self.df)
            else:
                return max_rows
        raise ValueError(f"max_rows has to be a float or an int, not {type(max_rows)}")

    def _init_row_indexes(self) -> list[int]:
        local_random = random.Random(self.random_seed)
        random_indexes = local_random.sample(range(len(self.df)), self.max_rows)
        start_idx, end_idx = TaxoDataset.split_indexes(length=len(random_indexes), split=self.split)
        row_indexes = random_indexes[start_idx:end_idx]
        return row_indexes

    @property
    def df_memory_usage_mb(self) -> int:
        if TaxoDataset._cached_memory_usage == 0:
            TaxoDataset._cached_memory_usage = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return TaxoDataset._cached_memory_usage

    def new_split(self, split: Literal['all', 'train', 'eval', 'test']) -> 'TaxoDataset':
        """
        Splits the dataset based on the specified subset type.

        This method creates a new instance of the `TaxoDataset` class that corresponds
        to the specified subset of the dataset. The subset type can be one of
        'all', 'train', 'eval', or 'test'. The method allows working with different
        partitions of the dataset without modifying the original dataset instance.

        Parameters
        ----------
        split : Literal['all', 'train', 'eval', 'test']
            Specifies the subset type of the dataset to return. It determines whether
            the entire dataset or a specific split (training, evaluation, or test)
            is accessed.

        Returns
        -------
        TaxoDataset
            A new instance of the `TaxoDataset` class corresponding to the specified
            subset type.
        """
        return TaxoDataset(taxo_path=self.taxo_path,
                           sequence_column_name=self.sequence_column_name,
                           ranks=self.ranks,
                           split=split,
                           max_rows=self.max_rows,
                           random_seed=self.random_seed,
                           use_cache=self.use_cache)

    def __len__(self) -> int:
        if self.row_indexes:
            l = len(self.row_indexes)
        else:
            l = self.end_index - self.start_index + 1
        return l

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if idx < 0:
            raise IndexError(f"Index {idx} is negative")
        if idx >= len(self):
            raise IndexError(f"Index {idx} is higher than the maximum number of rows ({self.max_rows})")

        if self.row_indexes:
            idx = self.row_indexes[idx]
        else:
            idx = self.start_index + idx
        row = self.df.iloc[idx]
        label = row[self.label_column_name]
        sequence = row[self.sequence_column_name]

        # TODO: use encoder provided
        # encoder = OneHotEncoder(sparse_output=False)
        # onehot_sequences = encoder.fit_transform(sequence)

        #tensor = torch.from_numpy(torch.ra.to_numpy())
        tensor = torch.torch.rand(3, 4)
        return tensor, label

    @staticmethod
    def split_indexes(length: int,
                      split: Literal['all', 'train', 'eval', 'test'],
                      train_pct: float = TRAIN_PCT,
                      eval_pct: float = EVAL_PCT,
                      ) -> tuple[int, int]:
        """
        Splits a dataset's index range into subsets based on specified fractions.

        This function calculates index ranges for train, evaluation, and test
        subsets of data using the given proportions. The user specifies which
        split subset range they wish to retrieve through the `split` parameter.

        The function assumes that the total size of dataset corresponds to the
        provided `length` parameter. The sum of train, evaluation, and test proportions
        (derived from `train_pct` and `eval_pct`, with the remaining allocated
        to test) must exactly equal the dataset size.

        Parameters
        ----------
        length : int
            Total number of elements in the dataset. Must be a positive integer.
        split : {'all', 'train', 'eval', 'test'}
            The desired subset of data to compute index range for. Options
            include:

            - 'all': Returns indices for the entire dataset.
            - 'train': Returns indices for the training subset.
            - 'eval': Returns indices for the evaluation/validation subset.
            - 'test': Returns indices for the testing subset.
        train_pct : float
            The fraction of the dataset to allocate for training. Must be in
            the range [0, 1].
        eval_pct : float
            The fraction of the dataset to allocate for evaluation. Must be in
            the range [0, 1].

        Returns
        -------
        tuple of (int, int)
            A tuple of two integers representing the start and end indices
            (inclusive) for the requested subset based on the `split` parameter.

        Raises
        ------
        ValueError
            If an unrecognized value is provided for the `split` parameter.
        """
        train_start = 0
        train_end = int(length * train_pct) - 1
        eval_start = train_end + 1
        eval_end = eval_start + int(length * eval_pct) - 1
        test_start = eval_end + 1
        test_end = length - 1
        test_pct = (100 - train_pct*100 - eval_pct*100)/100 # We do this to avoid floating point errors
        assert test_end -2 <= test_start + int(length * test_pct) - 1 < test_end + 1
        assert (train_end - train_start + 1) + (eval_end - eval_start + 1) + (test_end - test_start + 1) == length
        if split == 'train':
            start_idx, end_idx = train_start, train_end
        elif split == 'eval':
            start_idx, end_idx = eval_start, eval_end
        elif split == 'test':
            start_idx, end_idx = test_start, test_end
        elif split == 'all':
            start_idx, end_idx = 0, length - 1
        else:
            raise ValueError(f"Unrecognized split: {split}")
        return start_idx, end_idx

    @staticmethod
    def get_pickle_path(csv_path: str):
        if csv_path.lower().endswith(".csv"):
            return csv_path[:-4] + ".pkl"
        else:
            return csv_path + ".pkl"

def __create_pickle_file(taxo_path: str,):
    TaxoDataset(taxo_path=taxo_path, split='all')
    print(TaxoDataset(taxo_path=taxo_path, split='all').df_memory_usage_mb, "MB")

if __name__ == "__main__":
    __create_pickle_file('/tmp/final_taxonomy.csv')