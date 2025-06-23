from pathlib import Path

import torch
from dataset.taxo_dataset import TaxoDataset
from torch.utils.data import random_split, Subset, DataLoader
from dataset.utils import warn
from collections import Counter


class TaxoDataLoaders:
    TRAIN_PCT = 0.8
    EVAL_PCT = 0.1
    assert 0.6 < TRAIN_PCT + EVAL_PCT <= 0.95

    def __init__(self,
                 parquets_path: Path,
                 label_column_name: str,
                 batch_size: int,
                 max_rows: int | float = 1.,
                 k: int = None,
                 bits: int = None,
                 seq_len_filter: int | None = None,
                 stratify = None,
                 ):

        self.taxo_dataset = TaxoDataset(parquets_path=parquets_path,
                                        label_column_name=label_column_name,
                                        k=k,
                                        bits=bits,
                                        seq_len_filter=seq_len_filter)

        max_rows = self._init_max_rows(max_rows)
        if max_rows <= len(self.taxo_dataset):
            self.dataset = Subset(self.taxo_dataset, range(0, max_rows))
        else:
            self.dataset = self.taxo_dataset

        train_size = int(len(self.dataset) * TaxoDataLoaders.TRAIN_PCT)
        eval_size = int(len(self.dataset) * TaxoDataLoaders.EVAL_PCT)
        test_size = len(self.dataset) - eval_size - train_size

        self.train_dataset, self.eval_dataset, self.test_dataset = random_split(self.dataset,
                                                                                [train_size, eval_size, test_size])

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.eval_loader = torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def _init_max_rows(self, max_rows: int | float) -> int:
        if isinstance(max_rows, float):
            if max_rows <= 0 or max_rows > 1:
                raise ValueError("It its a float, max_row has to be between 0 and 1.")
            max_rows = int(max_rows * len(self.taxo_dataset))
            return max_rows

        if isinstance(max_rows, int):
            if max_rows <= 0:
                raise ValueError("Max rows has to be a positive number.")
            if len(self.taxo_dataset) < max_rows:
                warn(f"{max_rows=} is higher than the number of total rows ({len(self.taxo_dataset)})."
                     f" Adjusting to {len(self.taxo_dataset)} rows.")
                return len(self.taxo_dataset)
            return max_rows

        raise ValueError(f"max_rows has to be a float or an int, not {type(max_rows)}")

    def get_labels(self) -> dict[str, dict[str, tuple[int, float]]]:
        """
        Get label statistics for train, eval and test datasets.

        Returns
        -------
        dict
            Nested dictionary: {dataset_name: {label: (count, percentage)}}

        Example: {'train': {'class_A': (100, 0.8), 'class_B': (25, 0.2)}}
        """

        labels: dict[str, dict[str, tuple[int, float]]] = {}
        for ds, name in ((self.train_dataset, 'train'), (self.eval_dataset, 'eval'), (self.test_dataset, 'test')):
            len_ds = len(ds)
            labels_ds = [self.taxo_dataset.get_label(idx) for idx in ds.indices]
            label_counts = Counter(labels_ds).most_common()
            labels[name] = {name: (n, n/len_ds) for name, n in label_counts}
            assert sum(l[0] for l in labels[name].values()) == len_ds
            assert abs(sum(l[1] for l in labels[name].values()) - 1.0) < 0.1
        return labels

    @property
    def data_loaders(self) -> (DataLoader, DataLoader, DataLoader):
        return self.train_loader, self.eval_loader, self.test_loader

    @property
    def num_labels(self) -> int:
        return self.taxo_dataset.num_labels

    @property
    def data_length(self) -> int:
        return self.taxo_dataset.data_length

    @property
    def dataset_length(self) -> int:
        return len(self.taxo_dataset)

    @property
    def min_sequence_len(self) -> int:
        return self.taxo_dataset.min_sequence_len

    @property
    def max_sequence_len(self) -> int:
        return self.taxo_dataset.max_sequence_len


