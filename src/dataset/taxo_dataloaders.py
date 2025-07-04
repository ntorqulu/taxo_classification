from pathlib import Path

import torch
from torch import Tensor

from dataset.taxo_dataset import TaxoDataset
from torch.utils.data import random_split, Subset, DataLoader
from dataset.utils import warn
from collections import Counter
from typing import List, Tuple


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
                 use_bert_collate: bool = False,
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

        # Use BERT collate function if specified
        collate_fn = bert_collate_fn if use_bert_collate else None
        num_workers = 0 if use_bert_collate else 4  # Avoid multiprocessing issues with BERT

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        self.eval_loader = torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
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

    def get_label_weights(self,
                          normalize: bool = True,
                          strong: bool = False,
                          min_frequency: float = 1.,
                          ) -> Tensor:
        """
        Computes label weights for the training dataset based on specified parameters.

        This method calculates the weights for each label in the dataset to be used during training.
        The weights can be adjusted for normalization, a stronger inverse relationship with frequency,
        and a minimum frequency threshold to prevent overly high weights for infrequent labels.

        Parameters
        ----------
        normalize : bool, optional
            If True, normalizes the label weights such that the sum of weighted samples matches
            the size of the training dataset. The default is True.
        strong : bool, optional
            If True, computes weights as the inverse square of the label frequency, resulting in
            a stronger penalty for frequent labels. The default is False.
        min_frequency : float, optional
            The minimum frequency threshold to clamp label counts. Any label frequency below this
            value is treated as this value to avoid excessively high weights for rare labels.
            The default is 10.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed label weights for the training dataset.
        """
        len_ds = len(self.train_dataset)
        label_ids = [self.taxo_dataset.get_label_id(idx) for idx in self.train_dataset.indices]
        label_ids_counts = Counter(label_ids)

        # Ensure all possible class IDs are represented (even if count is 0)
        num_labels = self.taxo_dataset.num_labels
        label_counts = [label_ids_counts.get(i, 0) for i in range(num_labels)]
        label_counts = torch.tensor(label_counts, dtype=torch.float32)
        label_counts = torch.clamp(label_counts, min=min_frequency)

        if strong:
            label_weights = 1. / (label_counts ** 2)
        else:
            label_weights = 1. / label_counts

        if normalize:
            total_weighted_samples = torch.sum(label_weights * label_counts)
            label_weights = label_weights * len_ds / total_weighted_samples

        return label_weights

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


def bert_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for BERT model to handle 4-row encoding tensors.
    
    Args:
        batch: List of (data, target) tuples
        
    Returns:
        Tuple of (data_batch, target_batch)
    """
    data_list = []
    target_list = []
    
    for data, target in batch:
        data_list.append(data)
        target_list.append(target)
    
    # Stack targets
    target_batch = torch.cat(target_list, dim=0)
    
    # For 4-row encoding, we need to pad sequences to the same length
    if data_list[0].dim() == 2 and data_list[0].shape[0] == 4:
        # 4-row encoding: [4, seq_len]
        max_seq_len = max(data.shape[1] for data in data_list)
        batch_size = len(data_list)
        
        # Create padded tensor
        padded_data = torch.zeros(batch_size, 4, max_seq_len, dtype=torch.float32)
        
        for i, data in enumerate(data_list):
            seq_len = data.shape[1]
            padded_data[i, :, :seq_len] = data
        
        data_batch = padded_data
    else:
        # For other encodings (k-mer), just stack
        data_batch = torch.stack(data_list, dim=0)
    
    return data_batch, target_batch


def create_bert_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    """
    Create a DataLoader for BERT models with the custom collate function.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (set to 0 to avoid multiprocessing issues)
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
        collate_fn=bert_collate_fn
    )


