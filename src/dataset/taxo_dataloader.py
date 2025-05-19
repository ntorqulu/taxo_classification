import torch
import numpy as np
from typing import Callable
from dataset.taxo_dataset import TaxoDataset


class TaxoDataLoaders:

    def __init__(self,
                 taxo_path: str,
                 sequence_encoder: Callable[[str, int], np.ndarray],
                 batch_size: int,
                 max_rows: int | float = 1.,
                 ):
        self.all_dataset = TaxoDataset(
            taxo_path=taxo_path,
            split='all',
            sequence_encoder=sequence_encoder,
            max_rows=max_rows,
        )

        train_dataset = self.all_dataset.new_split('all')
        eval_dataset = self.all_dataset.new_split('eval')
        test_dataset = self.all_dataset.new_split('test')

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False, # We do the suffle on the DataLoader to increase performance
        )

        self.eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False, # We do the suffle on the DataLoader to increase performance
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False, # We do the suffle on the DataLoader to increase performance
        )

    @property
    def num_labels(self) -> int:
        return self.all_dataset.num_labels

    @property
    def data_length(self) -> int:
        return self.all_dataset.data_length
