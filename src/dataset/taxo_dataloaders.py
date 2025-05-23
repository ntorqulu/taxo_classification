import torch
from dataset.taxo_dataset import TaxoDataset
from torch.utils.data import random_split, Subset, DataLoader
from dataset.utils import warn

class TaxoDataLoaders:
    TRAIN_PCT = 0.8
    EVAL_PCT = 0.1
    def __init__(self,
                 taxo_path: str,
                 label_column_name: str,
                 batch_size: int,
                 max_rows: int | float = 1.,
                 k: int = None,
                 bits: int = None,
                 ):
        self.taxo_dataset = TaxoDataset(taxo_path=taxo_path, label_column_name=label_column_name, k=k, bits=bits)

        max_rows = self._init_max_rows(max_rows)
        if max_rows <= len(self.taxo_dataset):
            self.dataset = Subset(self.taxo_dataset, range(0, max_rows))
        else:
            self.dataset = self.taxo_dataset

        train_size = int(len(self.dataset) * TaxoDataLoaders.TRAIN_PCT)
        eval_size = int(len(self.dataset) * TaxoDataLoaders.EVAL_PCT)
        test_size = len(self.dataset) - eval_size - train_size

        train_dataset, eval_dataset, test_dataset = random_split(self.dataset, [train_size, eval_size, test_size])

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,  # We do the suffle on the DataLoader to increase performance
        )

        self.eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=False,  # We do the suffle on the DataLoader to increase performance
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  # We do the suffle on the DataLoader to increase performance
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

    @property
    def data_loaders(self) -> (DataLoader, DataLoader, DataLoader):
        return self.train_loader, self.eval_loader, self.test_loader

    @property
    def num_labels(self) -> int:
        return self.taxo_dataset.num_labels

    @property
    def data_length(self) -> int:
        return self.taxo_dataset.data_length
