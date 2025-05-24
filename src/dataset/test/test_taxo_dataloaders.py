import pytest
from torch.utils.data import DataLoader

from dataset.taxo_dataloaders import TaxoDataLoaders
from dataset.taxo_dataset import TaxoDataset
from dataset.utils import get_default_dataset_path

test_label_column_name = TaxoDataset.FILTERS_COLUMN_NAMES[0]

def test_init():
    t = TaxoDataLoaders(taxo_path=get_default_dataset_path(), label_column_name=test_label_column_name,
                        batch_size=10, k=1)
    train_loader: DataLoader
    eval_loader: DataLoader
    test_loader: DataLoader
    train_loader, eval_loader, test_loader = t.data_loaders
    assert isinstance(train_loader, DataLoader)
    assert isinstance(eval_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

def test_init_max_rows():
    for r in range(1000, 10000, 123):
        t = TaxoDataLoaders(taxo_path=get_default_dataset_path(), label_column_name=test_label_column_name,
                            batch_size=10, max_rows=r, k=1)
        train_loader: DataLoader
        eval_loader: DataLoader
        test_loader: DataLoader
        train_loader, eval_loader, test_loader = t.data_loaders
        assert len(t.dataset) == len(train_loader.dataset)+len(eval_loader.dataset)+len(test_loader.dataset)


