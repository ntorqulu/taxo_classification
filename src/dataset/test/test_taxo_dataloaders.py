import pytest
from torch.utils.data import DataLoader
from constants.taxonomy_labels import TAXONOMY_LABELS
from dataset.cached_dataframe import CachedDataFrame
from dataset.taxo_dataloaders import TaxoDataLoaders
from dataset.utils import get_base_parquets_path, DEFAULT_DATASET_NAME

# Load the dataframe to get a column_name
parquet_dir = get_base_parquets_path()
parquet_dir = [d for d in parquet_dir.iterdir() if d.is_dir()][1]
CachedDataFrame.get_data_frame(parquet_dir)
test_label_column_name = CachedDataFrame.get_level_column_names()[0]
CachedDataFrame.flush_cache()

def test_init():
    t = TaxoDataLoaders(parquets_path=parquet_dir,
                        label_column_name=test_label_column_name,
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
        t = TaxoDataLoaders(parquets_path=parquet_dir,
                            label_column_name=test_label_column_name,
                            batch_size=10, max_rows=r, k=1)
        train_loader: DataLoader
        eval_loader: DataLoader
        test_loader: DataLoader
        train_loader, eval_loader, test_loader = t.data_loaders
        assert len(t.dataset) == len(train_loader.dataset)+len(eval_loader.dataset)+len(test_loader.dataset)

def test_get_label_weights():
    # TODO:
    assert 1 == 0