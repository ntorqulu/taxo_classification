import pytest
import os

from .taxo_dataset import *
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

# gunzip -c final_taxonomy.csv.gz > /tmp/final_taxonomy.csv
TAXO_PATH = "/tmp/database.csv"

def max_dif(a: int, b: int, maxdif=1) -> bool:
    return abs(a - b) <= maxdif

def num_rows_taxopath():
    with open(TAXO_PATH) as f:
        num_rows = sum(1 for _ in f) - 1
    return num_rows

def test_cache():
    # WARNING! It has to be the first test
    assert os.path.exists(TAXO_PATH)
    pikle_path = TaxoDataset.get_pickle_path(TAXO_PATH)
    if os.path.exists(pikle_path):
        os.remove(pikle_path)
    assert TaxoDataset._cached_df is None
    TaxoDataset(TAXO_PATH, split='all')
    assert TaxoDataset._cached_df is not None
    assert os.path.exists(pikle_path)

def test_split_indexes():
    dataset_length: int = 12_000
    start_idx, end_idx = TaxoDataset.split_indexes(length=dataset_length, split='all', train_pct=0.8, eval_pct=0.1)
    assert start_idx == 0
    assert end_idx == 12_000 - 1
    start_idx, end_idx = TaxoDataset.split_indexes(length=dataset_length, split='train', train_pct=0.8, eval_pct=0.1)
    assert start_idx == 0
    assert end_idx == 9_600 - 1
    start_idx, end_idx = TaxoDataset.split_indexes(length=dataset_length, split='eval', train_pct=0.8, eval_pct=0.1)
    assert start_idx == 9_600
    assert end_idx == 10_800 - 1
    start_idx, end_idx = TaxoDataset.split_indexes(length=dataset_length, split='test', train_pct=0.8, eval_pct=0.1)
    assert start_idx == 10_800
    assert end_idx == 12_000 - 1


def test_len():
    def test_len_maxrows(max_rows: int | float):
        if isinstance(max_rows, float):
            num_rows = int(num_rows_taxopath() * max_rows)
        else:
            num_rows = max_rows

        all_dataset = TaxoDataset(TAXO_PATH, split='all', max_rows=max_rows)
        assert max_dif(len(all_dataset), num_rows,1), f"all: {max_rows=}"

        train_dataset = all_dataset.new_split('train')
        assert max_dif(len(train_dataset), int(num_rows * TaxoDataset.TRAIN_PCT),1), f"train: {max_rows=}"

        eval_dataset = all_dataset.new_split('eval')
        assert max_dif(len(eval_dataset), int(num_rows * TaxoDataset.EVAL_PCT), 1), f"eval: {max_rows=}"

        test_dataset = all_dataset.new_split('test')
        assert max_dif(len(test_dataset), int(num_rows * TaxoDataset.TEST_PCT), 2), f"test: {max_rows=}"

    test_len_maxrows(max_rows=1.)
    test_len_maxrows(max_rows=.5)
    test_len_maxrows(max_rows=.1)
    test_len_maxrows(max_rows=1000)
    test_len_maxrows(max_rows=12_345)
    test_len_maxrows(max_rows=100_000)

def test_class_instance_errors():
    with pytest.raises(FileNotFoundError):
        TaxoDataset('/tmp/does_not_exist_dont_worry_its_a_test', split='all')

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split=None)

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='invented_split')

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='all', max_rows=-1)

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='all', max_rows=0)

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='all', max_rows=0.)

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='all', max_rows=5.)

    with pytest.raises(ValueError):
        TaxoDataset(TAXO_PATH, split='all', ranks={'INVENTED_COLUMN':'a'})

def test_index_access():
    def test_limits(taxodataset: TaxoDataset):
        assert taxodataset[0]
        with pytest.raises(IndexError):
            assert taxodataset[len(taxodataset)]

    all_dataset = TaxoDataset(TAXO_PATH, split='all')

    with pytest.raises(IndexError):
        assert all_dataset[-1]

    with pytest.raises(IndexError):
        assert all_dataset[len(all_dataset)]

    train_dataset = all_dataset.new_split('train')
    test_limits(train_dataset)

    eval_dataset = all_dataset.new_split('eval')
    test_limits(eval_dataset)

    test_dataset = all_dataset.new_split('test')
    test_limits(test_dataset)

