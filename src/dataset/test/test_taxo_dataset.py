import random

import pytest
from torch import Tensor

from dataset.parquet_builder import ParquetBuilder
from dataset.taxo_dataset import TaxoDataset
from dataset.utils import get_base_parquets_path, DEFAULT_DATASET_NAME
from feature_extraction.main import SequenceCoder
from dataset.cached_dataframe import CachedDataFrame
from collections import Counter


# Load the dataframe to get a column_name
parquet_dir = get_base_parquets_path()
parquet_dir = [d for d in parquet_dir.iterdir() if d.is_dir()][1]
CachedDataFrame.get_data_frame(parquet_dir)
test_label_column_name = CachedDataFrame.get_level_column_names()[0]
CachedDataFrame.flush_cache()

test_filter_key = test_label_column_name
sequencecoder = SequenceCoder()
parquets_path = get_base_parquets_path() / DEFAULT_DATASET_NAME

def Xtest_init_label_column_name():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name="non_existent_column", k=1)

    for label_column_name in CachedDataFrame.get_level_column_names():
        TaxoDataset(parquets_path=parquets_path, label_column_name=label_column_name, k=1)

def Xtest_init_k():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=-1)

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=0)

    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)


def Xtest_init_bits():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, bits=-1)

    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, bits=0)  # 4 row
    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, bits=1)


def Xtest_init_k_bits():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name)
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1, bits=1)


def Xtest_init_value_filters():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    value_filters={"non_existent_column":"xx"})

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    value_filters={test_filter_key:[1]})

    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                value_filters={test_filter_key:["xx"]})

    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                value_filters={l:"x" for l in CachedDataFrame.get_level_column_names()})

def Xtest_init_min_cardinality_filters():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    min_cardinality_filters={"non_existent_column":1})

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    min_cardinality_filters={test_filter_key: "x"})

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    min_cardinality_filters={test_filter_key:  0})

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                    min_cardinality_filters={test_filter_key: -1})

    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                min_cardinality_filters={l:10 for l in CachedDataFrame.get_level_column_names()})


def Xtest_init_seq_len_filter():
    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1, seq_len_filter=-1)

    with pytest.raises(ValueError):
        TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1, seq_len_filter=0)

    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)
    for l in range(CachedDataFrame.get_min_sequence_len(), CachedDataFrame.get_max_sequence_len()+1):
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1, seq_len_filter=l)
        assert len(t) == len([t for t in t._df[CachedDataFrame.SEQUENCE_COLUMN_NAME] if len(t) == l])

def Xtest_init_indexes_basic():
    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)
    assert t._filter_indexes is None


def Xtest_init_indexes_one_column():
    # Test all filters with all the values
    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)

    for column_name in CachedDataFrame.get_level_column_names():
        values = t._df[column_name].unique().tolist()
        for value in random.sample(values, min(100, len(values))):
            t:TaxoDataset
            t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                            value_filters={column_name: value})
            assert t._filter_indexes is not None
            assert (t._df[column_name] == value).sum() == len(t._filter_indexes), f"{column_name}={value}"


def Xtest_init_indexes_multiple_columns():
    # Test all columns with the first value
    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)

    df_tmp=t._df
    fiter_tmp = {}
    for column_name in CachedDataFrame.get_level_column_names():
        for value in t._df[column_name].unique():
            if column_name not in fiter_tmp:
                fiter_tmp[column_name] = t._df[column_name][0]
                df_tmp = df_tmp[df_tmp[column_name] == value]
                t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                                value_filters=fiter_tmp)
                assert t._filter_indexes is not None
                assert len(df_tmp) == len(t._filter_indexes), f"{len(df_tmp) }={len(t._filter_indexes)}"

def Xtest_init_labels_ids_non_filtered():
    for label_column_name in CachedDataFrame.get_level_column_names():
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=label_column_name, k=1)
        assert len(t.label_ids) == len(t._df[label_column_name].unique().tolist())
        assert len(t.label_ids) == t.num_labels


def Xtest_init_labels_ids_value_filtered():
    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)

    for column_name in CachedDataFrame.get_level_column_names():
        values = t._df[column_name].unique().tolist()
        for value in random.sample(values, min(100, len(values))):
            t:TaxoDataset
            t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1,
                            value_filters={column_name: value})
            df_tmp = t._df[t._df[column_name] == value]
            assert len(t.label_ids) == len(df_tmp[test_label_column_name].unique().tolist())
            assert len(t.label_ids) == t.num_labels


def Xtest_num_labels():
    # Implemented in test_init_labels_ids_non_filtered
    pass


def Xtest_len():
    # Tested on other tests
    pass

def Xtest_min_max_sequencelen():
    range_min_max = range(CachedDataFrame.get_min_sequence_len(), CachedDataFrame.get_max_sequence_len()+1)
    for seq_len_filter in [None] + list(range_min_max):
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1, seq_len_filter=seq_len_filter)
        min_seq = min(len(t.get_sequence(idx)) for idx in range(0, len(t)))
        max_seq = max(len(t.get_sequence(idx)) for idx in range(0, len(t)))
        assert t.min_sequence_len == min_seq, f"{t.min_sequence_len} != {min_seq}"
        assert t.max_sequence_len == max_seq, f"{t.max_sequence_len} != {max_seq}"

def Xtest_getitem_k():
    for k in ParquetBuilder.KMERS_SIZES:
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=k)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)

def Xtest_getitem_bits():
    for b in sequencecoder.bit_mapping:
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, bits=b)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)

    t = TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, bits=0)
    d, v = t[1]
    assert isinstance(d, Tensor)
    assert isinstance(v, Tensor)

def Xtest_get_label_id():
    assert 1 == 0

def test_min_cardinality_filters():
    TaxoDataset(parquets_path=parquets_path, label_column_name=test_label_column_name, k=1)
    for column_name in CachedDataFrame.get_level_column_names():
        values = CachedDataFrame._df[column_name].to_list()
        cardinality = Counter(values)
        avg_cardinality = sum(cardinality.values())//len(cardinality)
        t = TaxoDataset(parquets_path=parquets_path, label_column_name=column_name, k=1,
                        min_cardinality_filters={column_name: avg_cardinality})
        label_values = [k for k in cardinality.keys() if cardinality[k] >= avg_cardinality]
        assert len(t.label_values) == len(label_values)
        assert set(t.label_values) == set(label_values)
