import pytest
from torch import Tensor

from constants.taxonomy_labels import TAXONOMY_LABELS
from dataset.parquet_builder import ParquetBuilder
from dataset.taxo_dataset import TaxoDataset
from dataset.utils import get_default_dataset_path
from feature_extraction.main import SequenceCoder
from dataset.cached_dataframe import CachedDataFrame

path = get_default_dataset_path()
test_label_column_name = list(TAXONOMY_LABELS.keys())[0]
test_filter_key = test_label_column_name
sequencecoder = SequenceCoder()


def test_init_label_column_name():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name="non_existent_column", k=1)

    for label_column_name in TAXONOMY_LABELS:
        TaxoDataset(taxo_path=path, label_column_name=label_column_name, k=1)


def test_init_filters():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                    filters={"non_existent_column":"xx"})

    with pytest.raises(NotImplementedError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                    filters={test_filter_key:["xx"]})

    TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                filters={l:"x" for l in TAXONOMY_LABELS})


def test_init_k():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=-1)

    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=0)

    TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)


def test_init_bits():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=-1)

    TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=0)  # 4 row
    TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=1)


def test_init_k_bits():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name)
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1, bits=1)

def test_init_max_len_filter():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1, max_len_filter=-1)

    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)
    min_len = min([len(l) for l in t._df[CachedDataFrame.SEQUENCE_COLUMN_NAME]])
    max_len = max([len(l) for l in t._df[CachedDataFrame.SEQUENCE_COLUMN_NAME]])
    for l in range(min_len, max_len+1):
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1, max_len_filter=l)
        assert len(t) == len([t for t in t._df[CachedDataFrame.SEQUENCE_COLUMN_NAME] if len(t) <= l])

def test_init_indexes_basic():
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)
    assert t._filter_indexes is None


def test_init_indexes_one_column():
    # Test all filters with all the values
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    for column_name in TAXONOMY_LABELS:
        for value in t._df[column_name].unique():
            t:TaxoDataset
            t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                            filters={column_name: value})
            assert t._filter_indexes is not None
            assert (t._df[column_name] == value).sum() == len(t._filter_indexes), f"{column_name}={value}"


def test_init_indexes_multiple_columns():
    # Test all columns with the first value
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    df_tmp=t._df
    fiter_tmp = {}
    for column_name in TAXONOMY_LABELS:
        for value in t._df[column_name].unique():
            if column_name not in fiter_tmp:
                fiter_tmp[column_name] = t._df[column_name][0]
                df_tmp = df_tmp[df_tmp[column_name] == value]
                t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                                filters=fiter_tmp)
                assert t._filter_indexes is not None
                assert len(df_tmp) == len(t._filter_indexes), f"{len(df_tmp) }={len(t._filter_indexes)}"


def test_init_labels_ids_non_filtered():
    for label_column_name in TAXONOMY_LABELS:
        t = TaxoDataset(taxo_path=path, label_column_name=label_column_name, k=1)
        assert len(t.labels_ids) == len(t._df[label_column_name].unique().tolist())
        assert len(t.labels_ids) == t.num_labels


def test_init_labels_ids_filtered():
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    for column_name in TAXONOMY_LABELS:
        for value in t._df[column_name].unique():
            t:TaxoDataset
            t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                            filters={column_name: value})
            df_tmp = t._df[t._df[column_name] == value]
            assert len(t.labels_ids) == len(df_tmp[test_label_column_name].unique().tolist())
            assert len(t.labels_ids) == t.num_labels


def test_num_labels():
    # Implemented in test_init_labels_ids_non_filtered
    pass


def test_len():
    # Tested on other tests
    pass


def test_getitem_k():
    for k in ParquetBuilder.KMERS_SIZES:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=k)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)

def test_getitem_bits():
    for b in sequencecoder.bit_mapping:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=b)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)

    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=0)
    d, v = t[1]
    assert isinstance(d, Tensor)
    assert isinstance(v, Tensor)
