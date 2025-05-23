import pytest
from torch import Tensor
from feature_extraction.main import SequenceCoder
from dataset.taxo_dataset import TaxoDataset
from dataset.utils import get_default_dataset_path
from dataset.parquet_builder import ParquetBuilder

path = get_default_dataset_path()
test_label_column_name = TaxoDataset.FILTERS_COLUMN_NAMES[0]
test_filter_key = test_label_column_name
sequencecoder = SequenceCoder()


def test_init_label_column_name():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name="non_existent_column", k=1)

    for label_column_name in TaxoDataset.FILTERS_COLUMN_NAMES:
        TaxoDataset(taxo_path=path, label_column_name=label_column_name, k=1)


def test_init_filters():
    with pytest.raises(ValueError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                    filters={"non_existent_column":"xx"})

    with pytest.raises(NotImplementedError):
        TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                    filters={test_filter_key:["xx"]})

    TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                filters={l:"x" for l in TaxoDataset.FILTERS_COLUMN_NAMES})


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


def test_init_indexes_basic():
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)
    assert t.indexes is None


def test_init_indexes_one_column():
    # Test all filters with all the values
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    for column_name in TaxoDataset.FILTERS_COLUMN_NAMES:
        for value in t.df[column_name].unique():
            t:TaxoDataset
            t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                            filters={column_name: value})
            assert t.indexes is not None
            assert (t.df[column_name] == value).sum() == len(t.indexes), f"{column_name}={value}"


def test_init_indexes_multiple_columns():
    # Test all columns with the first value
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    df_tmp=t.df
    fiter_tmp = {}
    for column_name in TaxoDataset.FILTERS_COLUMN_NAMES:
        for value in t.df[column_name].unique():
            if column_name not in fiter_tmp:
                fiter_tmp[column_name] = t.df[column_name][0]
                df_tmp = df_tmp[df_tmp[column_name] == value]
                t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                                filters=fiter_tmp)
                assert t.indexes is not None
                assert len(df_tmp) == len(t.indexes), f"{len(df_tmp) }={len(t.indexes)}"


def test_init_labels_ids_non_filtered():
    for label_column_name in TaxoDataset.FILTERS_COLUMN_NAMES:
        t = TaxoDataset(taxo_path=path, label_column_name=label_column_name, k=1)
        assert len(t.labels_ids) == len(t.df[label_column_name].unique().tolist())
        assert len(t.labels_ids) == t.num_labels


def test_init_labels_ids_filtered():
    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1)

    for column_name in TaxoDataset.FILTERS_COLUMN_NAMES:
        for value in t.df[column_name].unique():
            t:TaxoDataset
            t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=1,
                            filters={column_name: value})
            df_tmp = t.df[t.df[column_name] == value]
            assert len(t.labels_ids) == len(df_tmp[test_label_column_name].unique().tolist())
            assert len(t.labels_ids) == t.num_labels


def test_num_labels():
    # Implemented in test_init_labels_ids_non_filtered
    pass


def test_data_length():
    # TODO
    for k in ParquetBuilder.KMERS_SIZES:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=k)
        assert t.data_length == 4**k

    for b in sequencecoder.bit_mapping:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=b)
        # assert t.data_length == ???

    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=0)
    # assert t.data_length == ???


def test_len():
    # Tested on other tests
    pass


def test_getitem_k():
    for k in ParquetBuilder.KMERS_SIZES:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, k=k)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)
        assert t.data_length == 4**k
        # TODO:


def test_getitem_bits():
    for b in sequencecoder.bit_mapping:
        t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=b)
        d, v = t[1]
        assert isinstance(d, Tensor)
        assert isinstance(v, Tensor)
        # assert t.data_length == ???
        # TODO:

    t = TaxoDataset(taxo_path=path, label_column_name=test_label_column_name, bits=0)
    d, v = t[1]
    assert isinstance(d, Tensor)
    assert isinstance(v, Tensor)
    # assert t.data_length == ???
    # TODO:

