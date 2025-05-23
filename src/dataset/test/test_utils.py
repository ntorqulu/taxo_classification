import pytest
from dataset.utils import *
from pathlib import Path


def test_get_default_data_dir():
    d = get_default_data_dir()
    files = os.listdir(d)
    assert "Pipeline4FinalDataset.py" in files
    assert "dataset.parquet" in files

def test_get_default_dataset_path():
    path = get_default_dataset_path()
    assert os.path.exists(path) or os.path.exists(path+'.gz')

def test_get_parquet_path():
    data_dir = get_default_data_dir()
    dataset_path = os.path.join(data_dir, "dataset.csv")
    assert Path(get_parquet_path(dataset_path)).name == "dataset.parquet"

    for k in range(1, 4+1):
        path = get_parquet_path(dataset_path, k=k)
        assert Path(path).name == f"dataset_kmer_{k}.parquet"

    for b in range(1, 6+1):
        path = get_parquet_path(dataset_path, bits=b)
        assert Path(path).name == f"dataset_bits_{b}.parquet"

    with pytest.raises(ValueError):
        get_parquet_path(dataset_path, bits=1, k=1)

def test_encoding_column_name():
    with pytest.raises(ValueError):
        encoding_column_name(bits=1, k=1)

    for k in range(1, 4 + 1):
        assert encoding_column_name(k=k) == f"kmer_{k}"

    for b in range(1, 6 + 1):
        assert encoding_column_name(bits=b) == f"bits_{b}"
