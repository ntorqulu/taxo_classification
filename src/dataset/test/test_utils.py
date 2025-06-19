import pytest
from pathlib import Path
from dataset.utils import *


def test_get_default_data_path():
    path = get_default_data_path()
    path.exists()

def test_get_base_parquets_path():
    path = get_base_parquets_path()
    assert path.exists()

def test_get_parquet_file_path():
    base_parquet_path = get_base_parquets_path()
    dirs = [d for d in base_parquet_path.iterdir() if d.is_dir()]

    for parquets_path in dirs:
        for k in range(1, 5+1):
            path = get_parquet_file_path(parquets_path=parquets_path, k=k)
            assert Path(path).name.endswith(f"_kmer_{k}.parquet"), Path(path).name

        for b in range(1, 4+1):
            path = get_parquet_file_path(parquets_path=parquets_path, bits=b)
            assert Path(path).name.endswith(f"_bits_{b}.parquet"), Path(path).name

        path = get_parquet_file_path(parquets_path=parquets_path, bits=0)
        assert  Path(path).name.endswith("_4rowmatrix.parquet"), Path(path).name

        with pytest.raises(ValueError):
            get_parquet_file_path(parquets_path=parquets_path, bits=1, k=1)

def test_encoding_column_name():
    with pytest.raises(ValueError):
        encoding_column_name(bits=1, k=1)

    for k in range(1, 4 + 1):
        assert encoding_column_name(k=k) == f"kmer_{k}"

    for b in range(1, 6 + 1):
        assert encoding_column_name(bits=b) == f"bits_{b}"
