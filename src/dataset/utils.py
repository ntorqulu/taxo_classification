import logging
import os
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def info(s: str):
    logging.info(s)

def warn(s: str):
    logging.warning(s)

def error(s: str):
    logging.error(s)

def get_default_data_dir() -> str:
    path = Path(__file__).resolve().parent.parent.parent / 'data'
    path = str(path)
    return path

def get_default_dataset_path() -> str:
    path = get_default_data_dir()
    path += "/dataset.csv"
    return path

def get_parquet_path(csv_path: str = get_default_dataset_path(), k: int | None = None, bits: int | None = None) -> str:
    if k is not None and bits is not None:
        raise ValueError("k and bits cannot be indicated at the same time")
    if csv_path.endswith('.csv'):
        extension = '.csv'
    elif csv_path.endswith('.csv.gz'):
        extension = '.csv.gz'
    else:
        raise RuntimeError(f"Unsupported file extension: {csv_path}")
    p = csv_path[:-len(extension)]
    if k is None and bits is None:
        p += '.parquet'
    elif k is not None:
        p += f'_kmer_{k}.parquet'
    else:
        assert bits is not None
        p += f'_bits_{bits}.parquet'
    return p

def encoding_column_name(k: int | None = None, bits: int | None = None) -> str:
    if k is not None and bits is not None:
        raise ValueError("k and bits cannot be indicated at the same time")
    if k is not None:
        return f"kmer_{k}"
    else:
        assert bits is not None
        return f"bits_{bits}"
