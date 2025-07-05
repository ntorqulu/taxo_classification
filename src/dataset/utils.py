import logging
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

DEFAULT_DATASET_NAME = "filtered_ranks"
DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_PARQUET_DIR_NAME = "parquets"

def info(s: str):
    logging.info(s)

def warn(s: str):
    logging.warning(s)

def error(s: str):
    logging.error(s)

def get_default_data_path() -> Path:
    path = Path(__file__).resolve().parent.parent.parent / DEFAULT_DATA_DIR_NAME
    assert path.exists(), path
    return path

def get_base_parquets_path() -> Path:
    path = get_default_data_path() / DEFAULT_PARQUET_DIR_NAME
    assert path.exists(), path
    return path

def get_parquet_file_path(parquets_path: Path = get_base_parquets_path(),
                          k: int | None = None,
                          bits: int | None = None  # 0 for 4 row matrix
                          ) -> Path:
    if k is not None and bits is not None:
        raise ValueError("k and bits cannot be indicated at the same time")

    assert parquets_path.exists(), parquets_path

    if k is None and bits is None:
        pattern = 'dataset*.parquet'
        files = list(parquets_path.glob(pattern))
        assert files, f"No files found in {parquets_path} for pattern: {pattern}"
        file = [min(files, key=lambda f: len(f.name))]
    elif k is not None:
        pattern = f'dataset*_kmer_{k}.parquet'
        file = list(parquets_path.glob(pattern))
    elif bits == 0:
        pattern = f'dataset*_4rowmatrix.parquet'
        file = list(parquets_path.glob(pattern))
    else:
        assert bits is not None
        pattern = f'dataset*_bits_{bits}.parquet'
        file = list(parquets_path.glob(pattern))

    assert len(file) == 1, f"Found {len(file)} files matching pattern: {pattern}"

    return file[0]

def encoding_column_name(k: int | None = None, bits: int | None = None) -> str:
    if k is not None and bits is not None:
        raise ValueError("k and bits cannot be indicated at the same time")

    if k is not None:
        return f"kmer_{k}"
    else:
        assert bits is not None
        return f"bits_{bits}"
