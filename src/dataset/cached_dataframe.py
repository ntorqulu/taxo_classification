import os
import gc
import pandas as pd
from dataset.utils import info, warn, get_parquet_path

class CachedDataFrame:
    _df: pd.DataFrame | None = None
    _parquet_path: str | None = None
    _k_encodings: dict[int, pd.DataFrame] = {}
    _bits_encodings: dict[int, pd.DataFrame] = {}

    @classmethod
    def flush_cache(cls):
        if cls._df is not None:
            del cls._df
            gc.collect()
        cls._df: pd.DataFrame = None
        cls._parquet_path = None
        for k in cls._k_encodings:
            df = cls._k_encodings[k]
            del df
            gc.collect()
        cls._k_encodings = {}
        for b in cls._bits_encodings:
            df = cls._bits_encodings[b]
            del df
            gc.collect()
        cls._bits_encodings = {}

    @classmethod
    def _is_main_cached(cls) -> bool:
        return cls._df is not None

    @classmethod
    def _is_encoding_cached(cls, k: int = None, bits: int = None) -> bool:
        if k is None and bits is None:
            return False
        if k is None and bits is not None:
            return bits in cls._bits_encodings
        if k is not None and bits is None:
            return k in cls._k_encodings
        assert k is not None and bits is not None
        raise ValueError("K and bits cannot be specified at the same time")

    @classmethod
    def _get_main_df(cls, parquet_path: str) -> pd.DataFrame:
        if not cls._is_main_cached():
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"File '{parquet_path}' does not exist. "
                                        "Please build the Parquet files first using the  ParquetBuilder class")
            cls._df = pd.read_parquet(parquet_path)
            cls._parquet_path = parquet_path
        elif cls._parquet_path != parquet_path:
            raise RuntimeError(f"Cached path differs on provided: {parquet_path}")
        return cls._df

    @classmethod
    def _get_encodings_df(cls, parquet_path: str, k: int, bits: int) -> pd.DataFrame:
        if not cls._is_encoding_cached(k, bits):
            assert os.path.exists(parquet_path), f"{parquet_path} does not exist"
            df = pd.read_parquet(parquet_path)
            if k is not None:
                assert k not in cls._k_encodings, k
                cls._k_encodings[k] = df
            if bits is not None:
                assert bits not in cls._bits_encodings, bits
                cls._bits_encodings[bits] = df
        if k is not None:
            return cls._k_encodings[k]
        if bits is not None:
            return cls._bits_encodings[bits]
        raise RuntimeError("Internal error. We shouldn't be here.")

    @classmethod
    def get_data_frame(cls, csv_path: str, k: int = None, bits: int = None) -> pd.DataFrame:
        parquet_path = get_parquet_path(csv_path, k=k, bits=bits)
        if k is None and bits is None:
            df = cls._get_main_df(parquet_path)
        else:
            df = cls._get_encodings_df(parquet_path, k, bits)
        assert df is not None
        return df
