import os
import gc
from xml.sax import default_parser_list

import pandas as pd
from pathlib import Path
from dataset.utils import info, warn, get_parquet_file_path

class CachedDataFrame:
    _df: pd.DataFrame | None = None
    _parquet_file_path: Path | None = None
    _k_encodings: dict[int, pd.DataFrame] = {}
    _bits_encodings: dict[int, pd.DataFrame] = {}

    SEQUENCE_COLUMN_NAME = 'sequence'
    LEVEL_COLUMN_NAME_SUFIX = '_name'

    @classmethod
    def flush_encodings_cache(cls):
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
    def flush_cache(cls):
        if cls._df is not None:
            del cls._df
            gc.collect()
        cls._df: pd.DataFrame = None
        cls._parquet_file_path = None
        cls.flush_encodings_cache()

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
    def _get_main_df(cls, parquet_file_path: Path) -> pd.DataFrame:
        if not cls._is_main_cached():
            if not os.path.exists(parquet_file_path):
                raise FileNotFoundError(f"File '{parquet_file_path}' does not exist. "
                                        "Please build the Parquet files first using the  ParquetBuilder class")
            cls._df = pd.read_parquet(parquet_file_path)
            cls._parquet_file_path = parquet_file_path
            info(f"Level column names: {', '.join(cls.get_level_column_names())}")
        elif cls._parquet_file_path != parquet_file_path:
            raise RuntimeError(f"Cached path differs on provided: {parquet_file_path}")
        return cls._df

    @classmethod
    def _get_encodings_df(cls, parquet_file_path: Path, k: int, bits: int) -> pd.DataFrame:
        if not cls._is_encoding_cached(k, bits):
            assert os.path.exists(parquet_file_path), f"{parquet_file_path} does not exist"
            df = pd.read_parquet(parquet_file_path)
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
    def get_data_frame(cls, parquets_path: Path, k: int = None, bits: int = None) -> pd.DataFrame:
        parquet_file_path = get_parquet_file_path(parquets_path=parquets_path, k=k, bits=bits)
        if k is None and bits is None:
            df = cls._get_main_df(parquet_file_path)
        else:
            df = cls._get_encodings_df(parquet_file_path, k, bits)
        assert df is not None
        return df

    @classmethod
    def get_level_column_names(cls) -> list[str]:
        assert CachedDataFrame._df is not None
        lcn = [c for c in CachedDataFrame._df.columns if c.endswith(CachedDataFrame.LEVEL_COLUMN_NAME_SUFIX)]
        return lcn

    @classmethod
    def get_length(cls) -> int:
        return len(cls._df)

    @classmethod
    def get_min_sequence_len(cls) -> int:
        return cls._df[cls.SEQUENCE_COLUMN_NAME].astype(str).str.len().min()

    @classmethod
    def get_max_sequence_len(cls) -> int:
        return cls._df[cls.SEQUENCE_COLUMN_NAME].astype(str).str.len().max()

