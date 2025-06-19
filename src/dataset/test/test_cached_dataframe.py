from dataset.cached_dataframe import CachedDataFrame
from dataset.utils import info, get_base_parquets_path, encoding_column_name, DEFAULT_DATASET_NAME
import gc

PARQUETS_PATH = get_base_parquets_path() / DEFAULT_DATASET_NAME

def test_is_main_cached():
    CachedDataFrame.flush_cache()
    assert not CachedDataFrame._is_main_cached()
    CachedDataFrame.get_data_frame(parquets_path=PARQUETS_PATH)
    assert CachedDataFrame._is_main_cached()
    CachedDataFrame.flush_cache()
    assert not CachedDataFrame._is_main_cached()

def test_is_encoding_cached_k():
    CachedDataFrame.flush_cache()
    assert not CachedDataFrame._is_main_cached()

    for k in range(1, 5+1): # TODO: Setting k = 6 is not working; it runs out of memory.
        print(f"{k=}")
        CachedDataFrame.flush_cache()
        df = CachedDataFrame.get_data_frame(parquets_path=PARQUETS_PATH, k=k)
        assert CachedDataFrame._is_encoding_cached(k=k)
        assert df.columns == [encoding_column_name(k=k)], df.columns
        del df
        gc.collect()

def test_is_encoding_cached_bits():
    CachedDataFrame.flush_cache()
    assert not CachedDataFrame._is_main_cached()

    for b in range(1, 4+1):
        print(f"{b=}")
        CachedDataFrame.flush_cache()
        df = CachedDataFrame.get_data_frame(parquets_path=PARQUETS_PATH, bits=b)
        assert CachedDataFrame._is_encoding_cached(bits=b)
        assert df.columns == [encoding_column_name(bits=b)], df.columns
        del df
        gc.collect()

def test_get_data_frame():
    CachedDataFrame.flush_cache()
    assert not CachedDataFrame._is_main_cached()
    df = CachedDataFrame.get_data_frame(parquets_path=PARQUETS_PATH)
    assert df is not None
    CachedDataFrame.flush_cache()
