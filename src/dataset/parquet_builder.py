import os
import pandas as pd
from dataset.utils import get_default_data_dir
from utils import info, get_parquet_path, encoding_column_name
from feature_extraction.main import SequenceCoder

class ParquetBuilder:
    CATEGORY_COLUMNS = ["kingdom_name", "phylum_name", "class_name", "order_name"]
    KMERS_SIZES = range(1, 6+1)

    def __init__(self,
                 csv_path: str = get_default_data_dir()+"/dataset.csv",
                 sequence_column_name: str = "sequence"):
        gzip_path: str = csv_path + ".gz"
        if os.path.exists(gzip_path):
            self.csv_path: str = gzip_path
        elif os.path.exists(csv_path):
            self.csv_path: str = csv_path
        else:
            raise FileNotFoundError(f"Neither {gzip_path} nor {csv_path} exists")
        self.sequence_column_name: str = sequence_column_name
        self.df: pd.DataFrame = self._load()
        self.sequence_coder = SequenceCoder()

    def _load(self):
        info(f"Loading csv file: '{self.csv_path}'")
        df = pd.read_csv(
            self.csv_path,
            low_memory=False, # To avoid warning about mixed types
        )
        # We convert it to category to speed up memory
        for column_name in ParquetBuilder.CATEGORY_COLUMNS:
            df[column_name] = df[column_name].astype("category")
        info(f"Loaded {len(df)} rows")
        return df

    def create_parquets(self, skip_if_exists: bool = True):
        if self.df is None:
            raise RuntimeError("You must load before saving")

        self.df.to_parquet(path=get_parquet_path(self.csv_path), compression="gzip")

        for bits in self.sequence_coder.bit_mapping:
            col_name = encoding_column_name(bits=bits)
            path = get_parquet_path(self.csv_path, bits=bits)
            if os.path.exists(path) and skip_if_exists:
                info(f"Skipping {bits=} because exists {path}")
                continue
            info(f"Encoding {col_name}")
            results = [self.sequence_coder.dna_to_bitcoding_optimized(s, bits)
                       for s in self.df[self.sequence_column_name]]
            info(f"Moving results to a dataframe")
            df = pd.DataFrame({col_name: results})
            info(f"Saving '{path}'")
            df.to_parquet(path=path, compression="gzip")
            del df

        for k in ParquetBuilder.KMERS_SIZES:
            col_name = encoding_column_name(k=k)
            path = get_parquet_path(self.csv_path, k=k)
            if os.path.exists(path) and skip_if_exists:
                info(f"Skipping {k=} because exists {path}")
                continue
            info(f"Encoding {col_name}")
            results = [self.sequence_coder.kmerize_one_seq_optimized(s, k)
                       for s in self.df[self.sequence_column_name]]
            info(f"Moving results to a dataframe")
            df = pd.DataFrame({col_name: results})
            info(f"Saving '{path}'")
            df.to_parquet(path=path, compression="gzip" if k <= 6 else None)
            del df

        info(f"Parquets created")

    def info_parquets(self):
        def print_basic_info(path):
            info(os.path.basename(path))
            df = pd.read_parquet(path=path)
            info(f"   {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
            info(f"   {len(df)} rows")
            info(f"   {[c for c in df.columns]}")
            info(str(df.head()))
            del df

        p = get_parquet_path(self.csv_path)
        print_basic_info(p)

        for bits in self.sequence_coder.bit_mapping:
            p = get_parquet_path(self.csv_path, bits=bits)
            print_basic_info(p)

        for k in ParquetBuilder.KMERS_SIZES:
            p = get_parquet_path(self.csv_path, k=k)
            print_basic_info(p)

def __main():
    p = ParquetBuilder()
    p.create_parquets() # Uncomment if any parquet file does not exists
    p.info_parquets()

if __name__ == "__main__":
    __main()
