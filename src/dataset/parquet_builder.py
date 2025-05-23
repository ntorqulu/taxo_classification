import os
import pandas as pd
from dataset.utils import get_default_data_dir, info, get_parquet_path, encoding_column_name
from feature_extraction.main import SequenceCoder

class ParquetBuilder:
    CATEGORY_COLUMNS = ["kingdom_name", "phylum_name", "class_name", "order_name"]
    KMERS_SIZES = range(1, 5+1)

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

    def create_dataset_parquet(self, skip_if_exists: bool = True):
        if self.df is None:
            raise RuntimeError("You must load the database first")

        path = get_parquet_path(self.csv_path)
        if not os.path.exists(path) or not skip_if_exists:
            self.df.to_parquet(path, compression="gzip")
        else:
            info(f"Skipping main because exists {path}")


    def create_kmer_parquets(self, skip_if_exists: bool = True):
        if self.df is None:
            raise RuntimeError("You must load the database first")

        for k in ParquetBuilder.KMERS_SIZES:
            col_name = encoding_column_name(k=k)
            path = get_parquet_path(self.csv_path, k=k)
            if os.path.exists(path) and skip_if_exists:
                info(f"Skipping {k=} because exists {path}")
                continue

            info(f"Encoding {col_name}")
            assert_s = self.df[self.sequence_column_name][0]
            assert len(self.sequence_coder.coding_kmer_optimized([assert_s], k)) == 1
            results = [self.sequence_coder.coding_kmer_optimized([s], k)
                       for s in self.df[self.sequence_column_name]]
            assert all(len(r)==1 for r in results)
            results = [r[0] for r in results]

            info(f"Moving results to a dataframe")
            df = pd.DataFrame({col_name: results})

            info(f"Saving '{path}'")
            df.to_parquet(path=path, compression="gzip")
            del df

    def create_bit_parquets(self, skip_if_exists: bool = True):
        if self.df is None:
            raise RuntimeError("You must load the database first")

        for bits in self.sequence_coder.bit_mapping:
            col_name = encoding_column_name(bits=bits)
            path = get_parquet_path(self.csv_path, bits=bits)
            if os.path.exists(path) and skip_if_exists:
                info(f"Skipping {bits=} because exists {path}")
                continue

            info(f"Encoding {col_name}")
            results = [self.sequence_coder.coding_one_hot_bit_optimized([s], bits)
                       for s in self.df[self.sequence_column_name]]
            results = [r[0] for r in results]

            info(f"Moving results to a dataframe")
            df = pd.DataFrame({col_name: results})

            info(f"Saving '{path}'")
            df.to_parquet(path=path, compression="gzip")
            del df

    def create_4row_parquet(self, skip_if_exists: bool = True):
        if self.df is None:
            raise RuntimeError("You must load the database first")

        path = get_parquet_path(self.csv_path, bits=0)
        if os.path.exists(path) and skip_if_exists:
            info(f"Skipping '4 row matrix' because exists {path}")
        else:
            col_name = "4row"
            info(f"Encoding {col_name}")
            s = self.df[self.sequence_column_name][0]
            s = self.sequence_coder.coding_one_hot_4rowMatrix_optimized([s], return_tensor=False)
            results = [self.sequence_coder.coding_one_hot_4rowMatrix_optimized([s], return_tensor=False)
                       for s in self.df[self.sequence_column_name]]

            info(f"Moving results to a dataframe")
            # We do this way to avoid "PerformanceWarning: DataFrame is highly fragmented."
            assert all(len(r) == 1 for r in results)
            assert all(r[0].shape[0] == 4 for r in results)
            df = pd.DataFrame({
                f"{col_name}_1": [r[0][0] for r in results],
                f"{col_name}_2": [r[0][1] for r in results],
                f"{col_name}_3": [r[0][2] for r in results],
                f"{col_name}_4": [r[0][3] for r in results],
            })

            info(f"Saving '{path}'")
            df.to_parquet(path=path, compression="gzip")
            del df


    def create_parquets(self, skip_if_exists: bool = True):
        """
        encoder.coding_kmer_optimized(sequences=["ACGTT"], k=3) k 1..5
        encoder.coding_one_hot_4rowMatrix_optimized(sequences=["ACGTT"], return_tensor=True)
        encoder.coding_one_hot_bit_optimized(sequences=["ACGTT"], bits=4, return_tensor=True) 1..4
        """

        self.create_dataset_parquet(skip_if_exists=skip_if_exists)
        self.create_kmer_parquets(skip_if_exists=skip_if_exists)
        self.create_bit_parquets(skip_if_exists=skip_if_exists)
        self.create_4row_parquet(skip_if_exists=skip_if_exists)
        info(f"Parquets created")


    def show_info_parquets(self):
        def print_basic_info(path):
            info(os.path.basename(path))
            df = pd.read_parquet(path=path)
            info(f"   {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
            info(f"   {len(df)} rows")
            info(f"   {[c for c in df.columns]}")
            if len(df.columns) > 1:
                info(str(df.head()))
            else:
                for c in df.columns:
                    info(f"   {c}")
                    info(f"      Min length: {df[c].apply(len).min()}")
                    info(f"      Mean length: {df[c].apply(len).mean()}")
                    info(f"      Max length: {df[c].apply(len).max()}")
                    info(f"      First row: {df[c]}")
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
    p.show_info_parquets()

if __name__ == "__main__":
    __main()
