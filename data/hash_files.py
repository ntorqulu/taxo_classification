import hashlib
import os
from dataset.utils import get_default_data_dir

def parquet_hashes(path: str) -> list[list[str]]:
    results = list()

    file_names = [f'dataset_kmer_{k}.parquet' for k in range(1, 4+1)]
    file_names += [f'dataset_bits_{b}.parquet' for b in range(1, 4+1)]
    file_names.append('dataset_4rowmatrix.parquet')

    for file_name in file_names:
        sha256 = hashlib.sha256()
        file_path = os.path.join(path, file_name)
        with open(file_path, "rb") as f:
            for piece in iter(lambda: f.read(4096), b""):
                sha256.update(piece)
        h = sha256.hexdigest()
        results.append([file_name, h])

    return results


if __name__ == "__main__":
    [print(file_name, hash_) for file_name, hash_ in parquet_hashes(get_default_data_dir())]