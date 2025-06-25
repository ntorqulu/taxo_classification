import hashlib
import os
from pathlib import Path
from dataset.utils import get_base_parquets_path

def parquet_hashes(path: Path) -> list[list[str]]:
    results = list()

    for file_path in path.iterdir():
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for piece in iter(lambda: f.read(4096), b""):
                sha256.update(piece)
        h = sha256.hexdigest()
        results.append([path.name + os.sep + file_path.name, h])

    return results


if __name__ == "__main__":
    for d in get_base_parquets_path().iterdir():
        if not d.is_dir():
            continue
        [print(file_name, hash_) for file_name, hash_ in parquet_hashes(d)]
