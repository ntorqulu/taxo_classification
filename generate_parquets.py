#!/usr/bin/env python
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.dataset.utils import info
from src.dataset.parquet_builder import ParquetBuilder

def main():
    p = ParquetBuilder()
    p.create_parquets(parallelize=False) # With parallelize=False, it takes less than 20 minutes.
    p.show_info_parquets()

if __name__ == "__main__":
    main()