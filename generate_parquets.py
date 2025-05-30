#!/usr/bin/env python
import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.dataset.utils import info
from src.dataset.parquet_builder import ParquetBuilder

def main():
    # read the parser arguments
    # there should e one for which if selected, specify which parquet to create alone
    parser = argparse.ArgumentParser(description='Paqruet Builder for taxonomy classification dataset')
    parser.add_argument(
        '--coding',
        type=str,
        default='all',
        help='Select coding type to create parquets. Options: all, kmer, bit, 4row',
        choices=['all', 'kmer', 'bit', '4row'])
    args = parser.parse_args()

    if args.coding == 'all':
        info("Creating all parquets")
        p = ParquetBuilder()
        p.create_parquets(parallelize=False) # With parallelize=False, it takes less than 20 minutes.
        p.show_info_parquets()
    elif args.coding == 'kmer':
        info("Creating kmer parquet")
        p = ParquetBuilder()
        p.create_parquets_unique(parallelize=False, coding='kmer')  # With parallelize=False, it takes less than 20 minutes.
        p.show_info_parquets()
    elif args.coding == 'bit':
        info("Creating bit parquet")
        p = ParquetBuilder()
        p.create_parquets_unique(parallelize=False, coding='bit')
        p.show_info_parquets()
    elif args.coding == '4row':
        info("Creating 4row parquet")
        p = ParquetBuilder()
        p.create_parquets_unique(parallelize=False, coding='4row')
        p.show_info_parquets()
    

if __name__ == "__main__":
    main()