#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.utils import info, get_default_data_dir, get_parquet_path

def check_bit_encoding_lengths(bits=None, data_dir=None):
    """
    Check if the bit encoding parquet files have consistent vector lengths.
    
    Args:
        bits: Specific bit encoding to check (1-4), or None to check all
        data_dir: Directory containing the parquet files
    """
    data_dir = data_dir or get_default_data_dir()
    csv_path = os.path.join(data_dir, "dataset.csv")
    
    # Get bit encodings to check
    bit_values = [bits] if bits is not None else range(1, 5)
    
    for bits in bit_values:
        path = get_parquet_path(csv_path, bits=bits)
        
        if not os.path.exists(path):
            info(f"❌ {os.path.basename(path)} does not exist")
            continue
            
        info(f"Checking {os.path.basename(path)}...")
        
        # Load the parquet file
        try:
            df = pd.read_parquet(path)
            col_name = f"bits_{bits}"
            
            if col_name not in df.columns:
                info(f"❌ Column '{col_name}' not found in {os.path.basename(path)}")
                continue
                
            # Check vector lengths
            lengths = df[col_name].apply(len)
            unique_lengths = set(lengths)
            
            info(f"Vector length statistics:")
            info(f"  Min: {lengths.min()}")
            info(f"  Max: {lengths.max()}")
            info(f"  Mean: {lengths.mean():.1f}")
            
            # Count occurrences of each length
            length_counts = Counter(lengths)
            most_common_length = length_counts.most_common(1)[0][0]
            
            info(f"  Most common length: {most_common_length} (appears {length_counts[most_common_length]} times)")
            info(f"  Number of unique lengths: {len(unique_lengths)}")
            
            if len(unique_lengths) == 1:
                info(f"✅ All vectors have the same length: {next(iter(unique_lengths))}")
            else:
                info(f"❌ Vectors have different lengths: {sorted(unique_lengths)}")
                
                # Show examples of sequences with different lengths
                for length in sorted(unique_lengths):
                    count = length_counts[length]
                    example_idx = df[lengths == length].index[0]
                    info(f"  Length {length}: {count} vectors (example idx: {example_idx})")
                
                # Show expected length based on bits
                expected_length = 320 * bits
                info(f"  Expected length (320 × {bits}): {expected_length}")
                
                if expected_length in unique_lengths:
                    correct_count = length_counts[expected_length]
                    info(f"  {correct_count} vectors have the correct length ({expected_length})")
                else:
                    info(f"  No vectors have the expected length ({expected_length})")
                
        except Exception as e:
            info(f"❌ Error checking {path}: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Check bit encoding lengths in parquet files')
    parser.add_argument('--bits', type=int, choices=[1, 2, 3, 4], help='Specific bit encoding to check')
    parser.add_argument('--data-dir', type=str, help='Directory containing the parquet files')
    args = parser.parse_args()
    
    # Run the check
    check_bit_encoding_lengths(bits=args.bits, data_dir=args.data_dir)