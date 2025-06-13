#!/usr/bin/env python
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.dataset.utils import info

def analyze_encoding_lengths(file_path, encoding_type, config_value=None):
    """
    Analyze the encoding lengths from a parquet file.
    
    Args:
        file_path: Path to the parquet file
        encoding_type: Type of encoding (kmer, bit, 4row)
        config_value: Configuration value (e.g., 1, 2, 3, etc.)
    """
    # Create a more descriptive encoding label that includes the config value
    encoding_label = f"{encoding_type}_{config_value}" if config_value else encoding_type
    info(f"Analyzing {encoding_label} encoding from {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        info(f"Loaded parquet file with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Extract encoding lengths
        lengths = []
        total_rows = len(df)
        
        # Find appropriate column based on encoding type and config
        if encoding_type == 'kmer':
            # Look for a specific kmer column like "kmer_3"
            if config_value:
                col_name = f"kmer_{config_value}"
                if col_name in df.columns:
                    info(f"Using column {col_name} for {encoding_label} analysis")
                else:
                    # Fall back to any kmer column
                    encoding_cols = [col for col in df.columns if 'kmer' in col.lower()]
                    if encoding_cols:
                        col_name = encoding_cols[0]
                        info(f"Specific column not found, using {col_name} instead")
                    else:
                        info(f"No kmer column found, using first column: {df.columns[0]}")
                        col_name = df.columns[0]
            else:
                # Just find any kmer column
                encoding_cols = [col for col in df.columns if 'kmer' in col.lower()]
                if encoding_cols:
                    col_name = encoding_cols[0]
                    info(f"Using column {col_name} for kmer analysis")
                else:
                    info(f"No kmer column found, using first column: {df.columns[0]}")
                    col_name = df.columns[0]
                    
        elif encoding_type == 'bit':
            # Look for a specific bits column like "bits_2"
            if config_value:
                col_name = f"bits_{config_value}"
                if col_name in df.columns:
                    info(f"Using column {col_name} for {encoding_label} analysis")
                else:
                    # Fall back to any bits column
                    encoding_cols = [col for col in df.columns if 'bits' in col.lower()]
                    if encoding_cols:
                        col_name = encoding_cols[0]
                        info(f"Specific column not found, using {col_name} instead")
                    else:
                        info(f"No bits column found, using first column: {df.columns[0]}")
                        col_name = df.columns[0]
            else:
                # Just find any bits column
                encoding_cols = [col for col in df.columns if 'bits' in col.lower()]
                if encoding_cols:
                    col_name = encoding_cols[0]
                    info(f"Using column {col_name} for bit analysis")
                else:
                    info(f"No bits column found, using first column: {df.columns[0]}")
                    col_name = df.columns[0]
                    
        elif encoding_type == '4row':
            # For 4row matrix, use the first 4row column
            encoding_cols = [col for col in df.columns if '4row' in col.lower()]
            if encoding_cols:
                col_name = encoding_cols[0]
                info(f"Using column {col_name} for 4row analysis")
            else:
                info(f"No 4row column found, using first column: {df.columns[0]}")
                col_name = df.columns[0]
                
        # Extract lengths from the selected column
        for i, row in tqdm(df.iterrows(), total=total_rows, desc=f"Processing {encoding_label}"):
            encoding = row[col_name]
            if isinstance(encoding, np.ndarray):
                if len(encoding.shape) > 1:
                    # 2D array (for 4row matrices)
                    lengths.append(encoding.shape[1])
                else:
                    # 1D array
                    lengths.append(len(encoding))
            elif isinstance(encoding, list):
                lengths.append(len(encoding))
            else:
                info(f"Warning: Unexpected encoding type at row {i}: {type(encoding)}")
                
        # Calculate statistics
        if lengths:
            stats = {
                'encoding_type': encoding_label,
                'count': len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'mean_length': np.mean(lengths),
                'median_length': np.median(lengths),
                'std_length': np.std(lengths),
                'lengths': lengths  # Keep full distribution for plotting
            }
            
            # Print summary info
            info(f"Analysis complete for {encoding_label}:")
            info(f"  - Count: {stats['count']}")
            info(f"  - Min length: {stats['min_length']}")
            info(f"  - Max length: {stats['max_length']}")
            info(f"  - Mean length: {stats['mean_length']:.2f}")
            info(f"  - Median length: {stats['median_length']}")
            info(f"  - Std dev: {stats['std_length']:.2f}")
            
            return stats
        else:
            info(f"No valid encodings found for {encoding_label}")
            return {
                'encoding_type': encoding_label,
                'error': "No valid encodings found",
                'lengths': []
            }
    
    except Exception as e:
        import traceback
        info(f"Error analyzing {encoding_label} encoding: {e}")
        info(traceback.format_exc())
        return {
            'encoding_type': encoding_label,
            'error': str(e),
            'lengths': []
        }

def generate_summary_table(stats_list, output_dir="reports/tables"):
    """Generate a summary table of all encoding statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    
    for stats in stats_list:
        if 'error' in stats:
            row = {
                'Encoding Type': stats['encoding_type'],
                'Error': stats['error'],
                'Count': 0,
                'Min Length': 0,
                'Max Length': 0,
                'Mean Length': 0,
                'Median Length': 0,
                'Std Dev': 0
            }
        else:
            row = {
                'Encoding Type': stats['encoding_type'],
                'Error': 'None',
                'Count': stats['count'],
                'Min Length': stats['min_length'],
                'Max Length': stats['max_length'],
                'Mean Length': round(stats['mean_length'], 2),
                'Median Length': stats['median_length'],
                'Std Dev': round(stats['std_length'], 2)
            }
        
        summary_data.append(row)
    
    # Sort by encoding type
    summary_data.sort(key=lambda x: x['Encoding Type'])
    
    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "encoding_length_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # Also print the table
    info("\nEncoding Length Summary:")
    info(summary_df.to_string(index=False))
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate encoding lengths from parquet files')
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data',
        help='Directory containing parquet files'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        info(f"Error: Data directory {args.data_dir} not found")
        return
    
    # Define the configurations to analyze
    configs = [
        ('kmer', 1),
        ('kmer', 2),
        ('kmer', 3),
        ('kmer', 4),
        ('kmer', 5),
        ('bits', 1),
        ('bits', 2),
        ('bits', 3),
        ('bits', 4),
        ('4rowmatrix', None)
    ]
    
    all_stats = []
    
    for encoding_type, config_value in configs:
        # Create file pattern based on encoding type and config
        if config_value:
            file_pattern = f"dataset_{encoding_type}_{config_value}.parquet"
        else:
            file_pattern = f"dataset_{encoding_type}.parquet"
        
        # Find matching files
        matching_files = []
        
        if args.recursive:
            # Search recursively
            for root, dirs, files in os.walk(args.data_dir):
                for file in files:
                    if file == file_pattern:
                        matching_files.append(Path(root) / file)
        else:
            # Search only in the specified directory
            matching_files = list(Path(args.data_dir).glob(file_pattern))
        
        if not matching_files:
            info(f"Warning: No files found for {encoding_type}_{config_value} with pattern '{file_pattern}'")
            continue
            
        # Analyze the first matching file
        file_path = str(matching_files[0])
        info(f"Found file for {encoding_type}_{config_value if config_value else ''}: {file_path}")
        
        # Use standardized names for analysis
        analysis_type = encoding_type
        if encoding_type == 'bits':
            analysis_type = 'bit'
        elif encoding_type == '4rowmatrix':
            analysis_type = '4row'
            
        stats = analyze_encoding_lengths(file_path, analysis_type, config_value)
        all_stats.append(stats)
    
    # Generate visualizations and summary
    if all_stats:
        csv_path = generate_summary_table(all_stats)
        info(f"Summary table saved to {csv_path}")
    else:
        info("No encoding statistics collected. Check your parquet files.")

if __name__ == "__main__":
    main()