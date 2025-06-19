import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

# Add the src directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset.utils import info

class TestParquetContent(unittest.TestCase):
    """Test class to verify the content of parquet files is correct."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Find parquet files in the project
        self.data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        self.parquet_files = list(self.data_dir.glob("**/*.parquet"))
        
        info(f"Found {len(self.parquet_files)} parquet files for testing")
        
        if not self.parquet_files:
            self.skipTest("No parquet files found for testing")
        
        # Define expected values
        self.expected_min_seq_length = 50  # Minimum expected sequence length
        self.expected_max_seq_length = 5000  # Maximum expected sequence length
        self.dna_pattern = re.compile(r'^[ACGTN-]+$', re.IGNORECASE)  # Pattern for valid DNA sequences
    
    def test_sequence_length_ranges(self):
        """Test that sequences have expected minimum and maximum lengths."""
        for file_path in self.parquet_files:
            info(f"Testing sequence length ranges in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Find sequence column
            seq_col = self._find_sequence_column(df)
            if not seq_col:
                info(f"  No sequence column found in {file_path.name}, skipping")
                continue
            
            # Calculate sequence lengths
            seq_lengths = df[seq_col].str.len()
            min_length = seq_lengths.min()
            max_length = seq_lengths.max()
            mean_length = seq_lengths.mean()
            median_length = seq_lengths.median()
            
            # Log statistics
            info(f"  Sequence lengths - Min: {min_length}, Max: {max_length}, Mean: {mean_length:.1f}, Median: {median_length}")
            
            # Validate against expected ranges
            self.assertGreaterEqual(min_length, 1, f"Minimum sequence length too short: {min_length}")
            self.assertLessEqual(max_length, 100000, f"Maximum sequence length too long: {max_length}")
            
            # Report sequences outside expected ranges
            short_seqs = df[seq_lengths < self.expected_min_seq_length]
            long_seqs = df[seq_lengths > self.expected_max_seq_length]
            
            if len(short_seqs) > 0:
                info(f"  Found {len(short_seqs)} sequences shorter than {self.expected_min_seq_length}")
                if len(short_seqs) < 5:
                    for idx, row in short_seqs.iterrows():
                        info(f"    Short sequence (len={len(row[seq_col])}): {row[seq_col]}")
            
            if len(long_seqs) > 0:
                info(f"  Found {len(long_seqs)} sequences longer than {self.expected_max_seq_length}")
                if len(long_seqs) < 5:
                    for idx, row in long_seqs.iterrows():
                        info(f"    Long sequence (len={len(row[seq_col])}): {row[seq_col][:50]}...")
    
    def test_sequence_content(self):
        """Test that sequences contain only valid DNA characters."""
        for file_path in self.parquet_files:
            info(f"Testing sequence content in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Find sequence column
            seq_col = self._find_sequence_column(df)
            if not seq_col:
                info(f"  No sequence column found in {file_path.name}, skipping")
                continue
            
            # Check if all sequences match the DNA pattern
            invalid_seqs = df[~df[seq_col].str.upper().str.match(self.dna_pattern)]
            
            # Report invalid sequences
            if len(invalid_seqs) > 0:
                info(f"  Found {len(invalid_seqs)} sequences with invalid characters")
                if len(invalid_seqs) < 5:
                    for idx, row in invalid_seqs.iterrows():
                        seq = row[seq_col]
                        invalid_chars = set(seq.upper()) - set('ACGTN-')
                        info(f"    Invalid sequence with characters {invalid_chars}: {seq[:50]}...")
            
            # Assert all sequences are valid
            self.assertEqual(len(invalid_seqs), 0, f"Found {len(invalid_seqs)} sequences with invalid DNA characters")
    
    def test_kmer_encoding_values(self):
        """Test that k-mer encodings have correct values."""
        for file_path in self.parquet_files:
            info(f"Testing k-mer encoding values in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Find k-mer columns
            kmer_cols = [col for col in df.columns if 'kmer' in col.lower()]
            if not kmer_cols:
                info(f"  No k-mer columns found in {file_path.name}, skipping")
                continue
            
            for kmer_col in kmer_cols:
                info(f"  Testing column: {kmer_col}")
                
                # Extract k value from column name
                k_match = re.search(r'(\d+)kmer', kmer_col)
                if not k_match:
                    info(f"    Could not determine k value from column name, skipping")
                    continue
                    
                k = int(k_match.group(1))
                
                # Calculate expected maximum value based on k
                max_expected_value = 4**k  # For DNA, we have 4 possible bases
                
                # Check a sample of rows
                sample_size = min(100, len(df))
                sample = df.sample(sample_size)
                
                for idx, row in sample.iterrows():
                    kmer_encoding = row[kmer_col]
                    
                    # Skip if null
                    if pd.isna(kmer_encoding):
                        continue
                    
                    # Check if values are within expected range
                    if isinstance(kmer_encoding, list):
                        max_value = max(kmer_encoding) if kmer_encoding else 0
                        self.assertLessEqual(max_value, max_expected_value, 
                                           f"K-mer encoding contains value {max_value} > {max_expected_value}")
    
    def test_bit_encoding_values(self):
        """Test that bit encodings have correct values."""
        for file_path in self.parquet_files:
            info(f"Testing bit encoding values in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Find bit columns
            bit_cols = [col for col in df.columns if 'bit' in col.lower()]
            if not bit_cols:
                info(f"  No bit columns found in {file_path.name}, skipping")
                continue
            
            for bit_col in bit_cols:
                info(f"  Testing column: {bit_col}")
                
                # Extract bit value from column name
                bit_match = re.search(r'(\d+)bit', bit_col)
                if not bit_match:
                    info(f"    Could not determine bit value from column name, skipping")
                    continue
                    
                bits = int(bit_match.group(1))
                
                # Check a sample of rows
                sample_size = min(100, len(df))
                sample = df.sample(sample_size)
                
                for idx, row in sample.iterrows():
                    bit_encoding = row[bit_col]
                    
                    # Skip if null
                    if pd.isna(bit_encoding):
                        continue
                    
                    # Check if values are 0 or 1 for bit encodings
                    if isinstance(bit_encoding, list):
                        unique_values = set(bit_encoding)
                        self.assertTrue(unique_values.issubset({0, 1}), 
                                      f"Bit encoding contains values other than 0 or 1: {unique_values}")
    
    def test_taxonomy_hierarchy(self):
        """Test that taxonomy follows proper hierarchy."""
        for file_path in self.parquet_files:
            info(f"Testing taxonomy hierarchy in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Check for taxonomy columns
            taxonomy_levels = ['phylum_name', 'class_name', 'order_name', 'family_name', 'genus_name', 'species_name']
            available_levels = [level for level in taxonomy_levels if level in df.columns]
            
            if len(available_levels) <= 1:
                info(f"  Insufficient taxonomy levels in {file_path.name}, skipping")
                continue
            
            # Check that taxonomy follows proper hierarchy
            # (e.g., same genus should always be in same family)
            for i in range(len(available_levels) - 1):
                higher_level = available_levels[i]
                lower_level = available_levels[i+1]
                
                info(f"  Testing {higher_level} -> {lower_level} hierarchy")
                
                # Group by lower level and check uniqueness of higher level
                lower_to_higher = df.groupby(lower_level)[higher_level].nunique()
                violations = lower_to_higher[lower_to_higher > 1]
                
                if len(violations) > 0:
                    info(f"    Found {len(violations)} taxonomy violations")
                    if len(violations) < 5:
                        for lower, count in violations.items():
                            higher_values = df[df[lower_level] == lower][higher_level].unique()
                            info(f"      {lower_level}='{lower}' has {count} different {higher_level} values: {higher_values}")
                
                # Assert proper hierarchy
                self.assertEqual(len(violations), 0, 
                               f"Found {len(violations)} taxonomy hierarchy violations between {higher_level} and {lower_level}")
    
    def test_class_distribution(self):
        """Test distribution of classes across taxonomy levels."""
        for file_path in self.parquet_files:
            info(f"Testing class distribution in {file_path.name}")
            
            # Load parquet file
            df = pd.read_parquet(file_path)
            
            # Check each taxonomy level
            taxonomy_levels = ['phylum_name', 'class_name', 'order_name', 'family_name', 'genus_name', 'species_name']
            available_levels = [level for level in taxonomy_levels if level in df.columns]
            
            for level in available_levels:
                # Count classes
                class_counts = df[level].value_counts()
                num_classes = len(class_counts)
                min_count = class_counts.min()
                max_count = class_counts.max()
                
                info(f"  {level}: {num_classes} classes, min count: {min_count}, max count: {max_count}")
                
                # List imbalanced classes if severe imbalance exists
                if max_count > 100 * min_count:
                    info(f"    Severe class imbalance detected")
                    rare_classes = class_counts[class_counts < 10].index.tolist()
                    if rare_classes:
                        info(f"    Rare classes (<10 samples): {rare_classes}")
                    
                    dominant_classes = class_counts[class_counts > 1000].index.tolist()
                    if dominant_classes:
                        info(f"    Dominant classes (>1000 samples): {dominant_classes}")
    
    def _find_sequence_column(self, df):
        """Find the column containing sequences."""
        for col in df.columns:
            if col.lower() in ['sequence', 'seq'] and isinstance(df[col].iloc[0], str):
                return col
        
        # Try to find any column that might contain sequences
        for col in df.columns:
            if isinstance(df[col].iloc[0], str):
                sample = df[col].iloc[0].upper()
                if set(sample).issubset(set('ACGTN-')):
                    return col
        
        return None

if __name__ == '__main__':
    unittest.main()