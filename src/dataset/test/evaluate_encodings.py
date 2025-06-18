import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import time
import matplotlib.pyplot as plt

# Add the src directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset.utils import info, get_parquet_path, encoding_column_name
from dataset.cached_dataframe import CachedDataFrame

class DatasetAnalyzer:
    """Analyzes encodings and class distributions for all taxonomy ranks."""
    
    def __init__(self, csv_path: str, output_dir: str = "analysis_results"):
        """
        Initialize the dataset analyzer.
        
        Args:
            csv_path: Path to the CSV file with sequence data
            output_dir: Directory to save analysis results
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the main dataframe
        info(f"Loading data from {csv_path}")
        self.df = CachedDataFrame.get_data_frame(csv_path)
        info(f"Loaded {len(self.df)} samples")
        
        # Detect taxonomic rank columns
        self.rank_columns = self._detect_rank_columns()
        info(f"Detected {len(self.rank_columns)} taxonomic rank columns: {self.rank_columns}")
        
        # Check if parquet files exist
        self.parquet_path = get_parquet_path(csv_path)
        if not os.path.exists(self.parquet_path):
            info(f"Warning: Main parquet file not found at {self.parquet_path}")
    
    def _detect_rank_columns(self):
        """Automatically detect taxonomic rank columns."""
        # Standard rank columns
        standard_ranks = [
            "superkingdom_name", "kingdom_name", "phylum_name", 
            "class_name", "order_name", "family_name", 
            "genus_name", "species_name"
        ]
        
        # Find all columns ending with "_name" that might be taxonomic ranks
        rank_cols = [col for col in self.df.columns if col.endswith("_name") and col != "scientific_name"]
        
        # Make sure standard ranks are first if they exist
        detected_ranks = []
        for rank in standard_ranks:
            if rank in rank_cols:
                detected_ranks.append(rank)
        
        # Add any additional rank columns
        for col in rank_cols:
            if col not in detected_ranks:
                detected_ranks.append(col)
        
        return detected_ranks
    
    def analyze_all_class_distributions(self):
        """Analyze the distribution of classes for all taxonomic ranks."""
        info("Analyzing class distributions for all taxonomic ranks")
        
        # Create report file
        report_path = os.path.join(self.output_dir, "taxonomy_distribution_report.md")
        with open(report_path, "w") as report:
            report.write("# Taxonomic Class Distribution Analysis\n\n")
            report.write(f"Dataset: `{self.csv_path}`  \n")
            report.write(f"Total samples: {len(self.df)}  \n")
            report.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            report.write("## Summary\n\n")
            report.write("| Taxonomic Rank | Unique Classes | Min Samples | Max Samples | Mean Samples | Median Samples |\n")
            report.write("|---------------|---------------|-------------|-------------|--------------|---------------|\n")
            
            # Create a directory for CSV distributions
            csv_dir = os.path.join(self.output_dir, "class_distributions")
            os.makedirs(csv_dir, exist_ok=True)
            
            # Create a directory for plots
            plot_dir = os.path.join(self.output_dir, "distribution_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Process each rank column
            for rank_col in self.rank_columns:
                try:
                    # Get basic column info
                    non_null_values = self.df[rank_col].dropna()
                    non_null_count = len(non_null_values)
                    null_count = len(self.df) - non_null_count
                    
                    # Count classes
                    class_counts = non_null_values.value_counts()
                    
                    # Calculate statistics
                    if len(class_counts) > 0:
                        min_samples = class_counts.min()
                        max_samples = class_counts.max()
                        mean_samples = class_counts.mean()
                        median_samples = class_counts.median()
                    else:
                        min_samples = max_samples = mean_samples = median_samples = 0
                    
                    # Log summary info
                    info(f"{rank_col}: {len(class_counts)} classes, samples per class: min={min_samples}, max={max_samples}, avg={mean_samples:.1f}")
                    
                    # Add to report summary table
                    report.write(f"| {rank_col} | {len(class_counts)} | {min_samples} | {max_samples} | {mean_samples:.1f} | {median_samples:.1f} |\n")
                    
                    # Save distribution to CSV
                    csv_file = os.path.join(csv_dir, f"{rank_col}_distribution.csv")
                    class_counts.to_frame('count').to_csv(csv_file)
                    
                    # Create distribution plots
                    self._create_distribution_plot(class_counts, rank_col, plot_dir)
                    
                    # Add detailed section to report
                    rank_name = rank_col.replace("_name", "").capitalize()
                    report.write(f"\n## {rank_name} Distribution\n\n")
                    report.write(f"- **Total classes**: {len(class_counts)}\n")
                    report.write(f"- **Missing values**: {null_count} ({null_count/len(self.df)*100:.1f}%)\n")
                    report.write(f"- **Samples per class**: Min={min_samples}, Max={max_samples}, Mean={mean_samples:.1f}, Median={median_samples:.1f}\n\n")
                    
                    # Add top and bottom classes
                    report.write("### Most Common Classes\n\n")
                    report.write("| Class | Samples | Percentage |\n")
                    report.write("|-------|---------|------------|\n")
                    
                    for class_name, count in class_counts.head(10).items():
                        percentage = count / non_null_count * 100
                        report.write(f"| {class_name} | {count} | {percentage:.2f}% |\n")
                    
                    report.write("\n### Least Common Classes\n\n")
                    report.write("| Class | Samples | Percentage |\n")
                    report.write("|-------|---------|------------|\n")
                    
                    for class_name, count in class_counts.tail(10).items():
                        percentage = count / non_null_count * 100
                        report.write(f"| {class_name} | {count} | {percentage:.2f}% |\n")
                    
                    # Add link to CSV
                    report.write(f"\nComplete distribution saved to: [`{os.path.basename(csv_file)}`]({os.path.relpath(csv_file, self.output_dir)})\n\n")
                    
                    # Add link to plot
                    plot_file = os.path.join(plot_dir, f"{rank_col}_distribution.png")
                    report.write(f"![{rank_name} Distribution]({os.path.relpath(plot_file, self.output_dir)})\n\n")
                    
                except Exception as e:
                    info(f"Error analyzing {rank_col}: {str(e)}")
                    report.write(f"\n## {rank_col.replace('_name', '').capitalize()} Distribution\n\n")
                    report.write(f"Error analyzing this rank: {str(e)}\n\n")
        
        info(f"Class distribution analysis complete. Report saved to {report_path}")
        return report_path
    
    def _create_distribution_plot(self, class_counts, rank_col, plot_dir):
        """Create distribution plots for the given class counts."""
        try:
            # Create histogram of class frequencies
            plt.figure(figsize=(12, 6))
            
            # Convert to a more readable format for histogram
            samples_per_class = list(class_counts)
            
            plt.hist(samples_per_class, bins=50, alpha=0.7, color='skyblue')
            plt.xlabel('Number of Samples')
            plt.ylabel('Number of Classes')
            plt.title(f'Distribution of Samples per Class for {rank_col}')
            
            # Add vertical lines for statistics
            plt.axvline(x=np.mean(samples_per_class), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(samples_per_class):.1f}')
            plt.axvline(x=np.median(samples_per_class), color='green', linestyle='-', 
                        label=f'Median: {np.median(samples_per_class):.1f}')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{rank_col}_distribution.png"))
            plt.close()
            
            # Create a more detailed view of the top classes
            plt.figure(figsize=(14, 8))
            top_classes = class_counts.head(30)
            top_classes.plot(kind='bar', color='lightblue')
            plt.title(f'Top 30 Most Common {rank_col} Classes')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{rank_col}_top_classes.png"))
            plt.close()
            
        except Exception as e:
            info(f"Error creating plot for {rank_col}: {str(e)}")
    
    def analyze_all_encodings(self):
        """Analyze all encoding types for all rows in the dataset."""
        info("Analyzing encodings for all rows")
        
        # Create report file
        report_path = os.path.join(self.output_dir, "encoding_analysis_report.md")
        with open(report_path, "w") as report:
            report.write("# Encoding Analysis Report\n\n")
            report.write(f"Dataset: `{self.csv_path}`  \n")
            report.write(f"Total samples: {len(self.df)}  \n")
            report.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            report.write("## Summary\n\n")
            report.write("| Encoding Type | Min Length | Max Length | Mean Length | Length Consistency |\n")
            report.write("|---------------|------------|------------|-------------|-------------------|\n")
            
            # Check k-mer encodings
            k_values = [1, 2, 3, 4, 5]
            for k in k_values:
                self._analyze_kmer_encoding(k, report)
            
            # Check bit encodings
            bit_values = [1, 2, 3, 4]
            for bits in bit_values:
                self._analyze_bit_encoding(bits, report)
            
            # Check 4-row encoding
            self._analyze_4row_encoding(report)
        
        info(f"Encoding analysis complete. Report saved to {report_path}")
        return report_path
    
    def _analyze_kmer_encoding(self, k, report_file):
        """Analyze k-mer encoding for a specific k value."""
        info(f"Analyzing {k}-mer encoding")
        
        col_name = encoding_column_name(k=k)
        
        try:
            # Load the encoded data
            encoded_df = CachedDataFrame.get_data_frame(self.csv_path, k=k)
            
            # Check if the column exists
            if col_name not in encoded_df.columns:
                info(f"Error: Column {col_name} not found in the encoded dataframe")
                report_file.write(f"| {k}-mer | N/A | N/A | N/A | Column not found |\n")
                return
            
            # Process all encodings
            lengths = []
            value_ranges = []
            
            for i in range(len(encoded_df)):
                encoding = encoded_df.iloc[i][col_name]
                if encoding is not None:
                    lengths.append(len(encoding))
                    if len(encoding) > 0:
                        value_ranges.append((min(encoding), max(encoding)))
            
            # Calculate statistics
            if lengths:
                min_length = min(lengths)
                max_length = max(lengths)
                mean_length = sum(lengths) / len(lengths)
                
                # Check length consistency
                length_counter = Counter(lengths)
                most_common_length, count = length_counter.most_common(1)[0]
                consistency = (count / len(lengths)) * 100
                
                # Value range analysis
                min_values = [r[0] for r in value_ranges if r]
                max_values = [r[1] for r in value_ranges if r]
                overall_min = min(min_values) if min_values else None
                overall_max = max(max_values) if max_values else None
                
                # Log information
                info(f"{k}-mer encoding:")
                info(f"  - Encoding length: min={min_length}, max={max_length}, avg={mean_length:.1f}")
                info(f"  - Length consistency: {consistency:.1f}% ({count}/{len(lengths)}) have length {most_common_length}")
                info(f"  - Value range: [{overall_min}, {overall_max}]")
                
                # Write to report
                report_file.write(f"| {k}-mer | {min_length} | {max_length} | {mean_length:.1f} | {consistency:.1f}% ({most_common_length}) |\n")
                
                # Add detailed section
                report_file.write(f"\n## {k}-mer Encoding\n\n")
                report_file.write(f"- **Samples processed**: {len(lengths)}\n")
                report_file.write(f"- **Encoding length**: Min={min_length}, Max={max_length}, Mean={mean_length:.1f}\n")
                report_file.write(f"- **Length consistency**: {consistency:.1f}% have length {most_common_length}\n")
                report_file.write(f"- **Value range**: [{overall_min}, {overall_max}]\n\n")
                
                # Add length distribution
                report_file.write("### Length Distribution\n\n")
                report_file.write("| Length | Count | Percentage |\n")
                report_file.write("|--------|-------|------------|\n")
                
                for length, count in length_counter.most_common(10):
                    percentage = (count / len(lengths)) * 100
                    report_file.write(f"| {length} | {count} | {percentage:.2f}% |\n")
                
                # Add example
                if len(encoded_df) > 0:
                    report_file.write("\n### Example Encoding\n\n")
                    first_encoding = encoded_df.iloc[0][col_name]
                    if first_encoding is not None and len(first_encoding) > 0:
                        report_file.write(f"```\n{first_encoding[:20]}{'...' if len(first_encoding) > 20 else ''}\n```\n\n")
            else:
                info(f"{k}-mer encoding: No valid encodings found")
                report_file.write(f"| {k}-mer | N/A | N/A | N/A | No valid encodings |\n")
        
        except Exception as e:
            info(f"Error analyzing {k}-mer encoding: {str(e)}")
            report_file.write(f"| {k}-mer | N/A | N/A | N/A | Error: {str(e)} |\n")
    
    def _analyze_bit_encoding(self, bits, report_file):
        """Analyze bit encoding for a specific bit value."""
        info(f"Analyzing {bits}-bit encoding")
        
        col_name = encoding_column_name(bits=bits)
        
        try:
            # Load the encoded data
            encoded_df = CachedDataFrame.get_data_frame(self.csv_path, bits=bits)
            
            # Check if the column exists
            if col_name not in encoded_df.columns:
                info(f"Error: Column {col_name} not found in the encoded dataframe")
                report_file.write(f"| {bits}-bit | N/A | N/A | N/A | Column not found |\n")
                return
            
            # Process all encodings
            lengths = []
            all_binary = True
            
            for i in range(len(encoded_df)):
                encoding = encoded_df.iloc[i][col_name]
                if encoding is not None:
                    lengths.append(len(encoding))
                    
                    # Check if all values are 0 or 1
                    if not all(v in (0, 1) for v in encoding):
                        all_binary = False
            
            # Calculate statistics
            if lengths:
                min_length = min(lengths)
                max_length = max(lengths)
                mean_length = sum(lengths) / len(lengths)
                
                # Check length consistency
                length_counter = Counter(lengths)
                most_common_length, count = length_counter.most_common(1)[0]
                consistency = (count / len(lengths)) * 100
                
                # Log information
                info(f"{bits}-bit encoding:")
                info(f"  - Encoding length: min={min_length}, max={max_length}, avg={mean_length:.1f}")
                info(f"  - Length consistency: {consistency:.1f}% ({count}/{len(lengths)}) have length {most_common_length}")
                info(f"  - All values are binary (0 or 1): {all_binary}")
                
                # Write to report
                report_file.write(f"| {bits}-bit | {min_length} | {max_length} | {mean_length:.1f} | {consistency:.1f}% ({most_common_length}) |\n")
                
                # Add detailed section
                report_file.write(f"\n## {bits}-bit Encoding\n\n")
                report_file.write(f"- **Samples processed**: {len(lengths)}\n")
                report_file.write(f"- **Encoding length**: Min={min_length}, Max={max_length}, Mean={mean_length:.1f}\n")
                report_file.write(f"- **Length consistency**: {consistency:.1f}% have length {most_common_length}\n")
                report_file.write(f"- **All values are binary (0 or 1)**: {all_binary}\n\n")
                
                # Add length distribution
                report_file.write("### Length Distribution\n\n")
                report_file.write("| Length | Count | Percentage |\n")
                report_file.write("|--------|-------|------------|\n")
                
                for length, count in length_counter.most_common(10):
                    percentage = (count / len(lengths)) * 100
                    report_file.write(f"| {length} | {count} | {percentage:.2f}% |\n")
                
                # Add example
                if len(encoded_df) > 0:
                    report_file.write("\n### Example Encoding\n\n")
                    first_encoding = encoded_df.iloc[0][col_name]
                    if first_encoding is not None and len(first_encoding) > 0:
                        report_file.write(f"```\n{first_encoding[:20]}{'...' if len(first_encoding) > 20 else ''}\n```\n\n")
            else:
                info(f"{bits}-bit encoding: No valid encodings found")
                report_file.write(f"| {bits}-bit | N/A | N/A | N/A | No valid encodings |\n")
        
        except Exception as e:
            info(f"Error analyzing {bits}-bit encoding: {str(e)}")
            report_file.write(f"| {bits}-bit | N/A | N/A | N/A | Error: {str(e)} |\n")
    
    def _analyze_4row_encoding(self, report_file):
        """Analyze 4-row matrix encoding."""
        info("Analyzing 4-row matrix encoding")
        
        try:
            # Load the encoded data
            encoded_df = CachedDataFrame.get_data_frame(self.csv_path, bits=0)
            
            # Check if all 4 rows exist
            row_columns = ["4row_1", "4row_2", "4row_3", "4row_4"]
            missing_columns = [col for col in row_columns if col not in encoded_df.columns]
            
            if missing_columns:
                info(f"Error: Missing columns: {missing_columns}")
                report_file.write("| 4-row matrix | N/A | N/A | N/A | Missing columns |\n")
                return
            
            # Process all matrices
            consistent_rows = 0
            all_binary = True
            matrix_widths = []
            
            for i in range(len(encoded_df)):
                row1 = encoded_df.iloc[i]["4row_1"]
                row2 = encoded_df.iloc[i]["4row_2"]
                row3 = encoded_df.iloc[i]["4row_3"]
                row4 = encoded_df.iloc[i]["4row_4"]
                
                # Check if all rows have the same length
                if len(row1) == len(row2) == len(row3) == len(row4):
                    consistent_rows += 1
                    matrix_widths.append(len(row1))
                
                # Check if all values are 0 or 1
                for row in [row1, row2, row3, row4]:
                    if not all(v in (0, 1) for v in row):
                        all_binary = False
            
            # Calculate statistics
            if matrix_widths:
                min_width = min(matrix_widths)
                max_width = max(matrix_widths)
                mean_width = sum(matrix_widths) / len(matrix_widths)
                
                # Check width consistency
                width_counter = Counter(matrix_widths)
                most_common_width, count = width_counter.most_common(1)[0]
                consistency = (count / len(matrix_widths)) * 100
                
                # Log information
                info("4-row matrix encoding:")
                info(f"  - Consistent row lengths: {consistent_rows} out of {len(encoded_df)} ({consistent_rows/len(encoded_df)*100:.1f}%)")
                info(f"  - Matrix width: min={min_width}, max={max_width}, avg={mean_width:.1f}")
                info(f"  - Width consistency: {consistency:.1f}% ({count}/{len(matrix_widths)}) have width {most_common_width}")
                info(f"  - All values are binary (0 or 1): {all_binary}")
                
                # Write to report
                report_file.write(f"| 4-row matrix | {min_width} | {max_width} | {mean_width:.1f} | {consistency:.1f}% ({most_common_width}) |\n")
                
                # Add detailed section
                report_file.write("\n## 4-row Matrix Encoding\n\n")
                report_file.write(f"- **Samples processed**: {len(encoded_df)}\n")
                report_file.write(f"- **Consistent row lengths**: {consistent_rows} out of {len(encoded_df)} ({consistent_rows/len(encoded_df)*100:.1f}%)\n")
                report_file.write(f"- **Matrix width**: Min={min_width}, Max={max_width}, Mean={mean_width:.1f}\n")
                report_file.write(f"- **Width consistency**: {consistency:.1f}% have width {most_common_width}\n")
                report_file.write(f"- **All values are binary (0 or 1)**: {all_binary}\n\n")
                
                # Add width distribution
                report_file.write("### Width Distribution\n\n")
                report_file.write("| Width | Count | Percentage |\n")
                report_file.write("|-------|-------|------------|\n")
                
                for width, count in width_counter.most_common(10):
                    percentage = (count / len(matrix_widths)) * 100
                    report_file.write(f"| {width} | {count} | {percentage:.2f}% |\n")
                
                # Add example
                if len(encoded_df) > 0:
                    report_file.write("\n### Example Matrix\n\n")
                    report_file.write("First 10 positions of first sample:\n\n")
                    report_file.write("```\n")
                    for j, row_name in enumerate(row_columns):
                        row = encoded_df.iloc[0][row_name]
                        if row is not None and len(row) > 0:
                            report_file.write(f"Row {j+1}: {row[:10]}{'...' if len(row) > 10 else ''}\n")
                    report_file.write("```\n\n")
            else:
                info("4-row matrix encoding: No valid encodings found")
                report_file.write("| 4-row matrix | N/A | N/A | N/A | No valid encodings |\n")
        
        except Exception as e:
            info(f"Error analyzing 4-row matrix encoding: {str(e)}")
            report_file.write(f"| 4-row matrix | N/A | N/A | N/A | Error: {str(e)} |\n")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze encodings and class distributions')
    parser.add_argument('--csv_path', type=str, default=None, 
                        help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default="analysis_results",
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # If no csv_path is provided, try to find it in standard locations
    if args.csv_path is None:
        potential_paths = [
            "data/dataset.csv",
            "data/database.csv",
            "data/processed/dataset.csv",
            "data/processed/database.csv",
            "data/processed/hierarchical_dataset.csv"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                args.csv_path = path
                break
            elif os.path.exists(path + ".gz"):
                args.csv_path = path + ".gz"
                break
        
        if args.csv_path is None:
            print("Error: Could not find CSV file. Please specify --csv_path")
            return 1
    
    # Create analyzer and run analysis
    try:
        analyzer = DatasetAnalyzer(args.csv_path, args.output_dir)
        
        info("===== ANALYZING CLASS DISTRIBUTIONS =====")
        class_report = analyzer.analyze_all_class_distributions()
        
        info("\n===== ANALYZING ENCODINGS =====")
        encoding_report = analyzer.analyze_all_encodings()
        
        info("\nAll analyses completed!")
        info(f"Class distribution report: {class_report}")
        info(f"Encoding analysis report: {encoding_report}")
        
        return 0
    
    except Exception as e:
        info(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())