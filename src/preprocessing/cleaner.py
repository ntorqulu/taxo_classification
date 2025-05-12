import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from collections import Counter
import re

# Constants
TAXONOMY_RANKS = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
STANDARD_BASES = set('ATGC')

class TaxonomyDataCleaner:
    """Class for cleaning and filtering DNA sequence data."""
    
    def __init__(self, 
                 processed_data_dir: str = 'data/processed',
                 logger=None):
        """
        Initialize the TaxonomyDataCleaner with directory paths.
        
        Parameters:
        -----------
        processed_data_dir : str
            Directory for processed output files
        logger : logging.Logger, optional
            Logger for output messages, creates new one if None
        """
        self.processed_data_dir = processed_data_dir
        
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
    
    def filter_by_sequence_length(self, df: pd.DataFrame, min_length: int = 100) -> pd.Series:
        """
        Filter sequences based on minimum length.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sequence data
        min_length : int
            Minimum acceptable sequence length
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        self.logger.info(f"Filtering sequences by minimum length: {min_length}")
        mask = df['sequence'].str.len() >= min_length
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after length filter")
        return mask
    
    def filter_by_n_content(self, df: pd.DataFrame, max_n_percent: float = 5.0) -> pd.Series:
        """
        Filter sequences based on N content percentage.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sequence data
        max_n_percent : float
            Maximum acceptable percentage of N bases
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        self.logger.info(f"Filtering sequences by maximum N content: {max_n_percent}%")
        
        # Count N bases in each sequence
        n_counts = df['sequence'].str.count('N')
        seq_lengths = df['sequence'].str.len()
        n_percent = (n_counts / seq_lengths) * 100
        
        mask = n_percent <= max_n_percent
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after N content filter")
        return mask
    
    def filter_nonstandard_bases(self, df: pd.DataFrame) -> pd.Series:
        """
        Filter sequences containing bases other than A, T, G, C, and N.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sequence data
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        self.logger.info("Filtering sequences with non-standard bases")
        
        # Function to check if a sequence contains only standard bases
        def has_only_standard_bases(seq):
            return all(base in STANDARD_BASES.union({'N'}) for base in seq)
        
        mask = df['sequence'].apply(has_only_standard_bases)
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after non-standard base filter")
        return mask
    
    def filter_by_gc_content(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.Series:
        """
        Filter sequences with outlier GC content based on z-score.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sequence data
        z_threshold : float
            Z-score threshold for considering a value as an outlier
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        self.logger.info(f"Filtering sequences with outlier GC content (z-score > {z_threshold})")
        
        # Calculate GC content for each sequence
        gc_content = df['sequence'].apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0)
        
        # Calculate z-scores
        mean_gc = gc_content.mean()
        std_gc = gc_content.std()
        z_scores = np.abs((gc_content - mean_gc) / std_gc) if std_gc > 0 else np.zeros(len(gc_content))
        
        mask = z_scores <= z_threshold
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after GC content filter")
        return mask
    
    def require_complete_ranks(self, df: pd.DataFrame, up_to_rank: Optional[str] = "order") -> pd.Series:
        """
        Filter records that have incomplete taxonomic information up to a certain rank.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information
        up_to_rank : str or None
            Require all taxonomic ranks up to this level to be non-null
            If None, no filtering is applied
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        if up_to_rank is None:
            self.logger.info("Not filtering by taxonomic completeness")
            return pd.Series(True, index=df.index)
        
        self.logger.info(f"Requiring complete taxonomic ranks up to: {up_to_rank}")
        
        # Find the index of the specified rank
        try:
            rank_index = TAXONOMY_RANKS.index(up_to_rank)
        except ValueError:
            self.logger.error(f"Invalid rank: {up_to_rank}. Using 'species' instead.")
            rank_index = TAXONOMY_RANKS.index('species')
        
        # Get the ranks that must be complete
        required_ranks = TAXONOMY_RANKS[:rank_index + 1]
        
        # Check each required rank
        mask = pd.Series(True, index=df.index)
        for rank in required_ranks:
            col = f"{rank}_name"
            mask = mask & df[col].notna()
        
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after taxonomic completeness filter")
        return mask
    
    def remove_duplicates(self, df: pd.DataFrame, method: str = 'sequence') -> pd.Series:
        """
        Identify and filter duplicate records.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sequence data
        method : str
            Method for identifying duplicates: 'sequence', 'seqID', or 'both'
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        self.logger.info(f"Removing duplicates using method: {method}")
        
        if method == 'sequence':
            # Keep first occurrence of each sequence
            mask = ~df.duplicated(subset=['sequence'])
        elif method == 'seqID':
            # Keep first occurrence of each sequence ID
            mask = ~df.duplicated(subset=['seqID'])
        elif method == 'both':
            # Keep first occurrence of each unique sequence and sequence ID combination
            mask = ~df.duplicated(subset=['seqID', 'sequence'])
        else:
            self.logger.error(f"Invalid duplicate removal method: {method}. Using 'sequence'.")
            mask = ~df.duplicated(subset=['sequence'])
        
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after duplicate removal")
        return mask
    
    def check_taxonomy_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Check for taxonomic inconsistencies (same sequence with different taxonomic assignments).
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information
            
        Returns:
        --------
        Tuple[pandas.DataFrame, Dict]
            Tuple of (dataframe with consistency flags, inconsistency statistics)
        """
        self.logger.info("Checking taxonomic consistency")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add a column to track inconsistency
        result_df['taxonomy_consistent'] = True
        
        # Group by sequence and check if any taxonomic assignments differ
        seq_groups = df.groupby('sequence')
        
        inconsistent_seqs = 0
        inconsistency_stats = {rank: 0 for rank in TAXONOMY_RANKS}
        
        for seq, group in seq_groups:
            if len(group) > 1:  # More than one record with this sequence
                # Check each taxonomic rank for inconsistencies
                for rank in TAXONOMY_RANKS:
                    col = f"{rank}_name"
                    values = group[col].dropna().unique()
                    
                    if len(values) > 1:  # Inconsistent values for this rank
                        inconsistency_stats[rank] += 1
                        
                        # Mark all records with this sequence as inconsistent
                        result_df.loc[group.index, 'taxonomy_consistent'] = False
                        
                        # Log the inconsistency
                        self.logger.debug(f"Inconsistent {rank} for sequence {seq[:20]}...: {values}")
                
                if not result_df.loc[group.index, 'taxonomy_consistent'].all():
                    inconsistent_seqs += 1
        
        self.logger.info(f"Found {inconsistent_seqs} sequences with taxonomic inconsistencies")
        for rank, count in inconsistency_stats.items():
            if count > 0:
                self.logger.info(f"  {rank}: {count} inconsistencies")
        
        return result_df, inconsistency_stats
    
    def filter_inconsistent_taxonomy(self, df: pd.DataFrame) -> pd.Series:
        """
        Filter records with inconsistent taxonomic assignments.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information and consistency flags
            
        Returns:
        --------
        pandas.Series
            Boolean mask of records to keep
        """
        if 'taxonomy_consistent' not in df.columns:
            self.logger.warning("No taxonomy consistency information found. Run check_taxonomy_consistency first.")
            return pd.Series(True, index=df.index)
        
        self.logger.info("Filtering records with inconsistent taxonomic assignments")
        
        mask = df['taxonomy_consistent']
        
        self.logger.info(f"Keeping {mask.sum()} of {len(df)} sequences after inconsistency filter")
        return mask
    
    def clean_data(
        self,
        df: pd.DataFrame,
        min_seq_length: int = 100,
        max_n_percent: float = 5.0,
        require_complete_ranks_up_to: Optional[str] = "order",
        remove_duplicates: bool = True,
        filter_nonstandard_bases: bool = True,
        enforce_taxonomy_consistency: bool = True,
        filter_gc_outliers: bool = True,
        return_mask_only: bool = False,
        verbose: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean taxonomy dataset with configurable parameters.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The original taxonomy dataframe
        min_seq_length : int
            Minimum acceptable sequence length
        max_n_percent : float
            Maximum acceptable percentage of N bases
        require_complete_ranks_up_to : str or None
            Require all taxonomic ranks up to this level to be non-null
            Options: "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species", None
        remove_duplicates : bool
            Whether to remove duplicate sequences
        filter_nonstandard_bases : bool
            Whether to remove sequences with bases other than ATGCN
        enforce_taxonomy_consistency : bool
            Whether to check and filter records with inconsistent taxonomy
        filter_gc_outliers : bool
            Whether to remove sequences with outlier GC content
        return_mask_only : bool
            If True, return only the boolean mask of records to keep
        verbose : bool
            Whether to print details about the cleaning process
            
        Returns:
        --------
        pandas.DataFrame or pandas.Series
            If return_mask_only is False (default), returns cleaned dataframe.
            If return_mask_only is True, returns a boolean Series indicating which rows to keep.
        """
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
            
        self.logger.info(f"Starting data cleaning process on {len(df)} records")
        
        # Make a copy of the original dataframe
        working_df = df.copy()
        
        # Initial mask - keep everything
        mask = pd.Series(True, index=df.index)
        
        # 1. Filter by sequence length
        if min_seq_length > 0:
            length_mask = self.filter_by_sequence_length(working_df, min_seq_length)
            mask = mask & length_mask
        
        # 2. Filter by N content
        if max_n_percent < 100:
            n_mask = self.filter_by_n_content(working_df, max_n_percent)
            mask = mask & n_mask
        
        # 3. Filter non-standard bases
        if filter_nonstandard_bases:
            bases_mask = self.filter_nonstandard_bases(working_df)
            mask = mask & bases_mask
        
        # 4. Filter by GC content
        if filter_gc_outliers:
            gc_mask = self.filter_by_gc_content(working_df)
            mask = mask & gc_mask
        
        # Apply filters so far to reduce dataset size for subsequent operations
        working_df = working_df[mask].copy()
        
        # 6. Check taxonomic consistency
        if enforce_taxonomy_consistency:
            working_df, inconsistency_stats = self.check_taxonomy_consistency(working_df)
            consistency_mask = self.filter_inconsistent_taxonomy(working_df)
            # Update the mask
            temp_mask = pd.Series(False, index=df.index)
            temp_mask.loc[working_df.index] = consistency_mask
            mask = mask & temp_mask
            # Update working dataframe
            working_df = working_df[consistency_mask].copy()
        
        # 7. Require complete ranks
        if require_complete_ranks_up_to is not None:
            ranks_mask = self.require_complete_ranks(working_df, require_complete_ranks_up_to)
            # Update the mask
            temp_mask = pd.Series(False, index=df.index)
            temp_mask.loc[working_df.index] = ranks_mask
            mask = mask & temp_mask
            # Update working dataframe
            working_df = working_df[ranks_mask].copy()
        
        # 9. Remove duplicates
        if remove_duplicates:
            dup_mask = self.remove_duplicates(working_df)
            # Update the mask
            temp_mask = pd.Series(False, index=df.index)
            temp_mask.loc[working_df.index] = dup_mask
            mask = mask & temp_mask
            # Update working dataframe
            working_df = working_df[dup_mask].copy()
        
        # Log summary
        self.logger.info(f"Cleaning complete: kept {mask.sum()} of {len(df)} original records ({mask.sum()/len(df):.2%})")
        
        if return_mask_only:
            return mask
        else:
            return df[mask].copy()
    
    def save_cleaned_data(self, df: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        Save cleaned data to CSV file.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Cleaned DataFrame to save
        output_file : str, optional
            Path for saving the cleaned data, if None uses default path
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if output_file is None:
            output_file = os.path.join(self.processed_data_dir, "cleaned_sequences.csv")
            
        self.logger.info(f"Saving cleaned data to {output_file}")
        df.to_csv(output_file, index=False)
        
        return output_file


# Example usage
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create cleaner instance
    cleaner = TaxonomyDataCleaner()
    
    # Example input file
    input_file = "data/processed/formatted_sequences.csv"
    
    try:
        # Load the formatted data
        df = pd.read_csv(input_file, dtype=str)
        
        # Clean the data
        cleaned_df = cleaner.clean_data(
            df,
            min_seq_length=299,
            max_n_percent=0.0,
            require_complete_ranks_up_to="species",
            remove_duplicates=True,
            filter_nonstandard_bases=True,
            enforce_taxonomy_consistency=True,
            filter_gc_outliers=True
        )
        
        # Save the cleaned data
        output_file = cleaner.save_cleaned_data(cleaned_df)
        
        print(f"Cleaned data saved to {output_file}")
        print(f"Original records: {len(df)}, Cleaned records: {len(cleaned_df)}")
        
    except Exception as e:
        print(f"Error during cleaning: {e}")
        sys.exit(1)