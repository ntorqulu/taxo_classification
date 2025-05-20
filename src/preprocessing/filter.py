import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from formatter import SequenceFormatter
from cleaner import TaxonomyDataCleaner


# Constants
TAXONOMY_RANKS = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

class TaxonomyDataFilter:
    """Class for filtering taxonomic data based on various criteria."""
    
    def __init__(self, 
                 data_dir: str = 'data/processed',
                 filtered_data_dir: str = 'data/filtered',
                 logger=None):
        """
        Initialize the TaxonomyDataFilter with directory paths.
        
        Parameters:
        -----------
        data_dir : str
            Directory for input data files
        filtered_data_dir : str
            Directory for filtered output files
        logger : logging.Logger, optional
            Logger for output messages, creates new one if None
        """
        self.data_dir = data_dir
        self.filtered_data_dir = filtered_data_dir
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.filtered_data_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        
    
    def clean_approximated_names(self, df: pd.DataFrame, remove_approximated: bool = False) -> pd.DataFrame:
        """
        Clean approximated taxonomic names by removing the approximation source or filtering out approximated records.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information
        remove_approximated : bool
            If True, completely remove records with approximated names
            If False, normalize names by removing the approximation source
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cleaned taxonomic names or without approximated records
        """
        self.logger.info(f"{'Removing' if remove_approximated else 'Normalizing'} approximated taxonomic names")
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        if remove_approximated:
            # Create a mask to identify records with approximated names
            has_approximation = pd.Series(False, index=cleaned_df.index)
            
            # Check each taxonomic rank for approximations
            for rank in TAXONOMY_RANKS:
                col = f"{rank}_name"
                if col in cleaned_df.columns:
                    # Mark records with approximated names in this rank
                    is_approximated = cleaned_df[col].str.contains(r' \(from .*\)$', regex=True, na=False)
                    has_approximation = has_approximation | is_approximated
            
            # Keep only records without approximations
            cleaned_df = cleaned_df[~has_approximation].copy()
            self.logger.info(f"Removed {sum(has_approximation)} of {len(df)} records with approximated names")
        else:
            # Just normalize by removing the approximation source
            for rank in TAXONOMY_RANKS:
                col = f"{rank}_name"
                if col in cleaned_df.columns:
                    # Extract the name before " (from ...)"
                    has_approximation = cleaned_df[col].str.contains(r' \(from .*\)$', regex=True, na=False)
                    cleaned_df[col] = cleaned_df[col].str.replace(r' \(from .*\)$', '', regex=True)
                    self.logger.info(f"Normalized {sum(has_approximation)} approximated names in {col}")
        
        return cleaned_df
    
    def create_hierarchical_dataset(self,
                               input_file=None,
                               formatted_df=None,
                               raw_data_params=None,
                               cleaning_params=None,
                               max_seq_length=320,
                               output_file="data/database.csv") -> pd.DataFrame:
        """
        Create a hierarchical dataset for taxonomic classification with 4 nested levels.
        
        This function performs the complete pipeline:
        1. Format raw data (optional)
        2. Clean the formatted data
        3. Filter by sequence length
        4. Create hierarchical classification levels
        
        Parameters:
        -----------
        input_file : str, optional
            Path to formatted CSV file. If None, formatted_df or raw_data_params must be provided.
        formatted_df : pd.DataFrame, optional
            Already formatted dataframe to use instead of loading from file
        raw_data_params : dict, optional
            Parameters for formatting raw data. Should include:
            - perl_script_path: Path to select_region.pl script
            - input_tsv: Path to input TSV file
            - names_dmp: Path to NCBI names.dmp
            - nodes_dmp: Path to NCBI nodes.dmp
        cleaning_params : dict, optional
            Parameters for cleaning data. Default parameters will be used if not provided.
        max_seq_length : int, optional
            Maximum sequence length to include (default: 320)
        output_file : str, optional
            Path to save the final dataset (default: "data/database.csv")
            
        Returns:
        --------
        pd.DataFrame
            The processed hierarchical dataset
        """

        self.logger.info("Starting hierarchical dataset creation pipeline")
        # Step 1: Format raw data if needed
        if formatted_df is None:
            if input_file:
                self.logger.info(f"Loading formatted data from {input_file}")
                formatted_df = pd.read_csv(input_file, dtype=str)
            elif raw_data_params:
                self.logger.info("Formatting raw data")
                formatter = SequenceFormatter()
                
                formatted_df = formatter.format_data(
                    run_coi_selection=True,
                    perl_script_path=raw_data_params.get('perl_script_path'),
                    input_tsv=raw_data_params.get('input_tsv'),
                    names_dmp=raw_data_params.get('names_dmp'),
                    nodes_dmp=raw_data_params.get('nodes_dmp')
                )
                # Save intermediary result
                formatted_df.to_csv("data/processed/formatted_sequences.csv", index=False)
            else:
                raise ValueError("Either input_file, formatted_df, or raw_data_params must be provided")
                
        # Step 2: Clean the formatted data
        self.logger.info("Cleaning the formatted data")
        cleaner = TaxonomyDataCleaner()
        
        clean_params = {
            'min_seq_length': 299,
            'max_n_percent': 0.0,
            'require_complete_ranks_up_to': "species",
            'remove_duplicates': True,
            'filter_nonstandard_bases': True,
            'enforce_taxonomy_consistency': True,
            'filter_gc_outliers': False
        }
        
        # Override default cleaning parameters if provided
        if cleaning_params:
            clean_params.update(cleaning_params)
        
        cleaned_df = cleaner.clean_data(formatted_df, **clean_params)
        
        # Save cleaned data
        output_clean = cleaner.save_cleaned_data(cleaned_df)
        self.logger.info(f"Saved cleaned data to {output_clean}")
        
        # Load the clean data to ensure we're using the saved version
        data = pd.read_csv(output_clean, dtype=str)
        
        # clean the approximated names
        self.logger.info("Cleaning approximated names")
        data = self.clean_approximated_names(data, remove_approximated=False)
        self.logger.info(f"After cleaning: {len(data)} sequences")
        
        # Step 3: Filter by max sequence length
        self.logger.info(f"Filtering sequences by max length: {max_seq_length}")
        data = data[data['sequence'].str.len() <= max_seq_length]
        self.logger.info(f"After length filtering: {len(data)} sequences")
        
        # Step 4: Create hierarchical levels
        self.logger.info("Creating hierarchical classification levels")
        
        # Level 1: Kingdom-based categories
        level_1 = ['Metazoa', 'Viridiplantae', 'Fungi', 'Other_euk', 'No_euk']
        data['level_1'] = data['kingdom_name']
        data.loc[data['kingdom_name'].str.contains('Bacteria', na=False), 'level_1'] = 'No_euk'
        data.loc[~data['level_1'].isin(level_1), 'level_1'] = 'Other_euk'
        
        # Level 2: Phylum-based categories for Metazoa
        level_2 = ['Arthropoda', 'Chordata', 'Mollusca', 'Annelida', 'Echinodermata', 
                'Platyhelminthes', 'Cnidaria', 'Other_metazoa', 'No_metazoa']
        data['level_2'] = data['phylum_name']
        data.loc[~data['level_1'].str.contains('Metazoa', na=False), 'level_2'] = 'No_metazoa'
        data.loc[~data['level_2'].isin(level_2), 'level_2'] = 'Other_metazoa'
        
        # Level 3: Class-based categories for Arthropoda
        level_3 = ['Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia', 
                'Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 
                'Pycnogonida', 'Other_arthropoda', 'No_arthropoda']
        data['level_3'] = data['class_name']
        data.loc[~data['level_2'].str.contains('Arthropoda', na=False), 'level_3'] = 'No_arthropoda'
        data.loc[~data['level_3'].isin(level_3), 'level_3'] = 'Other_arthropoda'
        
        # Level 4: Order-based categories for Insecta
        level_4 = ['Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
                'Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea', 
                'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
                'Other_insecta', 'No_insecta']
        data['level_4'] = data['order_name']
        data.loc[~data['level_3'].str.contains('Insecta', na=False), 'level_4'] = 'No_insecta'
        data.loc[~data['level_4'].isin(level_4), 'level_4'] = 'Other_insecta'
        
        # Save the final dataset
        data.to_csv(output_file, index=False)
        self.logger.info(f"Saved hierarchical dataset to {output_file}")
        
        # Print summary statistics
        self.logger.info(f"Final dataset summary:")
        self.logger.info(f"  Total sequences: {len(data)}")
        self.logger.info(f"  Level 1 categories: {data['level_1'].nunique()} ({data['level_1'].value_counts().to_dict()})")
        self.logger.info(f"  Level 2 categories: {data['level_2'].nunique()} (top 5: {data['level_2'].value_counts().head().to_dict()})")
        self.logger.info(f"  Level 3 categories: {data['level_3'].nunique()} (top 5: {data['level_3'].value_counts().head().to_dict()})")
        self.logger.info(f"  Level 4 categories: {data['level_4'].nunique()} (top 5: {data['level_4'].value_counts().head().to_dict()})")
        
        # delete columns superkingdom_name, kingdom_name, phylum_name, class_name, order_name, family_name, genus_name, species_name
        data.drop(columns=['superkingdom_name', 'kingdom_name', 'phylum_name', 'class_name', 
                           'order_name', 'family_name', 'genus_name', 'species_name'], inplace=True)
        # Rename columns to match the hierarchical levels
        data.rename(columns={
            'level_1': 'kingdom_name',
            'level_2': 'phylum_name',
            'level_3': 'class_name',
            'level_4': 'order_name'
        }, inplace=True)
        
        # Save the final dataset with hierarchical levels
        data.to_csv(output_file, index=False)
        self.logger.info(f"Saved hierarchical dataset with levels to {output_file}")
        
        return data
    

# Example usage
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create filter instance
    filter = TaxonomyDataFilter()

    # Option 1: Starting from raw data
    raw_params = {
        'perl_script_path': "src/preprocessing/select_region.pl",
        'input_tsv': "data/raw/NJORDR_sequences.tsv",
        'names_dmp': "data/raw/names.dmp",
        'nodes_dmp': "data/raw/nodes.dmp"
    }
    hierarchical_dataset = filter.create_hierarchical_dataset(raw_data_params=raw_params)
    
    # store the cleaned data
    hierarchical_dataset.to_csv("data/processed/hierarchical_dataset_cleaned.csv", index=False)
    
        