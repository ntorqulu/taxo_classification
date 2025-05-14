import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging


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
    
    def filter_small_classes(self, df: pd.DataFrame, min_count: int = 5, target_column: str = "genus_name") -> pd.DataFrame:
        """
        Filter records belonging to classes with too few examples.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic information
        min_count : int
            Minimum number of examples per class to keep
        target_column : str
            Column to use for class counting
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with classes having sufficient examples
        """
        self.logger.info(f"Filtering classes with fewer than {min_count} examples in {target_column}")
        
        # Count occurrences of each class
        class_counts = df[target_column].value_counts()
        
        # Identify classes with enough examples
        valid_classes = class_counts[class_counts >= min_count].index
        
        # Filter the dataframe
        filtered_df = df[df[target_column].isin(valid_classes)].copy()
        
        self.logger.info(f"Kept {len(filtered_df)} of {len(df)} records after filtering small classes")
        self.logger.info(f"Remaining classes: {len(valid_classes)} of {len(class_counts)}")
        
        return filtered_df
    
    def filter_by_taxonomy(self, 
                          df: pd.DataFrame, 
                          taxonomic_filters: Dict[str, List[str]],
                          exclude: bool = False) -> pd.DataFrame:
        """
        Filter data based on taxonomic criteria.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic data
        taxonomic_filters : Dict[str, List[str]]
            Dictionary mapping taxonomic ranks to lists of allowed values
            Example: {'phylum_name': ['Chordata'], 'class_name': ['Mammalia', 'Aves']}
        exclude : bool
            If True, exclude the specified taxa instead of including them
            
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame
        """
        self.logger.info(f"Filtering by taxonomy: {'excluding' if exclude else 'including'} specified taxa")
        
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Initial mask - keep all records if including, none if excluding
        mask = pd.Series(not exclude, index=filtered_df.index)
        
        # Apply each taxonomic filter
        for rank, values in taxonomic_filters.items():
            if rank not in filtered_df.columns:
                self.logger.warning(f"Column {rank} not found in dataframe, skipping this filter")
                continue
                
            # Create mask for this rank
            if exclude:
                # Exclude the specified values
                rank_mask = ~filtered_df[rank].isin(values)
            else:
                # Include only the specified values
                rank_mask = filtered_df[rank].isin(values)
            
            # Combine with the overall mask using AND
            mask = mask & rank_mask
        
        # Apply the final mask
        result_df = filtered_df[mask].copy()
        
        self.logger.info(f"Kept {len(result_df)} of {len(df)} records after taxonomic filtering")
        
        return result_df
    

    def balance_class_representation(self,
                                df: pd.DataFrame,
                                target_column: str = 'genus_name',
                                min_examples: int = 5,
                                max_examples: Optional[int] = None,
                                sampling_strategy: str = 'random',
                                random_state: int = 42) -> pd.DataFrame:
        """
        Balance dataset by removing underrepresented classes and limiting overrepresented ones.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with taxonomic data
        target_column : str
            Column containing class information to balance
        min_examples : int
            Minimum number of examples required for each class (removes classes with fewer)
        max_examples : int, optional
            Maximum number of examples to keep for each class (samples if more)
        sampling_strategy : str
            Strategy for sampling overrepresented classes: 'random', 'first', or 'diverse'
        random_state : int
            Random seed for reproducible sampling
            
        Returns:
        --------
        pandas.DataFrame
            Balanced DataFrame
        """
        self.logger.info(f"Balancing class representation for {target_column}")
        
        if target_column not in df.columns:
            self.logger.error(f"Target column {target_column} not found in dataframe")
            return df
        
        # Make a copy to avoid modifying the original
        balanced_df = df.copy()
        
        # Count examples per class
        class_counts = balanced_df[target_column].value_counts()
        total_classes = len(class_counts)
        
        # 1. Remove underrepresented classes
        if min_examples > 1:
            underrepresented = class_counts[class_counts < min_examples].index
            if len(underrepresented) > 0:
                balanced_df = balanced_df[~balanced_df[target_column].isin(underrepresented)].copy()
                self.logger.info(f"Removed {len(underrepresented)} underrepresented classes with fewer than {min_examples} examples")
        
        # 2. Limit overrepresented classes
        if max_examples is not None and max_examples > 0:
            overrepresented = class_counts[class_counts > max_examples].index
            
            if len(overrepresented) > 0:
                # Initialize a list to collect the balanced samples
                balanced_samples = []
                
                # First add all classes that aren't overrepresented
                balanced_samples.append(
                    balanced_df[~balanced_df[target_column].isin(overrepresented)]
                )
                
                # Then sample from each overrepresented class
                for cls in overrepresented:
                    class_df = balanced_df[balanced_df[target_column] == cls]
                    
                    if sampling_strategy == 'first':
                        # Take the first max_examples
                        sampled = class_df.iloc[:max_examples]
                    elif sampling_strategy == 'diverse':
                        # Try to get a diverse sample - we'll use sequence length as a diversity metric
                        # Sort by sequence length and take evenly spaced samples
                        if 'sequence' in class_df.columns:
                            class_df['seq_len'] = class_df['sequence'].str.len()
                            class_df = class_df.sort_values('seq_len')
                            indices = np.linspace(0, len(class_df) - 1, max_examples).astype(int)
                            sampled = class_df.iloc[indices].drop('seq_len', axis=1)
                        else:
                            # Fall back to random if sequence column not available
                            sampled = class_df.sample(max_examples, random_state=random_state)
                    else:
                        # Default to random sampling
                        sampled = class_df.sample(max_examples, random_state=random_state)
                    
                    balanced_samples.append(sampled)
                
                # Combine all balanced samples
                balanced_df = pd.concat(balanced_samples, ignore_index=False)
                
                self.logger.info(f"Limited {len(overrepresented)} overrepresented classes to a maximum of {max_examples} examples")
        
        # Calculate final statistics
        final_class_counts = balanced_df[target_column].value_counts()
        final_classes = len(final_class_counts)
        
        self.logger.info(f"Class balance result: {final_classes} classes (from {total_classes}) with "
                    f"{len(balanced_df)} examples (min: {final_class_counts.min()}, max: {final_class_counts.max()}, "
                    f"avg: {final_class_counts.mean():.1f})")
        
        return balanced_df

    def filter_for_balanced_training(self,
                                    input_file: str,
                                    target_rank: str = 'genus',
                                    min_examples: int = 5,
                                    max_examples: Optional[int] = None,
                                    filter_approximated: bool = False,
                                    taxonomic_scope: Optional[Dict[str, List[str]]] = None,
                                    balance_sampling_strategy: str = 'random',
                                    output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Filter and balance data for training, removing underrepresented classes and capping overrepresented ones.
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file with taxonomic data
        target_rank : str
            Taxonomic rank to balance (e.g., 'genus', 'family')
        min_examples : int
            Minimum examples required per class
        max_examples : int, optional
            Maximum examples to keep per class
        filter_approximated : bool
            Whether to remove records with approximated names
        taxonomic_scope : Dict[str, List[str]], optional
            Optional filtering to specific taxa before balancing
        balance_sampling_strategy : str
            Strategy for sampling: 'random', 'first', or 'diverse'
        output_file : str, optional
            Path for saving the balanced data
            
        Returns:
        --------
        pandas.DataFrame
            Filtered and balanced DataFrame
        """
        self.logger.info(f"Preparing balanced training data for {target_rank} prediction")
        
        # Ensure target_rank has proper format
        if not target_rank.endswith('_name'):
            target_column = f"{target_rank}_name"
        else:
            target_column = target_rank
            target_rank = target_rank.replace('_name', '')
        
        # Load the data
        try:
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded {len(df)} records from {input_file}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
        # Step 1: Handle approximated names if requested
        if filter_approximated:
            df = self.clean_approximated_names(df, remove_approximated=True)
        
        # Step 2: Apply taxonomic scope filtering if provided
        if taxonomic_scope:
            df = self.filter_by_taxonomy(df, taxonomic_scope)
        
        # Step 3: Balance class representation
        balanced_df = self.balance_class_representation(
            df,
            target_column=target_column,
            min_examples=min_examples,
            max_examples=max_examples,
            sampling_strategy=balance_sampling_strategy
        )
        
        # Step 4: Save to file if requested
        if output_file:
            balanced_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved balanced data to {output_file}")
        else:
            # Generate default output filename
            default_output = os.path.join(
                self.filtered_data_dir, 
                f"{target_rank}_balanced_min{min_examples}" + 
                (f"_max{max_examples}" if max_examples else "") + 
                ".csv"
            )
            balanced_df.to_csv(default_output, index=False)
            self.logger.info(f"Saved balanced data to {default_output}")
        
        # Print summary
        class_counts = balanced_df[target_column].value_counts()
        self.logger.info(f"Final balanced dataset: {len(balanced_df)} records across {len(class_counts)} classes")
        self.logger.info(f"Class sizes - Min: {class_counts.min()}, Max: {class_counts.max()}, Mean: {class_counts.mean():.1f}")
        
        return balanced_df
    

# Example usage
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create filter instance
    data_filter = TaxonomyDataFilter()
    
    try:
        # Load example data
        input_file = "data/processed/cleaned_sequences.csv"
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
        
        # Example 1: Clean approximated names
        # Option 1: Just normalize the names (remove approximation source)
        normalized_df = data_filter.clean_approximated_names(df, remove_approximated=False)
        print(f"Normalized names: {len(normalized_df)} records")
        
        # Option 2: Remove records with approximated names
        filtered_df = data_filter.clean_approximated_names(df, remove_approximated=True)
        print(f"After removing approximated names: {len(filtered_df)} records")
        
        # Example 2: Filter small classes
        min_count = 10
        target_column = "genus_name"
        filtered_small_df = data_filter.filter_small_classes(df, min_count=min_count, target_column=target_column)
        print(f"After filtering classes with fewer than {min_count} examples: {len(filtered_small_df)} records")
        print(f"Number of {target_column} classes: {filtered_small_df[target_column].nunique()}")
        
        # Example 3: Filter by taxonomy
        # Include only specific taxonomic groups
        include_taxa = {
            "phylum_name": ["Chordata", "Arthropoda"],
            "class_name": ["Mammalia", "Aves", "Insecta"]
        }
        filtered_taxa_df = data_filter.filter_by_taxonomy(df, include_taxa, exclude=False)
        print(f"After including specific taxa: {len(filtered_taxa_df)} records")
        
        # Exclude specific taxonomic groups
        exclude_taxa = {
            "order_name": ["Diptera", "Coleoptera"]
        }
        filtered_exclude_df = data_filter.filter_by_taxonomy(df, exclude_taxa, exclude=True)
        print(f"After excluding specific taxa: {len(filtered_exclude_df)} records")
        
        # Example 4: Balance class representation
        # Balance classes with minimum and maximum counts
        balanced_df = data_filter.balance_class_representation(
            df,
            target_column="genus_name",
            min_examples=5,
            max_examples=50,
            sampling_strategy="random"
        )
        print(f"After balancing classes: {len(balanced_df)} records")
        genus_counts = balanced_df["genus_name"].value_counts()
        print(f"Genus class statistics - Min: {genus_counts.min()}, Max: {genus_counts.max()}, Mean: {genus_counts.mean():.1f}")
        
        # Example 5: Comprehensive filtering for training
        # Apply a complete workflow for preparing training data
        training_df = data_filter.filter_for_balanced_training(
            input_file=input_file,
            target_rank="genus",
            min_examples=10,
            max_examples=100,
            filter_approximated=True,
            taxonomic_scope={"phylum_name": ["Chordata"]},
            balance_sampling_strategy="diverse",
            output_file="data/filtered/genus_balanced_training.csv"
        )
        
        print(f"Prepared balanced training dataset: {len(training_df)} records")
        print(f"Number of genera: {training_df['genus_name'].nunique()}")
        
        # Save filtered results for different scenarios
        data_filter.clean_approximated_names(df, remove_approximated=False).to_csv(
            "data/filtered/normalized_names.csv", index=False
        )
        
        filtered_small_df.to_csv(
            f"data/filtered/{target_column}_min{min_count}.csv", index=False
        )
        
        filtered_taxa_df.to_csv(
            "data/filtered/chordata_arthropoda_filtered.csv", index=False
        )
        
        print("Saved filtered datasets to data/filtered/ directory")
        
    except Exception as e:
        print(f"Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
        