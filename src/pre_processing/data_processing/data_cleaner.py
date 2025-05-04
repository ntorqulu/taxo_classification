import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

class TaxonomyDataCleaner:
    """
    Class for cleaning and preprocessing taxonomy sequence data.
    Provides configurable cleaning options for different taxonomic classification tasks.
    """
    
    def __init__(
        self,
        taxonomy_ranks: List[str] = None,
    ):
        """
        Initialize the data cleaner with taxonomy rank information.
        
        Parameters:
        -----------
        taxonomy_ranks : List[str], optional
            List of taxonomy rank column names in the dataset
        """
        self.taxonomy_ranks = taxonomy_ranks or [
            'euk_class', 'metazoa_class', 'arthropoda_class', 'insecta_class'
        ]
    
    def clean_data(
        self,
        df: pd.DataFrame,
        min_seq_length: int = 100,
        max_n_percent: float = 5.0,
        keep_approximations: bool = True,
        require_complete_ranks_up_to: Optional[str] = "order",
        min_count_per_class: int = 5,
        merge_rare_classes: bool = False,
        remove_duplicates: bool = False,
        filter_nonstandard_bases: bool = False,
        enforce_taxonomy_consistency: bool = False,
        filter_gc_outliers: bool = False,
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
        keep_approximations : bool
            Whether to clean approximated taxonomic names
        require_complete_ranks_up_to : str or None
            Require all taxonomic ranks up to this level to be non-null
            Options: "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species", None
        min_count_per_class : int
            Minimum number of examples per class to keep (for merged classes)
        merge_rare_classes : bool
            Whether to merge rare classes into an "Other" category
        remove_duplicates : bool
            Whether to remove duplicate sequences
        filter_nonstandard_bases : bool
            Whether to remove sequences with bases other than ATGCN
        enforce_taxonomy_consistency : bool
            Whether to check and report taxonomic inconsistencies
        filter_gc_outliers : bool
            Whether to remove sequences with outlier GC content
        return_mask_only : bool
            If True, return only the boolean mask of records to keep
        verbose : bool
            Whether to print details about the cleaning process
            
        Returns:
        --------
        pandas.DataFrame or pandas.Series
            If return_mask_only is False (default), returns cleaned dataframe with additional metadata columns.
            If return_mask_only is True, returns a boolean Series indicating which rows to keep.
        """
        # Work on a copy to avoid modifying the original
        cleaned_df = df.copy()
        original_count = len(cleaned_df)
        
        if verbose:
            print(f"Starting data cleaning process...")
            print(f"Original dataset: {original_count} sequences")
        
        # Make sure seq_length is numeric
        if 'seq_length' not in cleaned_df.columns:
            cleaned_df['seq_length'] = cleaned_df['sequence'].str.len()
        else:
            cleaned_df['seq_length'] = pd.to_numeric(cleaned_df['seq_length'], errors='coerce')
        
        # Calculate N content if not already present
        if 'N_percent' not in cleaned_df.columns:
            cleaned_df['N_count'] = cleaned_df['sequence'].str.count('N')
            cleaned_df['N_percent'] = cleaned_df['N_count'] / cleaned_df['seq_length'] * 100
        
        # Create a mask to track kept sequences
        kept_mask = pd.Series(True, index=cleaned_df.index)
        
        # 1. Filter by sequence length
        if min_seq_length > 0:
            length_mask = cleaned_df['seq_length'] >= min_seq_length
            removed_count = (~length_mask & kept_mask).sum()
            kept_mask &= length_mask
            if verbose:
                print(f"Removed {removed_count} sequences shorter than {min_seq_length}bp")
        
        # 2. Filter non-standard bases
        if filter_nonstandard_bases:
            # Check for bases other than ATGCN
            nonstandard_mask = cleaned_df['sequence'].str.contains('[^ATGCN]', case=False)
            removed_count = (nonstandard_mask & kept_mask).sum()
            kept_mask &= ~nonstandard_mask
            
            if verbose:
                print(f"Removed {removed_count} sequences with non-standard bases")
        
        # 3. Filter by N content
        if max_n_percent < 100:
            n_content_mask = cleaned_df['N_percent'] <= max_n_percent
            removed_count = (~n_content_mask & kept_mask).sum()
            kept_mask &= n_content_mask
            if verbose:
                print(f"Removed {removed_count} sequences with >{max_n_percent}% N content")
        
        # 4. Remove duplicate sequences
        if remove_duplicates:
            # Count before
            pre_dedup_count = kept_mask.sum()
            
            # Find duplicates (exact sequence matches)
            duplicate_mask = cleaned_df.duplicated(subset=['sequence'], keep='first')
            removed_count = (duplicate_mask & kept_mask).sum()
            kept_mask &= ~duplicate_mask
            
            if verbose:
                print(f"Removed {removed_count} duplicate sequences")
        
        # 5. Process approximated taxonomic names
        # First, mark approximated values for all ranks regardless of keep_approximations
        for rank in self.taxonomy_ranks:
            # Skip if column doesn't exist
            if rank not in cleaned_df.columns:
                continue
                
            # Mark approximated values
            pattern = r'\(from '
            cleaned_df[f"{rank}_approx"] = cleaned_df[rank].astype(str).str.contains(pattern, na=False)

        # Then process them based on keep_approximations flag
        if keep_approximations:
            for rank in self.taxonomy_ranks:
                if rank not in cleaned_df.columns or f"{rank}_approx" not in cleaned_df.columns:
                    continue
                    
                # Clean up approximated values by removing the "(from ...)" part
                if cleaned_df[f"{rank}_approx"].any():
                    # Use a safer string split approach
                    def clean_approx_name(x):
                        if pd.isna(x) or not isinstance(x, str):
                            return x
                        parts = x.split(' (from ')
                        return parts[0] if len(parts) > 0 else x
                        
                    # Apply the cleaning function only to approximated values
                    approx_mask = cleaned_df[f"{rank}_approx"]
                    cleaned_df.loc[approx_mask, rank] = cleaned_df.loc[approx_mask, rank].apply(clean_approx_name)
            
            if verbose:
                approx_count = sum(cleaned_df[f"{rank}_approx"].sum() for rank in self.taxonomy_ranks if f"{rank}_approx" in cleaned_df.columns)
                print(f"Cleaned {approx_count} approximated taxonomic values")
        else:
            # If not cleaning approximations, delete the rows with approximated values
            for rank in self.taxonomy_ranks:
                if rank not in cleaned_df.columns or f"{rank}_approx" not in cleaned_df.columns:
                    continue
                    
                # Remove rows with approximated values
                approx_mask = cleaned_df[f"{rank}_approx"]
                removed_count = (approx_mask & kept_mask).sum()
                kept_mask &= ~approx_mask
                
                if verbose:
                    print(f"Removed {removed_count} sequences with approximated {rank}")
            
            # Drop all approximation columns at once
            approx_cols = [f"{rank}_approx" for rank in self.taxonomy_ranks if f"{rank}_approx" in cleaned_df.columns]
            if approx_cols:
                cleaned_df.drop(columns=approx_cols, inplace=True)

        # 6. Check taxonomic consistency
        if enforce_taxonomy_consistency:
            inconsistencies = 0
            
            # For each pair of parent-child ranks
            for i in range(1, len(self.taxonomy_ranks)):
                parent_rank = self.taxonomy_ranks[i-1]
                child_rank = self.taxonomy_ranks[i]
                
                # Group by child taxon and check if parent is consistent
                if parent_rank in cleaned_df.columns and child_rank in cleaned_df.columns:
                    # Group by child, get unique parents
                    grouped = cleaned_df.dropna(subset=[child_rank]).groupby(child_rank)[parent_rank]
                    
                    # Find child taxa with inconsistent parents
                    inconsistent_children = []
                    for name, group in grouped:
                        if group.nunique() > 1:
                            inconsistent_children.append(name)
                            inconsistencies += 1
                    
                    if inconsistent_children and verbose:
                        print(f"Found {len(inconsistent_children)} {child_rank} with inconsistent {parent_rank}")
                        
                        # For each inconsistent child, print details about the inconsistency
                        for child_name in inconsistent_children:
                            # Get all sequences with this child taxon
                            inconsistent_seqs = cleaned_df[cleaned_df[child_rank] == child_name]
                            
                            # Group by the parent rank to see different values
                            parent_groups = inconsistent_seqs.groupby(parent_rank)
                            
                            print(f"\n  Inconsistency details for {child_rank}='{child_name}':")
                            print(f"  Has {parent_groups.ngroups} different {parent_rank} values:")
                            
                            # Print each parent value and some sequence IDs
                            for parent_name, parent_group in parent_groups:
                                # Get sequence IDs, limiting to first 3 for brevity
                                if 'sequence_id' in parent_group.columns:
                                    seq_ids = parent_group['sequence_id'].head(3).tolist()
                                    seq_id_str = ", ".join(str(id) for id in seq_ids)
                                    
                                    print(f"    {parent_rank}='{parent_name}' ({len(parent_group)} sequences)")
                                    print(f"      Example seq_ids: {seq_id_str}")
                                else:
                                    # If no sequence_id column, use index
                                    indices = parent_group.index.tolist()[:3]
                                    indices_str = ", ".join(str(idx) for idx in indices)
                                    
                                    print(f"    {parent_rank}='{parent_name}' ({len(parent_group)} sequences)")
                                    print(f"      Example indices: {indices_str}")
            
            if verbose and inconsistencies > 0:
                print(f"\nTotal taxonomic inconsistencies: {inconsistencies}")
        
        # 7. Filter by taxonomic completeness
        if require_complete_ranks_up_to:
            rank_indices = {rank.replace('_class', ''): i for i, rank in enumerate(
                [r.replace('_class', '') for r in self.taxonomy_ranks]
            )}
            
            if require_complete_ranks_up_to in rank_indices:
                required_level = rank_indices[require_complete_ranks_up_to]
                
                # Require all ranks up to the specified level
                required_ranks = self.taxonomy_ranks[:required_level + 1]
                completeness_mask = cleaned_df[required_ranks].notna().all(axis=1)
                
                removed_count = (~completeness_mask & kept_mask).sum()
                kept_mask &= completeness_mask
                
                if verbose:
                    print(f"Removed {removed_count} sequences missing taxonomy up to {require_complete_ranks_up_to}")
        
        # 8. Filter GC content outliers
        if filter_gc_outliers:
            # Calculate GC content if not already done
            if 'GC_content' not in cleaned_df.columns:
                for base in ['G', 'C']:
                    if f'{base}_count' not in cleaned_df.columns:
                        cleaned_df[f'{base}_count'] = cleaned_df['sequence'].str.count(base)
                cleaned_df['GC_content'] = (cleaned_df['G_count'] + cleaned_df['C_count']) / cleaned_df['seq_length'] * 100
            
            # Use IQR method to detect outliers
            Q1 = cleaned_df.loc[kept_mask, 'GC_content'].quantile(0.25)
            Q3 = cleaned_df.loc[kept_mask, 'GC_content'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outliers (sequences outside 1.5*IQR)
            gc_outlier_mask = (cleaned_df['GC_content'] < (Q1 - 1.5 * IQR)) | (cleaned_df['GC_content'] > (Q3 + 1.5 * IQR))
            removed_count = (gc_outlier_mask & kept_mask).sum()
            kept_mask &= ~gc_outlier_mask
            
            if verbose:
                print(f"Removed {removed_count} sequences with outlier GC content (outside {Q1-1.5*IQR:.1f}% to {Q3+1.5*IQR:.1f}%)")
        
        # 9. Optional: Merge rare classes
        if merge_rare_classes:
            for rank in self.taxonomy_ranks:
                if rank not in cleaned_df.columns:
                    continue
                    
                # Get counts for this rank
                counts = cleaned_df.loc[kept_mask, rank].value_counts()
                rare_classes = counts[counts < min_count_per_class].index
                
                if len(rare_classes) > 0:
                    # Create a mapping dictionary
                    rare_mask = cleaned_df[rank].isin(rare_classes)
                    cleaned_df.loc[rare_mask, rank] = f"Other_{rank.split('_')[0]}"
                    
                    if verbose:
                        print(f"Merged {len(rare_classes)} rare classes in {rank} (< {min_count_per_class} examples)")
        
        # Return early if only mask requested
        if return_mask_only:
            return kept_mask
            
        # 10. Add metadata columns for analysis
        # Mark which sequences are kept after cleaning
        cleaned_df['kept_after_cleaning'] = kept_mask
        
        # Calculate taxonomic completeness
        cleaned_df['taxonomic_completeness'] = cleaned_df[self.taxonomy_ranks].notna().sum(axis=1) / len(self.taxonomy_ranks) * 100
        
        # Final summary
        if verbose:
            kept_count = kept_mask.sum()
            print(f"\nFinal cleaned dataset: {kept_count} sequences ({kept_count/original_count*100:.2f}% of original)")
            print(f"Dropped {original_count - kept_count} sequences")
        
        return cleaned_df
    
    def balance_dataset(self, df, target_column, method='undersample', max_ratio=100, min_samples=5, 
                    random_state=42, verbose=True):
        """
        Balance dataset using various strategies to address class imbalance.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Column to balance (e.g., 'phylum_name')
        method : str
            Balancing method to use:
            - 'undersample': Reduce majority classes (keeps original minority samples)
            - 'random_oversample': Increase minority classes by duplication
            - 'stratified': Create balanced sample with equal representation
            - 'hybrid': Combine undersampling of majority and oversampling of minority
        max_ratio : int
            Maximum ratio between most common and least common class (for undersampling)
        min_samples : int
            Minimum samples required per class for resampling methods
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print details about the balancing process
            
        Returns:
        --------
        pandas.DataFrame
            Balanced dataframe
        """
        # Filter out rows with missing target values
        valid_df = df.dropna(subset=[target_column])
        if len(valid_df) < len(df) and verbose:
            print(f"Removed {len(df) - len(valid_df)} rows with missing {target_column} values")
        
        if len(valid_df) == 0:
            print(f"Warning: No valid data after removing missing {target_column} values")
            return df
        
        # Get class distribution
        counts = valid_df[target_column].value_counts()
        if verbose:
            print(f"Original class distribution in {target_column}:")
            print(counts.to_string())
            print(f"Imbalance ratio: {counts.max() / counts.min():.2f}")
        
        # Simple undersampling
        if method == 'undersample':
            # Calculate sampling targets for each class
            min_class_count = counts.min()
            target_counts = {cls: min(count, min_class_count * max_ratio) 
                            for cls, count in counts.items()}
            
            # Sample from each class
            balanced_df = pd.DataFrame()
            for cls, target in target_counts.items():
                cls_df = valid_df[valid_df[target_column] == cls]
                if len(cls_df) > target:
                    # Need to undersample
                    sampled = cls_df.sample(n=target, random_state=random_state)
                else:
                    # Keep all samples for minority classes
                    sampled = cls_df
                balanced_df = pd.concat([balanced_df, sampled])
            
            if verbose:
                new_counts = balanced_df[target_column].value_counts()
                print(f"After undersampling:")
                print(new_counts.to_string())
                print(f"New imbalance ratio: {new_counts.max() / new_counts.min():.2f}")
            
            return balanced_df
        
        # Random oversampling
        elif method == 'random_oversample':
            # Set target to the majority class count
            max_class_count = counts.max()
            
            # Sample from each class to reach the target
            balanced_df = pd.DataFrame()
            for cls, count in counts.items():
                cls_df = valid_df[valid_df[target_column] == cls]
                if count < max_class_count:
                    # Need to oversample - sample with replacement
                    sampled = cls_df.sample(n=max_class_count, replace=True, random_state=random_state)
                else:
                    # Keep all samples for majority class
                    sampled = cls_df
                balanced_df = pd.concat([balanced_df, sampled])
            
            if verbose:
                new_counts = balanced_df[target_column].value_counts()
                print(f"After random oversampling:")
                print(new_counts.to_string())
            
            return balanced_df
        
        # Hybrid approach - undersample majorities and oversample minorities
        elif method == 'hybrid':
            # Calculate balanced target count - middle ground between min and max
            min_count = counts.min()
            max_count = counts.max()
            target_count = int(np.sqrt(min_count * max_count))  # geometric mean
            
            # Sample from each class
            balanced_df = pd.DataFrame()
            for cls, count in counts.items():
                cls_df = valid_df[valid_df[target_column] == cls]
                if count > target_count:
                    # Undersample
                    sampled = cls_df.sample(n=target_count, random_state=random_state)
                elif count < target_count:
                    # Oversample
                    sampled = cls_df.sample(n=target_count, replace=True, random_state=random_state)
                else:
                    # Keep as is
                    sampled = cls_df
                balanced_df = pd.concat([balanced_df, sampled])
            
            if verbose:
                new_counts = balanced_df[target_column].value_counts()
                print(f"After hybrid sampling (target count: {target_count}):")
                print(new_counts.to_string())
            
            return balanced_df
        
        # Stratified sampling - roughly equal numbers of each class
        elif method == 'stratified':
            # Set target to smaller of: min_count * 3 or 100
            target_count = min(100, min_class_count * 3)
            
            balanced_df = pd.DataFrame()
            for cls in counts.index:
                cls_df = valid_df[valid_df[target_column] == cls]
                n_samples = min(len(cls_df), target_count)
                sampled = cls_df.sample(n=n_samples, random_state=random_state)
                balanced_df = pd.concat([balanced_df, sampled])
            
            if verbose:
                new_counts = balanced_df[target_column].value_counts()
                print(f"After stratified sampling:")
                print(new_counts.to_string())
            
            return balanced_df
        
        else:
            raise ValueError(f"Unknown balancing method: {method}. "
                            f"Use one of: 'undersample', 'random_oversample' "
                            f"'stratified', 'hybrid'")