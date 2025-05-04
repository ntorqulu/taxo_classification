from data_processing import TaxonomyDataCleaner, load_taxonomy_data
import pandas as pd
import argparse
import os
import logging

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('taxo_classification')

def main(args):
    """
    Run the data preprocessing pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing:
        - output: Directory to save cleaned data
        - strategy: Cleaning strategy ('strict', 'moderate', 'lenient', or 'all')
        - balance: Balancing strategy to apply
        - balance_rank: Taxonomic rank to balance
        - balance_ratio: Maximum class imbalance ratio
        - min_samples: Minimum samples per class for advanced balancing
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(log_file=f"logs/log_{args.strategy}_{args.balance}_{timestamp}.log")
    logger.info("Starting data preprocessing pipeline...")
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    if output_dir is None:
        output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading taxonomy data...")
    df = load_taxonomy_data()
    
    # Initialize data cleaner
    cleaner = TaxonomyDataCleaner()
    
    if args.strategy == 'all':
        # Process with all strategies
        logger.info("Applying all cleaning strategies...")
        strategies = cleaner.create_cleaning_strategies(df)
        
        # Apply balancing to all strategies if requested
        if args.balance != 'none':
            logger.info(f"Applying {args.balance} balancing strategy to all cleaning strategies...")
            for name in strategies.keys():
                logger.info(f"Balancing {name} strategy...")
                strategies[name] = cleaner.balance_dataset(
                    strategies[name],
                    target_column=args.balance_rank,
                    method=args.balance,
                    max_ratio=args.balance_ratio,
                    min_samples=args.min_samples,
                    random_state=args.random_state,
                    verbose=True
                )
                logger.info(f"Balanced {name} dataset")
        
        # Save each strategy
        for name, clean_df in strategies.items():
            balance_suffix = f"_{args.balance}" if args.balance != 'none' else ""
            output_path = os.path.join(output_dir, f"{name}{balance_suffix}_cleaned_taxonomy.csv")
            clean_df[clean_df['kept_after_cleaning']].to_csv(output_path, index=False)
            logger.info(f"Saved {name} strategy data to: {output_path}")
    else:
        # Apply the selected strategy
        logger.info(f"Applying {args.strategy} cleaning strategy...")
        
        # Configure parameters based on strategy
        if args.strategy == 'strict':
            params = {
                'min_seq_length': 200,
                'max_n_percent': 1.0,
                'require_complete_ranks_up_to': 'order',
                'remove_duplicates': True,
                'filter_nonstandard_bases': True,
                'enforce_taxonomy_consistency': True,
                'filter_gc_outliers': True,
                'keep_approximations': True
            }
        elif args.strategy == 'lenient':
            params = {
                'min_seq_length': 290,
                'max_n_percent': 0.0,
                'require_complete_ranks_up_to': "Order",
                'merge_rare_classes': True,
                'min_count_per_class': 3,
                'remove_duplicates': True,
                'filter_nonstandard_bases': True,
                'enforce_taxonomy_consistency': True,
                'filter_gc_outliers': True,
                'keep_approximations': False
            }
        else:  # moderate (default)
            params = {
                'min_seq_length': 100,
                'max_n_percent': 3.0,
                'require_complete_ranks_up_to': 'phylum',
                'remove_duplicates': True,
                'filter_nonstandard_bases': True,
                'enforce_taxonomy_consistency': False,
                'filter_gc_outliers': False,
                'keep_approximations': True
            }
        
        # Clean the data
        cleaned_df = cleaner.clean_data(df, **params)
        
        # Apply balance_dataset with enhanced parameters
        if args.balance != 'none':
            logger.info(f"Applying {args.balance} balancing strategy...")
            cleaned_df = cleaner.balance_dataset(
                cleaned_df, 
                target_column=args.balance_rank, 
                method=args.balance, 
                max_ratio=args.balance_ratio,
                min_samples=args.min_samples,
                random_state=args.random_state,
                verbose=True
            )
            logger.info(f"Balanced dataset with {args.balance} strategy.")
        else:
            logger.info("No balancing applied.")
        
        # Save the cleaned data
        balance_suffix = f"_{args.balance}" if args.balance != 'none' else ""
        output_path = os.path.join(output_dir, f"{args.strategy}{balance_suffix}_cleaned_taxonomy.csv")
        cleaned_df[cleaned_df['kept_after_cleaning']].to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to: {output_path}")
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy data preprocessing pipeline")
    parser.add_argument("--output", help="Output directory for processed data", default=None)
    parser.add_argument(
        "--strategy", 
        choices=["strict", "moderate", "lenient", "all"], 
        default="lenient",
        help="Cleaning strategy to apply"
    )
    parser.add_argument(
        "--balance", 
        choices=["none", "undersample", "random_oversample", "stratified", "hybrid"],
        default="none",
        help="Apply class balancing to the selected taxonomic rank"
    )
    parser.add_argument(
        "--balance-rank",
        default="phylum_name",
        help="Taxonomic rank to balance (e.g., 'phylum_name', 'class_name')"
    )
    parser.add_argument(
        "--balance-ratio",
        type=int,
        default=100,
        help="Maximum ratio between most common and least common class (for undersampling)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples required per class for resampling methods"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility in balancing operations"
    ) 
    args = parser.parse_args()
    main(args)