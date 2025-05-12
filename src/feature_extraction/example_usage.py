#!/usr/bin/env python
# feature_extraction_example.py

import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction import FeatureExtractor, KmerEncoder


def setup_logging(log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('feature_extraction_example')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract features from DNA sequences')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input CSV file with processed sequences')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the extracted features')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate PCA visualization of features')
    parser.add_argument('--label-column', type=str, default='insecta_class',
                        help='Column to use for coloring in visualization')
    
    return parser.parse_args()

def main():
    """Main function to extract features from sequences."""
    args = parse_args()
    logger = setup_logging()
    
    # 1. Load the processed CSV file
    logger.info(f"Loading data from {args.input}")
    data = pd.read_csv(args.input)
    logger.info(f"Loaded {len(data)} sequences")
    
    # Display basic statistics about the dataset
    logger.info(f"Sequence length statistics:")
    seq_lengths = data['sequence'].str.len()
    logger.info(f"  Min length: {seq_lengths.min()}")
    logger.info(f"  Max length: {seq_lengths.max()}")
    logger.info(f"  Mean length: {seq_lengths.mean():.1f}")
    
    # 2. Extract sequences
    sequences = data['sequence'].values
    
    # 3. Create feature extractors
    logger.info("Setting up feature extractors")
    
    # K-mer based features
    kmer3_encoder = KmerEncoder(k=3, mode='count')
    
    # Create the feature extractor
    extractor = FeatureExtractor()
    extractor.add_encoder(kmer3_encoder, 'kmer3')
    
    # 4. Extract features
    logger.info("Extracting features...")
    
    extractor.fit(sequences)
    
    # Extract only k-mer features for faster processing
    kmer_features = extractor.transform(
        sequences, 
        encoders=['kmer3'],
        concatenate=True,
        normalize=False
    )
    
    # Get feature names
    feature_names = extractor.get_feature_names(encoders=['kmer3'])
    
    logger.info(f"Extracted feature matrix shape: {kmer_features.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    
    # 5. Save features if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create DataFrame with features
        feature_df = pd.DataFrame(kmer_features, columns=feature_names)
        
        # Add sequence IDs and labels if available
        if 'sequence_id' in data.columns:
            feature_df.insert(0, 'sequence_id', data['sequence_id'])
        
        if args.label_column in data.columns:
            feature_df.insert(0, args.label_column, data[args.label_column])
        
        # Save to CSV
        feature_df.to_csv(args.output, index=False)
        logger.info(f"Saved feature matrix to {args.output}")
    
    # 6. Visualize features with PCA if requested
    if args.visualize:
        logger.info("Generating PCA visualization")
        
        # Reduce dimensions with PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(kmer_features)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PCA1': pca_result[:, 0],
            'PCA2': pca_result[:, 1]
        })
        
        # Add labels if available
        if args.label_column in data.columns:
            pca_df[args.label_column] = data[args.label_column]
            
            # Plot with labels
            plt.figure(figsize=(12, 10))
            
            # If there are many classes, limit to top 10
            if pca_df[args.label_column].nunique() > 10:
                top_classes = pca_df[args.label_column].value_counts().nlargest(10).index
                plot_df = pca_df[pca_df[args.label_column].isin(top_classes)].copy()
                logger.info(f"Plotting top 10 classes out of {pca_df[args.label_column].nunique()} total classes")
            else:
                plot_df = pca_df.copy()
            
            # Plot
            sns.scatterplot(
                data=plot_df, 
                x='PCA1', 
                y='PCA2', 
                hue=args.label_column,
                alpha=0.7,
                s=50
            )
            
            # Add title and labels
            plt.title(f'PCA of Sequence Features by {args.label_column}', fontsize=14)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            plt.tight_layout()
            
            # Save figure
            plt.savefig('feature_pca_visualization.png', dpi=300)
            logger.info("Saved PCA visualization to feature_pca_visualization.png")
        else:
            logger.warning(f"Cannot generate labeled visualization: {args.label_column} not found in data")
            
            # Simple plot without labels
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title('PCA of Sequence Features', fontsize=14)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            plt.tight_layout()
            plt.savefig('feature_pca_visualization.png', dpi=300)
            logger.info("Saved PCA visualization to feature_pca_visualization.png")
    
    logger.info("Feature extraction completed successfully")

if __name__ == "__main__":
    main()