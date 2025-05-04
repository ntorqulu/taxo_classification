import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import os
import sys
import argparse
from typing import Dict, Any, Optional

# Import from other modules
from ...feature_extraction import FeatureExtractor, KmerEncoder, StatisticalEncoder
from ...models.utils.model_factory import create_taxonomy_classifier
from ...models.training.trainer import Trainer

def setup_logging(log_dir: Optional[str] = None):
    """Configure logging for training."""
    os.makedirs(log_dir or 'logs', exist_ok=True)
    log_file = os.path.join(log_dir or 'logs', 'taxonomy_training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger('taxonomy_training')

def prepare_data(data_path: str, target_column: str, filter_classes: Optional[list] = None):
    """
    Load and prepare data for training.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    target_column : str
        Column to use as classification target
    filter_classes : list, optional
        Classes to exclude from training
        
    Returns:
    --------
    tuple
        (data, sequences, y, label_encoder, num_classes)
    """
    logger = logging.getLogger('taxonomy_training')
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} sequences")
    
    # Filter out unwanted classes
    if filter_classes:
        before_count = len(data)
        data = data[~data[target_column].isin(filter_classes)]
        logger.info(f"Filtered out {before_count - len(data)} sequences with classes: {filter_classes}")
    
    # Remove missing values
    data = data.dropna(subset=[target_column, 'sequence'])
    logger.info(f"After removing missing values: {len(data)} sequences")
    
    # Label encode the target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[target_column])
    num_classes = len(label_encoder.classes_)
    
    # Class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    for i, count in enumerate(class_counts):
        class_name = label_encoder.inverse_transform([i])[0]
        logger.info(f"Class {i} ({class_name}): {count} sequences ({count/len(data):.2%})")
    
    return data, data['sequence'].values, y, label_encoder, num_classes

def extract_features(sequences, feature_config: Dict[str, Any] = None):
    """
    Extract features from sequences.
    
    Parameters:
    -----------
    sequences : array-like
        Sequences to extract features from
    feature_config : dict, optional
        Configuration for feature extraction
        
    Returns:
    --------
    numpy.ndarray
        Extracted features
    """
    logger = logging.getLogger('taxonomy_training')
    logger.info("Extracting features")
    
    # Default feature config
    if feature_config is None:
        feature_config = {
            'kmer': {'k': 3, 'mode': 'frequency'},
            'stats': {'features': ['gc_content', 'entropy', 'nucleotide_freq']}
        }
    
    # Create feature extractors
    extractor = FeatureExtractor()
    
    if 'kmer' in feature_config:
        kmer_params = feature_config['kmer']
        kmer_encoder = KmerEncoder(**kmer_params)
        extractor.add_encoder(kmer_encoder, 'kmer')
        logger.info(f"Added k-mer encoder with parameters: {kmer_params}")
    
    if 'stats' in feature_config:
        stats_params = feature_config['stats']
        stats_encoder = StatisticalEncoder(**stats_params)
        extractor.add_encoder(stats_encoder, 'stats')
        logger.info(f"Added statistical encoder with parameters: {stats_params}")
    
    # Extract features
    X = extractor.fit_transform(sequences)
    logger.info(f"Extracted features with shape: {X.shape}")
    
    return X

def prepare_dataloaders(X, y, test_size=0.2, val_size=0.1, batch_size=64, random_state=42):
    """
    Prepare DataLoaders for training.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of training data to use for validation
    batch_size : int
        Batch size for DataLoaders
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger('taxonomy_training')
    
    # First split: training+validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: training vs validation
    # Adjust val_size to be a proportion of train_val, not the original dataset
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=adjusted_val_size, 
        random_state=random_state, 
        stratify=y_train_val
    )
    
    logger.info(f"Data split: {len(y_train)} train, {len(y_val)} validation, {len(y_test)} test")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, test_loader, num_classes, input_size, 
                model_config=None, training_config=None, model_dir='models/saved'):
    """
    Train the taxonomy classifier model.
    
    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    test_loader : DataLoader
        DataLoader for test data
    num_classes : int
        Number of classes to predict
    input_size : int
        Size of input features
    model_config : dict, optional
        Model configuration
    training_config : dict, optional
        Training configuration
    model_dir : str
        Directory to save models
        
    Returns:
    --------
    tuple
        (model, trainer, history, final_metrics)
    """
    logger = logging.getLogger('taxonomy_training')
    
    # Default configurations
    if model_config is None:
        model_config = {
            'model_type': 'cnn',
            'input_channels': 1,
            'conv_channels': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'fc_sizes': [512, 256],
            'dropout': 0.5
        }
    
    if training_config is None:
        training_config = {
            'lr': 0.001,
            'epochs': 30,
            'patience': 10,
            'scheduler_factor': 0.5,
            'scheduler_patience': 5
        }
    
    # Create model
    logger.info(f"Creating model with config: {model_config}")
    model = create_taxonomy_classifier(
        num_classes=num_classes,
        input_size=input_size,
        **model_config
    )
    
    # Create optimizer and scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=training_config['scheduler_factor'], 
        patience=training_config['scheduler_patience']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
    )
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    logger.info(f"Training for {training_config['epochs']} epochs with patience {training_config['patience']}")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
        patience=training_config['patience'],
        model_dir=model_dir,
        verbose=True
    )
    
    # Evaluate on test set
    logger.info("Evaluating final model on test set")
    final_metrics = trainer.validate(test_loader)
    logger.info(f"Final test metrics: {final_metrics}")
    
    return model, trainer, history, final_metrics

def save_results(model_dir, label_encoder, history=None, class_weights=None):
    """
    Save training results and class mapping.
    
    Parameters:
    -----------
    model_dir : str
        Directory to save results
    label_encoder : LabelEncoder
        Encoder for class labels
    history : dict, optional
        Training history
    class_weights : array-like, optional
        Class weights used in training
    """
    logger = logging.getLogger('taxonomy_training')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save class mapping
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    pd.DataFrame(list(class_mapping.items()), 
                columns=['class_id', 'class_name']).to_csv(
                os.path.join(model_dir, 'class_mapping.csv'), index=False
    )
    logger.info(f"Saved class mapping to {os.path.join(model_dir, 'class_mapping.csv')}")
    
    # Save history if provided
    if history:
        pd.DataFrame(history).to_csv(
            os.path.join(model_dir, 'training_history.csv'), index=False
        )
        logger.info(f"Saved training history to {os.path.join(model_dir, 'training_history.csv')}")
    
    # Save class weights if provided
    if class_weights is not None:
        np.save(os.path.join(model_dir, 'class_weights.npy'), class_weights)
        logger.info(f"Saved class weights to {os.path.join(model_dir, 'class_weights.npy')}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train taxonomy classifier')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default='data/merged/insects_trimmed_coi.csv',
                      help='Path to input data')
    parser.add_argument('--target-column', type=str, default='insecta_class',
                      help='Column to use as classification target')
    parser.add_argument('--filter-classes', type=str, nargs='+', default=['Others', 'No_insecta'],
                      help='Classes to exclude from training')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='cnn',
                      help='Type of model to train')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    
    # Output parameters
    parser.add_argument('--model-dir', type=str, default='models/saved',
                      help='Directory to save model and results')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory to save logs')
    
    return parser.parse_args()

def main(args=None):
    """
    Run the taxonomy classifier training pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command line arguments
    """
    if args is None:
        args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting taxonomy classifier training")
    
    # Load and prepare data
    data, sequences, y, label_encoder, num_classes = prepare_data(
        args.data_path, args.target_column, args.filter_classes
    )
    
    # Feature extraction
    feature_config = {
        'kmer': {'k': 3, 'mode': 'frequency'},
        'stats': {'features': ['gc_content', 'entropy', 'nucleotide_freq', 'z_curve']}
    }
    X = extract_features(sequences, feature_config)
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y, batch_size=args.batch_size
    )
    
    # Model configuration
    model_config = {
        'model_type': args.model_type,
        'input_channels': 1,
        'dropout': args.dropout
    }
    
    # Training configuration
    training_config = {
        'lr': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5
    }
    
    # Train model
    model, trainer, history, final_metrics = train_model(
        train_loader, val_loader, test_loader,
        num_classes=num_classes,
        input_size=X.shape[1],
        model_config=model_config,
        training_config=training_config,
        model_dir=args.model_dir
    )
    
    # Save results
    save_results(args.model_dir, label_encoder, history)
    
    logger.info("Training complete!")
    return model, trainer, history, final_metrics

if __name__ == "__main__":
    main()