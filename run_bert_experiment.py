#!/usr/bin/env python3
"""
Script to run BERT experiments for single-rank taxonomic classification.
This script uses the existing training infrastructure.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.utils.model_factory import create_model
from dataset.taxo_dataset import TaxoDataset
from dataset.taxo_dataloaders import create_bert_dataloader
from models.training.singlerank_trainer import Trainer
from constants.taxonomy_labels import TAXONOMY_LABELS


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_datasets(config: dict, data_path: str) -> tuple[TaxoDataset, TaxoDataset, TaxoDataset]:
    """Create training, validation, and test datasets."""
    # Get configuration parameters
    label_column = config['label_column_name']
    k = config.get('k')
    bits = config.get('bits')
    seq_len_filter = config.get('seq_len_filter')
    max_rows = config.get('max_rows', 1.0)
    seed = config.get('seed', 123)
    
    # Ensure k or bits is provided
    if k is None and bits is None:
        raise ValueError("Either 'k' or 'bits' must be specified in the config")
    
    # Create full dataset
    full_dataset = TaxoDataset(
        parquets_path=Path(data_path),
        label_column_name=label_column,
        k=k,  # type: ignore
        bits=bits,  # type: ignore
        seq_len_filter=seq_len_filter
    )
    
    # Limit dataset size if specified
    if max_rows < 1.0:
        import random
        random.seed(seed)
        total_size = len(full_dataset)
        target_size = int(total_size * max_rows)
        indices = random.sample(range(total_size), target_size)
        full_dataset._filter_indexes = indices
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    import random
    random.seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create train dataset
    train_dataset = TaxoDataset(
        parquets_path=Path(data_path),
        label_column_name=label_column,
        k=k,  # type: ignore
        bits=bits,  # type: ignore
        seq_len_filter=seq_len_filter
    )
    train_dataset._filter_indexes = train_indices
    
    # Create validation dataset
    val_dataset = TaxoDataset(
        parquets_path=Path(data_path),
        label_column_name=label_column,
        k=k,  # type: ignore
        bits=bits,  # type: ignore
        seq_len_filter=seq_len_filter
    )
    val_dataset._filter_indexes = val_indices
    
    # Create test dataset
    test_dataset = TaxoDataset(
        parquets_path=Path(data_path),
        label_column_name=label_column,
        k=k,  # type: ignore
        bits=bits,  # type: ignore
        seq_len_filter=seq_len_filter
    )
    test_dataset._filter_indexes = test_indices
    
    return train_dataset, val_dataset, test_dataset


def run_bert_experiment(config: dict, data_path: str, output_dir: str, device: torch.device) -> dict:
    """Run BERT experiment."""
    print(f"Starting BERT experiment with model type: {config.get('model_type', 'bert')}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config, data_path)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_labels}")
    
    # Create model
    print("Creating model...")
    model_config = config.copy()
    model_config['output_size'] = train_dataset.num_labels
    
    model = create_model(
        model_type=model_config['model_type'],
        **{k: v for k, v in model_config.items() if k != 'model_type'}
    )
    
    print(f"Model created: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    batch_size = config.get('batch_size', 16)
    train_loader = create_bert_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = create_bert_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = create_bert_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.0001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Setup scheduler
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.get('epochs', 30)
            )
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.get('scheduler_patience', 3),
                factor=config.get('scheduler_factor', 0.5)
            )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,  # type: ignore
        log_dir=os.path.join(output_dir, 'logs'),
        checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
        class_names=train_dataset.labels_names
    )
    
    # Train model
    print("Starting training...")
    epochs = config.get('epochs', 30)
    patience = config.get('patience', 5)
    fast_mode = config.get('fast_mode', True)
    eval_frequency = config.get('eval_frequency', 5)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        patience=patience,
        save_best=True,
        fast_mode=fast_mode,
        eval_frequency=eval_frequency
    )
    
    # Print final results
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Best test accuracy: {max(history['test_accuracy']):.4f}")
    print(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
    
    return {
        'best_val_accuracy': max(history['val_accuracy']),
        'best_test_accuracy': max(history['test_accuracy']),
        'final_test_accuracy': history['test_accuracy'][-1],
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description="Run BERT experiment for single-rank taxonomic classification")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet data directory")
    parser.add_argument("--output_dir", type=str, default="runs/bert_experiment", 
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
    else:
        device = torch.device(args.device)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    result = run_bert_experiment(config, args.data_path, str(output_dir), device)
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 