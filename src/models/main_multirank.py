import argparse
import json
import os
import time
from pathlib import Path
import torch
import logging
from models.utils.model_factory import create_model
from models.training.multirank_trainer import HierarchicalTrainer
from dataset.hierarchical_dataset import HierarchicalDataset
from dataset.utils import info, get_base_parquets_path, DEFAULT_DATASET_NAME

def main():
    parser = argparse.ArgumentParser(description='Multi-Rank Taxonomy Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    info(f"Loaded configuration from {args.config}")

    # Handle data path from config file
    dataset_name = config.get('dataset_name', DEFAULT_DATASET_NAME)
    if config.get('parquets_path'):
        # Use path from config file
        data_path = Path(config['parquets_path']) / dataset_name
    else:
        # Use default path
        data_path = get_base_parquets_path() / dataset_name
    
    # Log the terminal output to a file
    log_to_file = config.get("log_file", False)
    exp_id = config.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_type = config.get('model_type', 'basic')
    label_column_name = config['label_column_name']
    k = config.get('k')
    bits = config.get('bits')
    if k:
        run_name = f"{model_type}_{exp_id}_{label_column_name}_k{k}"
    else:
        run_name = f"{model_type}_{exp_id}_{label_column_name}_bits{bits}"

    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_filename = os.path.join("logs", f"{run_name}.log")
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    info(f"Using data path: {data_path}")
    info(f"Dataset name: {dataset_name}")

    # Set random seed
    torch.manual_seed(config.get('seed', 123))

    # Create data loaders
    dataset = HierarchicalDataset(
        parquets_path=data_path,
        k=config.get('k'),
        bits=config.get('bits')
    )
    
    # Log dataset information
    info(f"Dataset loaded with {len(dataset)} samples")
    info(f"Number of classes per level: {dataset.num_labels_per_level}")
    info(f"Labels per level: {dataset.labels_names_per_level}")
    
    # Determine target levels based on label_column_name
    label_column_name = config.get('label_column_name', 'order_name')
    all_levels = ['kingdom_name', 'phylum_name', 'class_name', 'order_name', 'family_name', 'genus_name', 'species_name']
    
    # Find the index of the target level
    try:
        target_level_index = all_levels.index(label_column_name)
        target_levels = all_levels[:target_level_index + 1]  # Include up to the target level
        info(f"Training on levels up to {label_column_name}: {target_levels}")
    except ValueError:
        # If label_column_name is not in the predefined levels, use all available levels from dataset
        target_levels = list(dataset.num_labels_per_level.keys())
        info(f"Training on all available levels from dataset: {target_levels}")
    
    # Verify that all target levels are available in the dataset
    available_levels = list(dataset.num_labels_per_level.keys())
    missing_levels = [level for level in target_levels if level not in available_levels]
    if missing_levels:
        info(f"Warning: Target levels {missing_levels} are not available in the dataset")
        info(f"Available levels: {available_levels}")
        # Use only the levels that are available in the dataset
        target_levels = [level for level in target_levels if level in available_levels]
        info(f"Adjusted target levels: {target_levels}")
    
    from torch.utils.data import DataLoader, random_split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4)

    # Get input size from dataset
    sample_batch = next(iter(train_loader))
    input_size = sample_batch['features'].shape[1]
    info(f"Input size: {input_size}")

    # Add experiment identifier to model name (similar to single-rank)
    exp_id = config.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_name = f"{config.get('model_type', 'hierarchical')}_{exp_id}"

    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_size=input_size,
        shared_hidden_sizes=config.get('shared_hidden_sizes', [512, 256]),
        level_specific_sizes=config.get('level_specific_sizes'),
        num_classes_per_level=dataset.num_labels_per_level,
        target_levels=target_levels,
        dropout=config.get('dropout', 0.3),
        use_confidence_weighting=config.get('use_confidence_weighting', True),
        name=model_name
    )
    info(f"Created {model.name} with {sum(p.numel() for p in model.parameters())} parameters")

    # Select loss function
    if config['model_type'] == 'cascade_hierarchical':
        from src.models.architectures.cascade_hierarchical_model import CascadeLoss
        criterion = CascadeLoss(
            level_weights=config.get('level_weights'),
            target_levels=target_levels,
            cascade_weight=config.get('cascade_weight', 0.1),
            confidence_weight=config.get('confidence_weight', 0.05)
        )
    elif config['model_type'] == 'gnn_hierarchical':
        from src.models.architectures.gnn_hierarchical_model import GNHLoss
        criterion = GNHLoss(
            level_weights=config.get('level_weights'),
            target_levels=target_levels,
            graph_weight=config.get('graph_weight', 0.1),
            consistency_weight=config.get('consistency_weight', 0.05)
        )
    else:
        from src.models.architectures.hierarchical_model import HierarchicalLoss
        criterion = HierarchicalLoss(
            level_weights=config.get('level_weights'),
            target_levels=target_levels,
            loss_type=config.get('loss_type', 'cross_entropy')
        )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # Scheduler
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 3)
        )

    # Set up run name and directories (similar to single-rank)
    encoding_suffix = f"k{config['k']}" if config.get('k') else f"bits{config['bits']}"
    run_name = f"{model.name}_multirank_{dataset_name}_{encoding_suffix}"
    log_dir = os.path.join("runs", run_name)
    checkpoint_dir = os.path.join("checkpoints", run_name)

    # Trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = HierarchicalTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        class_names_per_level=dataset.labels_names_per_level,
        target_levels=target_levels
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config.get('epochs', 50),
        patience=config.get('patience', 10),
        save_best=True,
        fast_mode=config.get('fast_mode', False),
        eval_frequency=config.get('eval_frequency', 1)
    )

    info("Training completed!")
    if 'val_accuracies' in history and history['val_accuracies']:
        info(f"Best validation accuracy: {max(history['val_accuracies']):.4f}")

    # Evaluate on test set
    test_loss, test_accuracies, test_metrics, hierarchical_acc = trainer.evaluate(
        test_loader, 
        epoch=len(history.get('train_losses', [1])), 
        prefix='test'
    )
    info("Test Results:")
    for level, acc in test_accuracies.items():
        info(f"  {level}: {acc:.4f}")
    info(f"  Hierarchical accuracy: {hierarchical_acc:.4f}")

    # Final model info (similar to single-rank)
    info(f"Model: {model.name}")
    info(f"TensorBoard logs: {os.path.join('runs', run_name)}")
    info(f"Model checkpoints: {os.path.join('checkpoints', run_name)}")

if __name__ == "__main__":
    main() 