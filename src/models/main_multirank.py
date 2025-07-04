import argparse
import json
import os
from pathlib import Path
import torch
from models.utils.model_factory import create_model
from models.training.multirank_trainer import HierarchicalTrainer
from dataset.hierarchical_dataset import HierarchicalDataset
from dataset.utils import info

def main():
    parser = argparse.ArgumentParser(description='Multi-Rank Taxonomy Classification')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the parquet dataset directory (e.g., data/parquets/filtered_ranks)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    info(f"Loaded configuration from {args.config}")

    # Override config with command line arguments
    config['checkpoint_dir'] = args.checkpoint_dir
    config['log_dir'] = args.log_dir

    # Set random seed
    torch.manual_seed(config.get('seed', 123))

    # Create data loaders
    dataset = HierarchicalDataset(
        parquets_path=Path(args.data_path),
        k=config.get('k'),
        bits=config.get('bits')
    )
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

    # Create model
    model = create_model(
        model_type=config['model_type'],
        input_size=input_size,
        shared_hidden_sizes=config.get('shared_hidden_sizes', [512, 256]),
        level_specific_sizes=config.get('level_specific_sizes'),
        dropout=config.get('dropout', 0.3),
        use_confidence_weighting=config.get('use_confidence_weighting', True)
    )
    info(f"Created {model.name} with {sum(p.numel() for p in model.parameters())} parameters")

    # Select loss function
    if config['model_type'] == 'cascade_hierarchical':
        from src.models.architectures.cascade_hierarchical_model import CascadeLoss
        criterion = CascadeLoss(
            level_weights=config.get('level_weights'),
            cascade_weight=config.get('cascade_weight', 0.1),
            confidence_weight=config.get('confidence_weight', 0.05)
        )
    elif config['model_type'] == 'gnn_hierarchical':
        from src.models.architectures.gnn_hierarchical_model import GNHLoss
        criterion = GNHLoss(
            level_weights=config.get('level_weights'),
            graph_weight=config.get('graph_weight', 0.1),
            consistency_weight=config.get('consistency_weight', 0.05)
        )
    else:
        from src.models.architectures.hierarchical_model import HierarchicalLoss
        criterion = HierarchicalLoss(
            level_weights=config.get('level_weights'),
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

    # Trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = HierarchicalTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=config.get('log_dir'),
        checkpoint_dir=config.get('checkpoint_dir'),
        class_names_per_level=dataset.labels_names_per_level
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
    info(f"Best validation accuracy: {max(history['val_accuracies']):.4f}")

    # Evaluate on test set
    test_loss, test_accuracies, test_metrics, hierarchical_acc = trainer.evaluate(
        test_loader, 
        epoch=len(history['train_losses']), 
        prefix='test'
    )
    info("Test Results:")
    for level, acc in test_accuracies.items():
        info(f"  {level}: {acc:.4f}")
    info(f"  Hierarchical accuracy: {hierarchical_acc:.4f}")

if __name__ == "__main__":
    main() 