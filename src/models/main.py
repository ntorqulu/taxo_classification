import time
import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import info, get_default_dataset_path
from dataset.taxo_dataloaders import TaxoDataLoaders
from constants.taxonomy_labels import get_class_names
from models.utils.model_factory import create_model
from models.training.trainer import Trainer
from models.results import Results, plot_results

def init_device(seed: int = 42) -> torch.device:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon (M1/M2/M3/M4)
        torch.backends.mps.enable_prior_normalization = True
        d = torch.device("mps")
        info("Using Apple Metal Performance Shaders (MPS) backend")
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.manual_seed(seed)
        d = torch.device("xpu")
    else:
        d = torch.device("cpu")
    info(f"Device: {d}")
    return d

def run_experiment(hparams: dict) -> dict:
    """
    Run a model training experiment.
    
    Args:
        hparams: Hyperparameters for the experiment
        
    Returns:
        Dictionary with training results
    """
    # Set up device
    device = init_device(hparams.get('seed', 42))
    
    # Load data
    taxo_path = hparams['taxo_path'] if hparams['taxo_path'] else get_default_dataset_path()
    taxo_data_loaders = TaxoDataLoaders(
        taxo_path=taxo_path,
        label_column_name=hparams["label_column_name"],
        k=hparams["k"],
        bits=hparams["bits"],
        batch_size=hparams["batch_size"],
        max_rows=hparams["max_rows"]
    )
    
    # Create model using factory
    model_params = {
        'input_size': taxo_data_loaders.data_length,
        'output_size': taxo_data_loaders.num_labels,
    }
    
    # Add model-specific parameters based on model type
    if hparams.get('model_type', 'basic') == 'basic':
        model_params['hidden_size'] = hparams.get('hidden_size', taxo_data_loaders.data_length // 2)
    
    elif hparams.get('model_type') == 'enhanced_mlp':
        model_params['hidden_sizes'] = hparams.get('hidden_sizes', [256, 128])
        model_params['dropout'] = hparams.get('dropout', 0.2)
        model_params['use_batch_norm'] = hparams.get('use_batch_norm', True)
    
    elif hparams.get('model_type') == 'cnn':
        model_params['kernel_sizes'] = hparams.get('kernel_sizes', [3, 5, 7])
        model_params['num_filters'] = hparams.get('num_filters', [64, 128, 256])
        model_params['fc_sizes'] = hparams.get('fc_sizes', [512, 256])
        model_params['dropout'] = hparams.get('dropout', 0.3)
    
    elif hparams.get('model_type') == 'nanni_cnn1':
        model_params['sequence_length'] = hparams.get('sequence_length', 313)
        model_params['hidden_size'] = hparams.get('hidden_size', 8)
    
    elif hparams.get('model_type') == 'nanni_cnn2':
        model_params['sequence_length'] = hparams.get('sequence_length', 313)
        model_params['hidden_size'] = hparams.get('hidden_size', 1024)
    
    # Add experiment identifier to model name
    exp_id = hparams.get('experiment_id', time.strftime('%Y%m%d-%H%M%S'))
    model_params['name'] = f"{hparams.get('model_type', 'basic')}_{exp_id}"
    
    # Create model
    model = create_model(
        model_type=hparams.get('model_type', 'basic'),
        **model_params
    ).to(device)
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    if hparams.get('optimizer', 'adam') == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hparams.get('learning_rate', 0.001),
            weight_decay=hparams.get('weight_decay', 0)
        )
    elif hparams.get('optimizer') == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams.get('learning_rate', 0.01),
            momentum=hparams.get('momentum', 0.9),
            weight_decay=hparams.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {hparams.get('optimizer')}")
    
    # Set up scheduler
    scheduler = None
    if hparams.get('use_scheduler', False):
        if hparams.get('scheduler', 'plateau') == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', 
                patience=hparams.get('scheduler_patience', 3),
                factor=hparams.get('scheduler_factor', 0.5)
            )
        elif hparams.get('scheduler') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.get('epochs', 15)
            )
        elif hparams.get('scheduler') == 'by_steps':
            every_n_epochs = hparams.get('every_n_epochs', 50)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda=lambda epoch: 0.5 ** (epoch // every_n_epochs)
            )

    
    # Set up trainer
    run_name = f"{model.name}_{hparams['label_column_name']}_k{hparams['k']}" if hparams['k'] else f"{model.name}_{hparams['label_column_name']}_bits{hparams['bits']}"
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    
    class_names = get_class_names(hparams['label_column_name'])
    info(f"Using {len(class_names)} labels for {hparams['label_column_name']} classification.")
    
    if class_names and taxo_data_loaders.num_labels != len(class_names):
        info(f"Warning: Number of class names ({len(class_names)}) doesn't match the number of labels in the dataset ({taxo_data_loaders.num_labels}).")
        # Adjust class_names to match the number of labels
        if taxo_data_loaders.num_labels > len(class_names):
            # Extend with generic labels
            class_names.extend([f"Class_{i}" for i in range(len(class_names), taxo_data_loaders.num_labels)])
        else:
            # Truncate
            class_names = class_names[:taxo_data_loaders.num_labels]
        info(f"Adjusted class names to match dataset: {len(class_names)} labels.")
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        class_names=class_names
    )
    
    # Train model
    history = trainer.train(
        train_loader=taxo_data_loaders.train_loader,
        val_loader=taxo_data_loaders.eval_loader,
        test_loader=taxo_data_loaders.test_loader,
        epochs=hparams.get('epochs', 15),
        patience=hparams.get('patience', 5),
        save_best=True,
        fast_mode=hparams.get('fast_mode', False),
        eval_frequency=hparams.get('eval_frequency', 1)
    )
    
    return {
        'model': model,
        'history': history,
        'run_name': run_name
    }

def check_available_devices():
    info(f"PyTorch version: {torch.__version__}")
    info(f"CUDA available: {torch.cuda.is_available()}")
    # FOR APPLE SILICON
    info(f"MPS available: {torch.backends.mps.is_available()}")
    info(f"MPS backend enabled: {torch.backends.mps.is_built()}")
    # Check for XPU (Intel GPU) support
    if hasattr(torch, 'xpu'):
        info(f"XPU available: {torch.xpu.is_available()}")
    else:
        info("XPU support not available in this PyTorch version.")
    
    
def main():
    check_available_devices()
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train taxonomy classification models')
    parser.add_argument('--config', type=str, default='hparams.json', help='Path to hyperparameters JSON file')
    parser.add_argument('--model_type', type=str, choices=['basic', 'enhanced_mlp', 'cnn', 'nanni_cnn1', 'nanni_cnn2'], 
                       help='Model type to train')
    parser.add_argument('--fast', action='store_true', help='Enable fast evaluation mode')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of detailed evaluation')
    args = parser.parse_args()
    
    # Load hyperparameters
    with open(args.config, "r") as f:
        hparams = json.load(f)
        
    # Override with command line arguments if provided
    if args.model_type:
        hparams['model_type'] = args.model_type
    
    if args.fast:
        hparams['fast_mode'] = True
        info("Fast evaluation mode enabled")
    
    if args.eval_freq != 1:
        hparams['eval_frequency'] = args.eval_freq
        info(f"Detailed evaluation will run every {args.eval_freq} epochs")
        
    # Set default dataset path if not provided
    if hparams['taxo_path'] == "":
        hparams['taxo_path'] = get_default_dataset_path()
        
    # Track timing
    t0 = time.time()
    info("Starting")
    
    # Run experiment
    result = run_experiment(hparams)
    
    # Print timing information
    seconds = time.time() - t0
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    info(f"Done! Elapsed time: {minutes}m {seconds}s")
    
    # Final model info
    info(f"Model: {result['model'].name}")
    info(f"TensorBoard logs: {os.path.join('runs', result['run_name'])}")
    info(f"Model checkpoints: {os.path.join('checkpoints', result['run_name'])}")


if __name__ == "__main__":
    main()