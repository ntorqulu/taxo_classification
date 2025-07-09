import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from dataset.cached_dataframe import CachedDataFrame
from dataset.utils import info, warn, get_base_parquets_path, DEFAULT_DATASET_NAME
from dataset.taxo_dataloaders import TaxoDataLoaders
from models.training.singlerank_trainer import Trainer

from models.utils.model_factory import create_model

import logging


def init_device(seed: int = 42) -> torch.device:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon (M1/M2/M3/M4)
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

def log_label_stats(taxo_data_loaders: TaxoDataLoaders):
    # Validate labels

    info(f"Validating labels")
    results = taxo_data_loaders.compare_label_values()
    for ds_name in results.keys():
        ds_results = results[ds_name]
        if ds_results is None or (not ds_results['missing'] and not ds_results['unknown']):
            info(f"Label values in {ds_name} dataset are valid")
            continue

        if ds_results['missing']:
            warn(f"Missing label values found in {ds_name} dataset: {ds_results['missing']}")

        if ds_results['unknown']:
            warn(f"Unknown label values found in {ds_name} dataset: {ds_results['unknown']}")

    # Log a summary of the label values and stratification

    label_stats = taxo_data_loaders.get_label_stats()

    summary = {}
    summary_total = {'train': (0, 0.0), 'eval': (0, 0.0), 'test': (0, 0.0)}
    for ds_name in ('train', 'eval', 'test'):
        for name, (count, pct) in label_stats[ds_name].items():
            if name not in summary:
                summary[name] = {'train': (0, 0.0), 'eval': (0, 0.0), 'test': (0, 0.0)}
            summary[name][ds_name] = (count, pct)
            summary_total[ds_name] = (summary_total[ds_name][0] + count, summary_total[ds_name][1] + pct)
    summary[' '] = summary_total
    name_max_len = max(len(v[0]) for v in label_stats[ds_name].items())
    info(f"{' ' * (name_max_len + 2)} {'train':^15}  {'eval':^15}  {'test':^15}")
    for name in summary.keys():
        line = f"{name:<{name_max_len + 2}} "
        for ds in ('train', 'eval', 'test'):
            line += f"{summary[name][ds][0]:>7} {100 * summary[name][ds][1]:>6.2f}% "
        info(line)

    # Log the dataset length and sequence lengths

    info(f"Full dataset - {CachedDataFrame.get_length()}")
    info(f"Full dataset - Min sequence length: {CachedDataFrame.get_min_sequence_len()}")
    info(f"Full dataset - Max sequence length: {CachedDataFrame.get_max_sequence_len()}")

    info(f"Filtered dataset - {taxo_data_loaders.dataset_length}")
    info(f"Filtered dataset - Min sequence length: {taxo_data_loaders.min_sequence_len}")
    info(f"Filtered dataset - Max sequence length: {taxo_data_loaders.max_sequence_len}")



def run_experiment(hparams: dict) -> dict:
    """
    Run a model training experiment.

    Args:
        hparams: Hyperparameters for the experiment

    Returns:
        Dictionary with training results
    """
    # Set up device
    device = init_device(hparams.get("seed", 42))

    # Load data
    parquets_path = hparams["parquets_path"] if hparams["parquets_path"] else get_base_parquets_path()
    dataset_name = hparams["dataset_name"]
    min_cardinality_filters = hparams.get("min_cardinality_filters", None)
    if min_cardinality_filters is not None:
        info(f"Filtering by cardinality: {min_cardinality_filters}")

    taxo_data_loaders = TaxoDataLoaders(
        parquets_path=Path(parquets_path) / dataset_name,
        label_column_name=hparams["label_column_name"],
        k=hparams["k"],
        bits=hparams["bits"],
        batch_size=hparams["batch_size"],
        max_rows=hparams["max_rows"],
        seq_len_filter=hparams.get("seq_len_filter", None),
        min_cardinality_filters=min_cardinality_filters or {},
        use_bert_collate=(hparams.get("model_type", "basic") == "bert")
    )

    log_label_stats(taxo_data_loaders)
    info("Level cardinalities:")
    CachedDataFrame.log_level_cardinalities()

    # Create model using factory
    model_params: dict[str, Any] = {
        "input_size": taxo_data_loaders.data_length,
        "output_size": taxo_data_loaders.num_labels,
    }

    # Add model-specific parameters based on model type
    if hparams.get("model_type", "basic") == "basic":
        model_params["hidden_size"] = hparams.get("hidden_size", taxo_data_loaders.data_length // 2)

    elif hparams.get("model_type") == "enhanced_mlp":
        model_params["hidden_sizes"] = hparams.get("hidden_sizes", [256, 128])
        model_params["dropout"] = hparams.get("dropout", 0.2)
        model_params["use_batch_norm"] = hparams.get("use_batch_norm", True)

    elif hparams.get("model_type") == "cnn":
        model_params["kernel_sizes"] = hparams.get("kernel_sizes", [3, 5, 7])
        model_params["num_filters"] = hparams.get("num_filters", [64, 128, 256])
        model_params["fc_sizes"] = hparams.get("fc_sizes", [512, 256])
        model_params["dropout"] = hparams.get("dropout", 0.3)

    elif hparams.get("model_type") == "nanni_cnn1":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)
        model_params["hidden_size"] = hparams.get("hidden_size", 8)

    elif hparams.get("model_type") == "nanni_cnn2":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)
        model_params["hidden_size"] = hparams.get("hidden_size", 1024)

    elif hparams.get("model_type") == "nanni_att":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)

    elif hparams.get("model_type") == "bert":
        model_params["vocab_size"] = hparams.get("vocab_size", 5)
        model_params["max_length"] = hparams.get("max_length", 512)
        model_params["hidden_size"] = hparams.get("hidden_size", 256)
        model_params["num_layers"] = hparams.get("num_layers", 6)
        model_params["num_heads"] = hparams.get("num_heads", 8)
        model_params["dropout"] = hparams.get("dropout", 0.3)
        model_params["classifier_hidden_size"] = hparams.get("classifier_hidden_size", 256)

    # Add experiment identifier to model name
    exp_id = hparams.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_params["name"] = f"{hparams.get('model_type', 'basic')}_{exp_id}"

    # Create model
    model = create_model(model_type=hparams.get("model_type", "basic"), **model_params).to(device)

    # Set up training components

    # Loss criterion
    balancing_method = hparams.get('balancing_method', 'none')
    if balancing_method == 'loss_soft':
        info("Using balanced loss criterion (soft)")
        weight = taxo_data_loaders.get_label_weights()
        weight = weight.to(device)
    elif balancing_method == 'loss_strong':
        info("Using balanced loss criterion (strong")
        weight = taxo_data_loaders.get_label_weights(strong=True)
        weight = weight.to(device)
    else:
        info(f"Not using balanced loss criterion: {balancing_method}")
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # Select optimizer
    if hparams.get("optimizer", "adam") == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hparams.get("learning_rate", 0.001), weight_decay=hparams.get("weight_decay", 0)
        )
    elif hparams.get("optimizer") == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams.get("learning_rate", 0.01),
            momentum=hparams.get("momentum", 0.9),
            weight_decay=hparams.get("weight_decay", 1e-4),
        )
    else:
        raise ValueError(f"Unknown optimizer: {hparams.get('optimizer')}")

    # Set up scheduler
    scheduler = None
    if hparams.get("use_scheduler", False):
        if hparams.get("scheduler", "plateau") == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=hparams.get("scheduler_patience", 3),
                factor=hparams.get("scheduler_factor", 0.5),
            )
        elif hparams.get("scheduler") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams.get("epochs", 15))
        elif hparams.get("scheduler") == "by_steps":
            every_n_epochs = hparams.get("every_n_epochs", 50)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // every_n_epochs)
            )

    # Set up trainer
    run_name = (
        f"{model.name}_{hparams['label_column_name']}_k{hparams['k']}"
        if hparams["k"]
        else f"{model.name}_{hparams['label_column_name']}_bits{hparams['bits']}"
    )
    log_dir = os.path.join("runs", run_name)
    if hparams.get("from_checkpoint", False):
        from_checkpoint_path = hparams.get("from_checkpoint_path", None)
        if from_checkpoint_path is None:
            raise ValueError("If from_checkpoint is True, from_checkpoint_path must be given.")
    else:
        from_checkpoint_path = None
        
    checkpoint_dir = os.path.join("checkpoints", run_name)

    # Get class names dynamically from the dataset instead of hardcoded ones
    class_names = taxo_data_loaders.class_names
    info(f"Using {len(class_names)} labels for {hparams['label_column_name']} classification.")
    info(f"Class names: {', '.join(class_names)}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        class_names=class_names,
        from_checkpoint=hparams.get("from_checkpoint", False),
        from_checkpoint_path=from_checkpoint_path,
    )

    # Train model
    history = trainer.train(
        train_loader=taxo_data_loaders.train_loader,
        val_loader=taxo_data_loaders.eval_loader,
        test_loader=taxo_data_loaders.test_loader,
        epochs=hparams.get("epochs", 15),
        patience=hparams.get("patience", 5),
        save_best=True,
        fast_mode=hparams.get("fast_mode", False),
        eval_frequency=hparams.get("eval_frequency", 1),
    )

    return {"model": model, "history": history, "run_name": run_name}


def check_available_devices():
    info(f"PyTorch version: {torch.__version__}")
    info(f"CUDA available: {torch.cuda.is_available()}")
    # FOR APPLE SILICON
    info(f"MPS available: {torch.backends.mps.is_available()}")
    info(f"MPS backend enabled: {torch.backends.mps.is_built()}")
    # Check for XPU (Intel GPU) support
    if hasattr(torch, "xpu"):
        info(f"XPU available: {torch.xpu.is_available()}")
    else:
        info("XPU support not available in this PyTorch version.")


def main():
    check_available_devices()
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train taxonomy classification models")
    parser.add_argument(
        "--config", type=str, default="hyperparams/kmer_hparams.json", help="Path to hyperparameters JSON file"
    )
    
    args = parser.parse_args()

    # Load hyperparameters
    with open(args.config, "r") as f:
        info(f"Using configuration file {args.config}")
        hparams = json.load(f)

    # Set default dataset path if not provided
    if hparams["parquets_path"] == "":
        hparams["parquets_path"] = get_base_parquets_path()
        
    # Log the terminal output to a file
    log_to_file = hparams.get("log_file", False)
    exp_id = hparams.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_type = hparams.get('model_type', 'basic')
    label_column_name = hparams['label_column_name']
    k = hparams.get('k')
    bits = hparams.get('bits')
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
    
    info(f"Using configuration file {args.config}")

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
