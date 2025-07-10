import argparse
import json
import os
import re
import time
from typing import Any

import numpy as np
import torch

from dataset.utils import info

from models.utils.model_factory import create_model

import logging
from feature_extraction.main import SequenceCoder
        


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

    # read input csv file
    input_csv = hparams.get("input_csv", None)
    if input_csv is None:
        raise ValueError("input_csv must be provided in hyperparameters.")
    if not os.path.exists(input_csv):
        raise ValueError(f"Input CSV file {input_csv} does not exist.")
    info(f"Using input CSV file: {input_csv}")
    output_csv = hparams.get("output_csv", "embeddings.tsv")
    info(f"Output embeddings will be saved to: {output_csv}")

    # read the CSV file into a pandas DataFrame
    import pandas as pd
    df = pd.read_table(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV file {input_csv} is empty.")
    sequence_column = hparams.get("sequence_column", "sequence")
    if sequence_column not in df.columns:
        raise ValueError(f"Input CSV file {input_csv} does not contain the required column '{sequence_column}'.")
    
    if hparams.get("k", None) is None and hparams.get("bits", None) is None:
        raise ValueError("Either 'k' or 'bits' must be specified in hyperparameters.")
    if hparams.get("k", None) is None and hparams.get("bits", None) == 0:
        # create 4rm encoding of the sequences in sequence_column
        # first filter by sequence length
        seq_len_filter = hparams.get("seq_len_filter", 313)
        df = df[df[sequence_column].str.len() == seq_len_filter]
        if df.empty:
            raise ValueError(f"No sequences found with length {seq_len_filter} in column '{sequence_column}'.")
        info(f"Filtering sequences by length {seq_len_filter} in column '{sequence_column}'.")
        # create 4rm encoding
        encoder = SequenceCoder()
        df_coded = encoder.coding_one_hot_4rowMatrix_optimized(df[sequence_column].tolist())
    else:
        raise ValueError("Only developed for 4rm encoding.")
             
    from_checkpoint_path = hparams.get("from_checkpoint_path", None)
    if from_checkpoint_path is None:
        raise ValueError("If transfer_learning is True, from_checkpoint_path must be given.")
    else:
        checkpoint = torch.load(from_checkpoint_path, map_location=device)
    info(f"Loading model weights from {from_checkpoint_path}")
        
    if seq_len_filter != checkpoint['model_config']['sequence_length']:
        raise ValueError(f"Checkpoint sequence length {checkpoint['model_params']['sequence_length']} does not match "
                         f"the sequence length {seq_len_filter} in the input CSV file.")

    # Create model using factory
    model_params: dict[str, Any] = {
        "input_size": seq_len_filter,
        "output_size": checkpoint['model_config']['output_size'],
    }

    # Add model-specific parameters based on model type
    if hparams.get("model_type", "basic") == "basic":
        model_params["hidden_size"] = hparams.get("hidden_size", taxo_data_loaders.data_length // 2)
        raise ValueError("Basic model is not implemented yet. Use a different model type.")

    elif hparams.get("model_type") == "enhanced_mlp":
        model_params["hidden_sizes"] = hparams.get("hidden_sizes", [256, 128])
        model_params["dropout"] = hparams.get("dropout", 0.2)
        model_params["use_batch_norm"] = hparams.get("use_batch_norm", True)
        raise ValueError("Enhanced MLP model is not implemented yet. Use a different model type.")

    elif hparams.get("model_type") == "cnn":
        model_params["kernel_sizes"] = hparams.get("kernel_sizes", [3, 5, 7])
        model_params["num_filters"] = hparams.get("num_filters", [64, 128, 256])
        model_params["fc_sizes"] = hparams.get("fc_sizes", [512, 256])
        model_params["dropout"] = hparams.get("dropout", 0.3)
        raise ValueError("CNN model is not implemented yet. Use a different model type.")

    elif hparams.get("model_type") == "nanni_cnn1":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)
        model_params["hidden_size"] = hparams.get("hidden_size", 8)

    elif hparams.get("model_type") == "nanni_cnn2":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)
        model_params["hidden_size"] = hparams.get("hidden_size", 1024)

    elif hparams.get("model_type") == "nanni_att":
        model_params["sequence_length"] = hparams.get("sequence_length", 313)
        raise ValueError("Nanni attention model is not implemented yet. Use a different model type.")

    elif hparams.get("model_type") == "bert":
        model_params["vocab_size"] = hparams.get("vocab_size", 5)
        model_params["max_length"] = hparams.get("max_length", 512)
        model_params["hidden_size"] = hparams.get("hidden_size", 256)
        model_params["num_layers"] = hparams.get("num_layers", 6)
        model_params["num_heads"] = hparams.get("num_heads", 8)
        model_params["dropout"] = hparams.get("dropout", 0.3)
        model_params["classifier_hidden_size"] = hparams.get("classifier_hidden_size", 256)
        raise ValueError("BERT model is not implemented yet. Use a different model type.")

    # Add experiment identifier to model name
    exp_id = hparams.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_params["name"] = f"{hparams.get('model_type', 'basic')}_{exp_id}"

    # Create model
    model = create_model(model_type=hparams.get("model_type", "basic"), **model_params).to(device)

    if hparams.get("model_type", "basic") == "nanni_cnn1":
        # check that checkpoint['model_name'] has the word 'nanni_cnn1'
        if not re.search(r'nanni_cnn1', checkpoint['model_name']):
            raise ValueError(f"Checkpoint model name {checkpoint['model_name']} does not match expected 'nanni_cnn1'")
        # transfer only the cnn1 part of the model
        new_state_dict = {
            "weight": checkpoint["model_state_dict"]["conv1.weight"],
            "bias": checkpoint["model_state_dict"]["conv1.bias"]
            }
        model.conv1.load_state_dict(new_state_dict)
    elif hparams.get("model_type", "basic") == "nanni_cnn2":
        # check that checkpoint['model_name'] has the word 'nanni_cnn2'
        if not re.search(r'nanni_cnn2', checkpoint['model_name']):
            raise ValueError(f"Checkpoint model name {checkpoint['model_name']} does not match expected 'nanni_cnn2'")
        # transfer only the cnn2 part of the model
        new_state_dict_conv1 = {
            "weight": checkpoint["model_state_dict"]["conv1.weight"],
            "bias": checkpoint["model_state_dict"]["conv1.bias"]
            }
        model.conv1.load_state_dict(new_state_dict)
        new_state_dict_conv2 = {
            "weight": checkpoint["model_state_dict"]["conv2.weight"],
            "bias": checkpoint["model_state_dict"]["conv2.bias"]
            }
        model.conv2.load_state_dict(new_state_dict_conv2)
    # Set up training components

    model.to(device)
    
    # convert DataFrame to tensor
    df_coded_tensor = torch.tensor(df_coded, dtype=torch.float32).to(device)
    if df_coded_tensor.dim() == 3:  # [batch, 4, 313]
        df_coded_tensor = df_coded_tensor.unsqueeze(1)  # [batch, 1, 4, 313]
    
    df_embedded = model.return_embedding(df_coded_tensor)  # Add channel dimension
    
    # convert the tensor of four dimensions to a two-dimensional tensor
    df_embedded = df_embedded.view(df_embedded.size(0), -1)

    from sklearn.decomposition import PCA
    # perform a PCA transformation on the embeddings
    pca = PCA(n_components=hparams.get("pca_components", 2))
    df_embedded_pca = pca.fit_transform(df_embedded.detach().numpy())

    # add the PCA components to the DataFrame
    df[f'pca_{hparams.get("pca_components", 2)}_1'] = df_embedded_pca[:, 0]
    df[f'pca_{hparams.get("pca_components", 2)}_2'] = df_embedded_pca[:, 1]

    return df



def main():
    # check_available_devices()
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Get embedding from taxonomy classification models")
    parser.add_argument(
        "--config", type=str, default="PCA/hparams_PCA_example.json", help="Path to hyperparameters JSON file"
    )
    
    args = parser.parse_args()

    # Load hyperparameters
    with open(args.config, "r") as f:
        info(f"Using configuration file {args.config}")
        hparams = json.load(f)

        
    # Log the terminal output to a file
    log_to_file = hparams.get("log_file", False)
    exp_id = hparams.get("experiment_id", time.strftime("%Y%m%d-%H%M%S"))
    model_type = hparams.get('model_type', 'basic')
    label_column_name = hparams['label_column_name']
    k = hparams.get('k')
    bits = hparams.get('bits')
    if k:
        run_name = f"{model_type}_{exp_id}_{label_column_name}_k{k}_embeddings"
    else:
        run_name = f"{model_type}_{exp_id}_{label_column_name}_bits{bits}_embeddings"

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

    result.to_csv(hparams.get("output_csv", "embeddings.tsv"), sep="\t", index=False)
    # Print timing information
    seconds = time.time() - t0
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    info(f"Done! Elapsed time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
