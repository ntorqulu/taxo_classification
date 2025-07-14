import argparse
import time

import numpy as np
import torch
from compute_attention import compute_attention_by_predicted_class, compute_attention_of_a_sequence
from dataset.cached_dataframe import CachedDataFrame
from dataset.taxo_dataloaders import TaxoDataLoaders
from dataset.utils import info, warn


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

    info("Validating labels")
    results = taxo_data_loaders.compare_label_values()
    for ds_name in results.keys():
        ds_results = results[ds_name]
        if ds_results is None or (not ds_results["missing"] and not ds_results["unknown"]):
            info(f"Label values in {ds_name} dataset are valid")
            continue

        if ds_results["missing"]:
            warn(f"Missing label values found in {ds_name} dataset: {ds_results['missing']}")

        if ds_results["unknown"]:
            warn(f"Unknown label values found in {ds_name} dataset: {ds_results['unknown']}")

    # Log a summary of the label values and stratification

    label_stats = taxo_data_loaders.get_label_stats()

    summary = {}
    summary_total = {"train": (0, 0.0), "eval": (0, 0.0), "test": (0, 0.0)}
    for ds_name in ("train", "eval", "test"):
        for name, (count, pct) in label_stats[ds_name].items():
            if name not in summary:
                summary[name] = {"train": (0, 0.0), "eval": (0, 0.0), "test": (0, 0.0)}
            summary[name][ds_name] = (count, pct)
            summary_total[ds_name] = (summary_total[ds_name][0] + count, summary_total[ds_name][1] + pct)
    summary[" "] = summary_total
    name_max_len = max(len(v[0]) for v in label_stats[ds_name].items())
    info(f"{' ' * (name_max_len + 2)} {'train':^15}  {'eval':^15}  {'test':^15}")
    for name in summary.keys():
        line = f"{name:<{name_max_len + 2}} "
        for ds in ("train", "eval", "test"):
            line += f"{summary[name][ds][0]:>7} {100 * summary[name][ds][1]:>6.2f}% "
        info(line)

    # Log the dataset length and sequence lengths

    info(f"Full dataset - {CachedDataFrame.get_length()}")
    info(f"Full dataset - Min sequence length: {CachedDataFrame.get_min_sequence_len()}")
    info(f"Full dataset - Max sequence length: {CachedDataFrame.get_max_sequence_len()}")

    info(f"Filtered dataset - {taxo_data_loaders.dataset_length}")
    info(f"Filtered dataset - Min sequence length: {taxo_data_loaders.min_sequence_len}")
    info(f"Filtered dataset - Max sequence length: {taxo_data_loaders.max_sequence_len}")


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
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint to load the model")
    parser.add_argument("--sequence", type=str, required=False, help="Sequence of 313 length")

    args = parser.parse_args()

    # Track timing
    t0 = time.time()
    info("Starting")

    if args.sequence:
        assert len(args.sequence) == 313
        # Get attention of a sequence
        compute_attention_of_a_sequence(args.sequence, args.checkpoint_path)

    else:
        # Attention for predicted class
        compute_attention_by_predicted_class(args.checkpoint_path)

    # Print timing information
    seconds = time.time() - t0
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    info(f"Done! Elapsed time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
