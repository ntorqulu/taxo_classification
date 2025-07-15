import re
from pathlib import Path

import torch
from dataset.cached_dataframe import CachedDataFrame
from dataset.taxo_dataloaders import TaxoDataLoaders
from dataset.utils import get_base_parquets_path, info, warn
from feature_extraction.main import SequenceCoder
from models.architectures.nanni2024 import nanni_att
from torch.utils.data import ConcatDataset

SEQUENCE_LENGTH_ATTENTION = 313


def encode_sequence(sequence: str, encoding_type: str = "4row", k: int = 4):
    """Encode a DNA sequence based on the specified encoding type."""
    sequence_coder = SequenceCoder()

    # Pad or truncate sequence to target length
    if len(sequence) != SEQUENCE_LENGTH_ATTENTION:
        raise ValueError(f"Sequence must be of lentgh 313 and this one has lentgh {len(sequence)}")

    # Encode based on encoding type
    if encoding_type == "4row" or encoding_type.startswith("4rowmatrix"):
        encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
    elif encoding_type.startswith("kmer"):
        # Extract k from encoding_type if specified (e.g., 'kmer_3')
        if "_" in encoding_type:
            try:
                k = int(encoding_type.split("_")[1])
            except:
                pass
        encoded = sequence_coder.coding_kmer_optimized([sequence], k=k)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
    elif encoding_type.startswith("bits"):
        # Extract bits from encoding_type if specified (e.g., 'bits_2')
        bits = 4
        if "_" in encoding_type:
            try:
                bits = int(encoding_type.split("_")[1])
            except:
                pass

        # Use coding_one_hot_bit_optimized with correct parameters.
        print(f"Using bits={bits} for encoding, with sequence length {len(sequence)}")

        # The max_seq_length parameter is what determines the output size.
        encoded = sequence_coder.coding_one_hot_bit_optimized(
            [sequence],
            bits=bits,
            max_seq_length=SEQUENCE_LENGTH_ATTENTION,  # This ensures the correct output size
        )

        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
        print(f"Encoded tensor shape: {encoded_tensor.shape}")

        # Check if the size matches what's expected
        expected_size = SEQUENCE_LENGTH_ATTENTION * bits
        if encoded_tensor.numel() != expected_size:
            print(f"WARNING: Encoded tensor size {encoded_tensor.numel()} doesn't match expected size {expected_size}")
    else:
        # Default to 4-row matrix
        encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)

    return encoded_tensor


def load_model(checkpoint_path):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model configuration from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("model_type", "")

    # If model_type not in the config, try to infer from the checkpoint path.
    if not model_type:
        path = Path(checkpoint_path)
        for part in path.parts:
            if any(model_name in part.lower() for model_name in ["nanni"]):
                model_type = part.lower()
                break

    # Initialize the appropriate model based on type
    if "nanni_att" in model_type:
        model = nanni_att(
            sequence_length=model_config.get("sequence_length", 313),
            output_size=model_config.get("output_size", 16),
            num_heads=model_config.get("num_heads", 8),
            embed_dim=model_config.get("embed_dim", 64),
            hidden_size=model_config.get("hidden_size", 100),
        )
    else:
        raise ValueError(f"Failed to load model: {str(model_type)}. Only nanni_att supported")

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Return class names from top-level checkpoint if present
    class_names = checkpoint.get("class_names", None)
    return model, device, model_config, class_names


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


def load_data(hparams: dict, only_test: bool = False) -> TaxoDataLoaders:
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
        use_bert_collate=(hparams.get("model_type", "basic") == "bert"),
    )

    log_label_stats(taxo_data_loaders)
    info("Level cardinalities:")
    CachedDataFrame.log_level_cardinalities()
    if only_test:
        return taxo_data_loaders.test_loader
    else:
        combined_dataset = ConcatDataset(
            [taxo_data_loaders.train_dataset, taxo_data_loaders.test_dataset, taxo_data_loaders.eval_dataset]
        )
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=30, shuffle=True, num_workers=4)
        return combined_loader


def get_model_by_name(model_name):
    """Get model configuration by name."""
    models = list_models()
    for model in models:
        if model["name"] == model_name:
            return model
    return None


CHECKPOINTS_DIR = Path("../Results")  # Adjust path as needed


def list_models():
    """Scan checkpoints directory for available models (only *best.pt in each subdir, display folder name as display_name)."""
    models = []
    if not CHECKPOINTS_DIR.exists():
        print(f"Warning: Checkpoints directory {CHECKPOINTS_DIR} not found.")
        return models
    for model_dir in CHECKPOINTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        # Only consider *best.pt files
        best_files = list(model_dir.glob("*best.pt"))
        if not best_files:
            continue
        best_file = best_files[0]  # If multiple, just take the first
        folder = model_dir.name

        # Obtaint the model title
        readme_file = model_dir / "README.md"
        display_name = None
        if readme_file.exists():
            first_line = readme_file.read_text().splitlines()[0]
            if first_line.startswith("#"):
                display_name = first_line[1:].strip()

        if not display_name:
            display_name = folder.replace("_", " ")

        try:
            checkpoint = torch.load(best_file, map_location="cpu", weights_only=False)
            config = checkpoint.get("model_config", {})
            model_type = config.get("model_type", "")
            rank = config.get("label_column_name", "")
            # Encoding logic based on folder name
            if folder.endswith("bits0"):
                encoding = "4row"
            # Use regex to match _k{number} pattern for k-mer encoding
            elif re.search(r"_k(\d+)$", folder):
                k_value = re.search(r"_k(\d+)$", folder).group(1)
                encoding = f"kmer_{k_value}"
            # Use regex to match _bits{number} pattern for bits encoding
            elif re.search(r"_bits(\d+)$", folder):
                bits_value = re.search(r"_bits(\d+)$", folder).group(1)
                encoding = f"bits_{bits_value}"
            elif "4row" in folder:
                encoding = "4row"
            else:
                encoding = folder
            models.append(
                {
                    "name": folder,
                    "path": str(best_file),
                    "model_type": model_type,
                    "rank": rank,
                    "encoding": encoding,
                    "display_name": display_name,
                }
            )
        except Exception as e:
            print(f"Error parsing model directory {model_dir}: {e}")
    return models


# Add this function to model_utils.py
def get_model_hyperparameters(model_dir):
    """Get hyperparameters from JSON file in model directory."""
    import json

    model_dir = Path(model_dir)
    hparams = {}

    # Look for JSON files in the model directory
    json_files = list(model_dir.glob("*.json"))

    if json_files:
        try:
            # Use the first JSON file found
            with open(json_files[0], "r") as f:
                hparams = json.load(f)
            return hparams
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")

    return hparams
