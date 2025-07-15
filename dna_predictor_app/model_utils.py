import os
import re
from pathlib import Path

import torch

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

        # Get the model title from README
        readme_file = model_dir / "README.md"
        display_name = None
        if readme_file.exists():
            try:
                first_line = readme_file.read_text().splitlines()[0]
                if first_line.startswith("#"):
                    display_name = first_line[1:].strip()
            except Exception:
                pass  # Silently skip README reading errors

        if not display_name:
            display_name = folder.replace("_", " ")

        try:
            # Load checkpoint with silent error handling
            checkpoint = None
            try:
                checkpoint = torch.load(best_file, map_location="cpu", weights_only=False)
            except Exception as e:
                error_msg = str(e).lower()
                if "invalid load key" in error_msg or "unsupported operand" in error_msg:
                    # Check file size - if very small, silently skip
                    file_size = best_file.stat().st_size if best_file.exists() else 0
                    if file_size < 1000:  # Very small file, likely corrupted
                        continue  # Silently skip corrupted files
                    
                    # Try alternative loading methods silently
                    try:
                        checkpoint = torch.load(best_file, map_location="cpu", weights_only=True)
                    except Exception:
                        continue  # Silently skip if alternative loading fails
                else:
                    continue  # Silently skip other loading errors

            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                continue  # Silently skip invalid checkpoint formats
            
            # Extract model configuration
            config = checkpoint.get("model_config", {})
            model_type = config.get("model_type", "")
            rank = config.get("label_column_name", "")
            
            # Handle case where checkpoint is directly a state dict
            if not config and any('weight' in k or 'bias' in k for k in checkpoint.keys()):
                model_type = "unknown"
                rank = "unknown"
            
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

            models.append({
                "name": folder,
                "path": str(best_file),
                "model_type": model_type,
                "rank": rank,
                "encoding": encoding,
                "display_name": display_name,
            })
            
        except Exception:
            # Silently skip any parsing errors
            continue
    
    return models


def get_encodings():
    """Return available encodings based on available models."""
    models = list_models()
    encodings = list(set([m["encoding"] for m in models]))
    return encodings


def get_ranks():
    """Return available taxonomic ranks based on available models."""
    models = list_models()
    ranks = list(set([m["rank"] for m in models]))
    return ranks


def get_model_by_name(model_name):
    """Get model configuration by name."""
    models = list_models()
    for model in models:
        if model["name"] == model_name:
            return model
    return None


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
