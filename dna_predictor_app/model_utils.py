import os
from pathlib import Path
import torch
import re

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
        best_files = list(model_dir.glob('*best.pt'))
        if not best_files:
            continue
        best_file = best_files[0]  # If multiple, just take the first
        folder = model_dir.name
        display_name = folder.replace('_', ' ')
        try:
            checkpoint = torch.load(best_file, map_location='cpu')
            config = checkpoint.get('model_config', {})
            model_type = config.get('model_type', '')
            rank = config.get('label_column_name', '')
            # Encoding logic based on folder name
            if folder.endswith('bits0'):
                encoding = '4row'
            # Use regex to match _k{number} pattern for k-mer encoding
            elif re.search(r'_k(\d+)$', folder):
                k_value = re.search(r'_k(\d+)$', folder).group(1)
                encoding = f'kmer_{k_value}'
            # Use regex to match _bits{number} pattern for bits encoding
            elif re.search(r'_bits(\d+)$', folder):
                bits_value = re.search(r'_bits(\d+)$', folder).group(1)
                encoding = f'bits_{bits_value}'
            elif '4row' in folder:
                encoding = '4row'
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
        except Exception as e:
            print(f"Error parsing model directory {model_dir}: {e}")
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