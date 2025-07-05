import os
from pathlib import Path

CHECKPOINTS_DIR = Path("../checkpoints")  # Adjust path as needed

def list_models():
    """Scan checkpoints directory for available models starting with 'best_'."""
    models = []
    
    for model_dir in CHECKPOINTS_DIR.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("best_"):
            # Look for .pt files in the model directory
            for file in model_dir.glob("*.pt"):
                # Parse model name: best_nanni_cnn2_20250705-193402_order_name_bits0
                parts = model_dir.name.split("_")
                
                # Extract components
                model_type = parts[1] + "_" + parts[2]  # nanni_cnn2
                date = parts[3]  # 20250705-193402
                rank = parts[4]  # order
                encoding = parts[5]  # bits0
                
                models.append({
                    "name": model_dir.name,
                    "path": str(file),
                    "model_type": model_type,
                    "rank": rank,
                    "encoding": encoding,
                    "date": date,
                    "display_name": f"{model_type.replace('_', ' ').title()} - {rank.title()}"
                })
    
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