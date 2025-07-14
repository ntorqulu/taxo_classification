from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from taxo_classification.attention.utils_attention import load_data
from utils_attention import encode_sequence, get_model_by_name, get_model_hyperparameters, load_model


def plot_attention_heatmap(attention, class_name):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()

    attention_2d = np.expand_dims(attention, axis=0)  # shape (1, L)

    plt.figure(figsize=(12, 2))
    plt.imshow(attention_2d, aspect="auto", cmap="YlOrRd")  # amarillo a rojo
    plt.colorbar(label="Attention Weight")
    plt.title(f"Attention Heatmap (Vector) - Clase: {class_name}")
    plt.xlabel("Token Index")
    plt.yticks([])  # Oculta eje Y porque es solo 1 fila
    plt.savefig(f"{class_name}.png")
    plt.close()


def compute_attention_of_a_sequence(sequence: str, model_path: str):
    """Predict the taxonomic classification of a single sequence."""
    model, device, config, checkpoint_class_names = load_model(model_path)

    model_dir = Path(model_path).parent
    model_metadata = get_model_by_name(Path(model_path).parent.name)
    hyperparameters = get_model_hyperparameters(model_dir)
    if model_metadata:
        encoding_type = model_metadata.get("encoding", "4row")
    else:
        encoding_type = "4row"

    class_names = checkpoint_class_names
    if not class_names:
        if "class_names" in config:
            class_names = config["class_names"]
        elif hasattr(model, "class_names"):
            class_names = getattr(model, "class_names", None)
        elif model_metadata and "class_names" in model_metadata:
            class_names = model_metadata["class_names"]

    encoded_sequence = encode_sequence(sequence, encoding_type=encoding_type, k=config.get("k", 4))
    encoded_sequence = encoded_sequence.to(device)
    batch_input = encoded_sequence.unsqueeze(0)
    if encoding_type == "4row" or encoding_type.startswith("4rowmatrix") and encoded_sequence.dim() == 2:
        batch_input = batch_input.unsqueeze(0)

    with torch.no_grad():
        batch_input, attn_weights = model(batch_input, return_attention=True)
        attn_weights = attn_weights.squeeze(0).cpu().numpy()
        class_pred = batch_input.argmax(dim=1)
        class_name = class_names[int(class_pred.item())]
        plot_attention_heatmap(attn_weights, class_name)


def compute_attention_by_predicted_class(model_path: str) -> dict:
    model, device, config, checkpoint_class_names = load_model(model_path)

    model_dir = Path(model_path).parent
    model_metadata = get_model_by_name(Path(model_path).parent.name)
    hyperparameters = get_model_hyperparameters(model_dir)

    class_names = checkpoint_class_names
    if not class_names:
        if "class_names" in config:
            class_names = config["class_names"]
        elif hasattr(model, "class_names"):
            class_names = getattr(model, "class_names", None)
        elif model_metadata and "class_names" in model_metadata:
            class_names = model_metadata["class_names"]
    model.eval()
    attention_by_class = {class_name: [] for class_name in class_names}

    taxo_data_loaders = load_data(hyperparameters)

    with torch.no_grad():
        for x, _ in taxo_data_loaders:
            preds, attn_weights = model.forward(x, return_attention=True)
            class_preds = preds.argmax(dim=1)

            for i, cls in enumerate(class_preds):
                attention_by_class[class_names[int(cls.item())]].append(attn_weights[i].cpu())

    for cls in attention_by_class:
        attention_by_class[cls] = torch.stack(attention_by_class[cls], dim=0)  # [N, num_heads, seq_len, seq_len]
        attention_by_class[cls] = attention_by_class[cls].mean(dim=0)
        plot_attention_heatmap(attention_by_class[cls], cls)

    return attention_by_class
