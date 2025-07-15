import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from model_utils import get_encodings, get_model_by_name, get_model_hyperparameters, get_ranks, list_models

# Add src to path - try multiple possible locations
possible_src_paths = [
    Path(__file__).parent.parent / "src",  # If running from dna_predictor_app
    Path(__file__).parent / "src",  # If src is copied to app directory
    Path("src"),  # If running from parent directory
]

for src_path in possible_src_paths:
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        break
else:
    print("Warning: Could not find src directory. Please ensure it's accessible.")

# Import modules that don't depend on data directory
try:
    from feature_extraction.main import SequenceCoder
    from models.architectures.basic_model import BasicTaxoModel
    from models.architectures.bert_model import BERTTaxoModel
    from models.architectures.cnn_model import CNNModel
    from models.architectures.enhanced_mlp import EnhancedMLP
    from models.architectures.nanni2024 import nanni_att, nanni_att_kmer, nanni_cnn1, nanni_cnn2
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the src directory is accessible and contains the required modules.")
    sys.exit(1)


# Import dataset.utils only when needed (not during startup)
def get_dataset_info():
    try:
        from dataset.utils import info

        return info
    except Exception as e:
        print(f"Warning: Could not import dataset.utils: {e}")
        return lambda x: print(f"INFO: {x}")


app = Flask(__name__)


def load_model(checkpoint_path):
    """Load the trained model from checkpoint, supporting multiple model types."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load checkpoint with comprehensive error handling
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid load key" in error_msg or "unsupported operand" in error_msg:
            print(f"Error: Checkpoint file appears to be corrupted: {checkpoint_path}")
            print(f"Error details: {e}")
            
            # Try to check if file exists and has reasonable size
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                file_size = checkpoint_file.stat().st_size
                print(f"File size: {file_size} bytes")
                if file_size < 1000:  # Very small file, likely corrupted
                    raise RuntimeError(f"Checkpoint file appears to be corrupted or empty: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
                
            # Try alternative loading methods
            try:
                print("Attempting to load with weights_only=True...")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                print("Successfully loaded with weights_only=True")
            except Exception as e2:
                print(f"Alternative loading failed: {e2}")
                raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        else:
            raise e

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("model_type", "")

    # If model_type not in config, try to infer from the checkpoint path
    if not model_type:
        path = Path(checkpoint_path)
        for part in path.parts:
            if any(model_name in part.lower() for model_name in ["cnn", "mlp", "bert", "nanni", "enhanced"]):
                model_type = part.lower()
                break
    
    if not model_type:
        raise ValueError(f"Could not determine model type from checkpoint or path: {checkpoint_path}")

    # Initialize the appropriate model based on type
    if "nanni_cnn2" in model_type:
        model = nanni_cnn2(
            sequence_length=model_config.get("sequence_length", 313),
            hidden_size=model_config.get("hidden_size", 1024),
            output_size=model_config.get("output_size", 16),
        )
    elif "nanni_cnn1" in model_type:
        model = nanni_cnn1(
            sequence_length=model_config.get("sequence_length", 313),
            hidden_size=model_config.get("hidden_size", 8),
            output_size=model_config.get("output_size", 16),
        )
    elif "nanni_att" in model_type and "kmer" in model_type:
        model = nanni_att_kmer(
            input_size=model_config.get("input_size", 1024),
            output_size=model_config.get("output_size", 16),
            num_heads=model_config.get("num_heads", 8),
            embed_dim=model_config.get("embed_dim", 64),
            hidden_size=model_config.get("hidden_size", 100),
        )
    elif "nanni_att" in model_type and "bits" in model_type:
        model = nanni_att(
            sequence_length=model_config.get("sequence_length", 313),
            output_size=model_config.get("output_size", 16),
            num_heads=model_config.get("num_heads", 8),
            embed_dim=model_config.get("embed_dim", 64),
            hidden_size=model_config.get("hidden_size", 100),
        )

        # Extract encoding details
        model_metadata = get_model_by_name(Path(checkpoint_path).parent.name)
        if model_metadata and "encoding" in model_metadata:
            encoding_type = model_metadata.get("encoding", "4row")
            if encoding_type.startswith("bits"):
                bits = 4
                if "_" in encoding_type:
                    try:
                        bits = int(encoding_type.split("_")[1])
                    except:
                        pass

                # Extract the exact input size from the checkpoint
                if "model_state_dict" in checkpoint and "vector_projection.weight" in checkpoint["model_state_dict"]:
                    vector_proj_weight = checkpoint["model_state_dict"]["vector_projection.weight"]
                    actual_input_size = vector_proj_weight.shape[1]
                    embed_dim = vector_proj_weight.shape[0]
                    
                    print(f"Using exact dimensions from checkpoint: input_size={actual_input_size}, embed_dim={embed_dim}")
                    model.vector_projection = nn.Linear(actual_input_size, embed_dim).to(device)
                else:
                    seq_length = model_config.get("sequence_length", 313)
                    input_size = seq_length * bits
                    print(f"Estimating dimensions: sequence_length={seq_length}, input_size={input_size}")
                    model.vector_projection = nn.Linear(input_size, model.embed_dim).to(device)
    
    elif "nanni_att" in model_type:
        model = nanni_att(
            sequence_length=model_config.get("sequence_length", 313),
            output_size=model_config.get("output_size", 16),
            num_heads=model_config.get("num_heads", 8),
            embed_dim=model_config.get("embed_dim", 64),
            hidden_size=model_config.get("hidden_size", 100),
        )
    elif "enhanced_mlp" in model_type or "enhancedmlp" in model_type:
        model = EnhancedMLP(
            input_size=model_config.get("input_size", 1024),
            hidden_sizes=model_config.get("hidden_sizes", [256, 128]),
            output_size=model_config.get("output_size", 16),
            dropout=model_config.get("dropout", 0.2),
            use_batch_norm=model_config.get("use_batch_norm", True),
        )
    elif "cnn" in model_type:
        model = CNNModel(
            input_size=model_config.get("input_size", 1024),
            output_size=model_config.get("output_size", 16),
            kernel_sizes=model_config.get("kernel_sizes", [3, 5, 7]),
            num_filters=model_config.get("num_filters", [64, 128, 256]),
            fc_sizes=model_config.get("fc_sizes", [512, 256]),
            dropout=model_config.get("dropout", 0.3),
        )
    elif "bert" in model_type:
        model = BERTTaxoModel(
            vocab_size=model_config.get("vocab_size", 4),
            max_length=model_config.get("max_length", 512),
            hidden_size=model_config.get("hidden_size", 128),
            num_layers=model_config.get("num_layers", 3),
            num_heads=model_config.get("num_heads", 4),
            dropout=model_config.get("dropout", 0.2),
            output_size=model_config.get("output_size", 16),
        )
    elif "basic" in model_type:
        model = BasicTaxoModel(
            input_size=model_config.get("input_size", 1024),
            hidden_size=model_config.get("hidden_size", 256),
            output_size=model_config.get("output_size", 16),
        )
    else:
        try:
            model_classname = model_config.get("name", "").split(".")[-1]
            model_class = globals().get(model_classname)
            if model_class and hasattr(model_class, "load"):
                model = model_class.load(checkpoint_path, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    # Load trained weights with comprehensive error handling
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Handle case where checkpoint is directly a state dict
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded checkpoint as direct state dict")
    except RuntimeError as e:
        if "Unexpected key(s) in state_dict" in str(e):
            print(f"Warning: State dict mismatch for {model_type}. Attempting to load with filtered state dict...")
            
            # Get the state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Define keys to filter out based on model type
            if 'nanni_att' in model_type:
                keys_to_filter = ['attn_pool.weight', 'attn_pool.bias']
            elif 'bert' in model_type:
                keys_to_filter = ['kmer_classifier']
            else:
                keys_to_filter = []
            
            # Filter out the problematic keys
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if not any(filter_key in k for filter_key in keys_to_filter)}
            
            # Print what we're filtering out
            filtered_keys = [k for k in state_dict.keys() 
                           if any(filter_key in k for filter_key in keys_to_filter)]
            if filtered_keys:
                print(f"Filtered out keys: {filtered_keys}")
            
            # Load with strict=False to ignore missing keys
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys (will use default initialization): {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys (ignored): {unexpected_keys}")
        else:
            raise e
    except Exception as e:
        raise RuntimeError(f"Error loading model state from {checkpoint_path}: {str(e)}")
    
    model.to(device)
    model.eval()

    # Return class_names from top-level checkpoint if present
    class_names = checkpoint.get("class_names", None)
    return model, device, model_config, class_names


def encode_sequence(sequence: str, encoding_type: str = "4row", target_length: int = 313, k: int = 4):
    """Encode a DNA sequence based on the specified encoding type."""
    sequence_coder = SequenceCoder()

    # Pad or truncate sequence to target length
    if len(sequence) < target_length:
        sequence = sequence + "N" * (target_length - len(sequence))
    elif len(sequence) > target_length:
        sequence = sequence[:target_length]

    print(f"Sequence length after padding/truncation: {len(sequence)}")

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

        # Use coding_one_hot_bit_optimized with correct parameters
        print(f"Using bits={bits} for encoding, with sequence length {len(sequence)}")

        # The max_seq_length parameter is what determines the output size
        encoded = sequence_coder.coding_one_hot_bit_optimized(
            [sequence],
            bits=bits,
            max_seq_length=target_length,  # This ensures the correct output size
        )

        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
        print(f"Encoded tensor shape: {encoded_tensor.shape}")

        # Check if the size matches what's expected
        expected_size = target_length * bits
        if encoded_tensor.numel() != expected_size:
            print(f"WARNING: Encoded tensor size {encoded_tensor.numel()} doesn't match expected size {expected_size}")
    else:
        # Default to 4-row matrix
        encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)

    return encoded_tensor


def predict_sequence(sequence: str, model_path: str):
    """Predict the taxonomic classification of a single sequence."""
    try:
        # Load model
        model, device, config, checkpoint_class_names = load_model(model_path)

        # Get model metadata from path or config
        model_dir = Path(model_path).parent
        model_metadata = get_model_by_name(Path(model_path).parent.name)
        hyperparameters = get_model_hyperparameters(model_dir)
        if model_metadata:
            encoding_type = model_metadata.get("encoding", "4row")
            rank = model_metadata.get("rank", "order")
            model_display_name = model_metadata.get("display_name", "Unknown Model")
        else:
            encoding_type = "4row"  # Default
            rank = config.get("label_column_name", "order")
            model_display_name = config.get("name", "Unknown Model")

        # --- Use class_names from checkpoint as primary source ---
        class_names = checkpoint_class_names
        if not class_names:
            if "class_names" in config:
                class_names = config["class_names"]
            elif hasattr(model, "class_names"):
                class_names = getattr(model, "class_names", None)
            elif model_metadata and "class_names" in model_metadata:
                class_names = model_metadata["class_names"]

        target_length = config.get("sequence_length", 313)  # Default value

        # Enhanced logic for determining correct sequence length for bits encoding
        if encoding_type.startswith("bits") or encoding_type == "4row":
            # Extract bits value - for "4row" or "bits0", use 4 bits
            bits = 4
            if encoding_type.startswith("bits") and "_" in encoding_type:
                try:
                    bits = int(encoding_type.split("_")[1])
                except:
                    pass

            # For nanni_att models, check vector_projection layer
            model_type = config.get("model_type", "").lower()
            if "nanni_att" in model_type:
                # Check if model has vector_projection layer with specific input size
                if hasattr(model, "vector_projection") and model.vector_projection is not None:
                    expected_input_size = model.vector_projection.in_features
                    print(f"Model vector_projection expects input_size={expected_input_size}")
                    
                    # Calculate required sequence length
                    required_seq_length = expected_input_size // bits
                    print(f"For bits={bits}, required sequence length={required_seq_length}")
                    target_length = required_seq_length
                else:
                    print("Warning: nanni_att model doesn't have vector_projection layer initialized")
                    
                    # For nanni_att models with bits encoding, we need to determine the correct input size
                    # Get the actual expected input size from the model path/name
                    if "bits4" in model_path.lower():
                        # For bits4 models, calculate from the expected 1280 input size
                        expected_input_size = 1280  # This is what the model expects
                        required_seq_length = expected_input_size // bits
                        print(f"Using fallback for bits4: expected_input_size={expected_input_size}, required sequence length={required_seq_length}")
                        target_length = required_seq_length
                    elif "bits3" in model_path.lower():
                        # For bits3 models, calculate from the expected 960 input size
                        expected_input_size = 960
                        required_seq_length = expected_input_size // bits
                        print(f"Using fallback for bits3: expected_input_size={expected_input_size}, required sequence length={required_seq_length}")
                        target_length = required_seq_length
                    elif "bits2" in model_path.lower():
                        # For bits2 models, calculate from the expected 640 input size
                        expected_input_size = 640
                        required_seq_length = expected_input_size // bits
                        print(f"Using fallback for bits2: expected_input_size={expected_input_size}, required sequence length={required_seq_length}")
                        target_length = required_seq_length
                    elif encoding_type == "4row" or "bits0" in model_path.lower():
                        # For 4row/bits0 models, use standard 4-bit encoding
                        expected_input_size = 1252  # 313 * 4
                        required_seq_length = expected_input_size // 4
                        print(f"Using fallback for 4row/bits0: expected_input_size={expected_input_size}, required sequence length={required_seq_length}")
                        target_length = required_seq_length
                    else:
                        print("Warning: Couldn't determine model's expected input size from path")
            else:
                # For other model types, try to determine expected input size
                expected_input_size = None
                if hasattr(model, "input_size"):
                    expected_input_size = model.input_size
                elif "input_size" in config:
                    expected_input_size = config["input_size"]
                elif isinstance(model, CNNModel) and hasattr(model, "fc_layers") and model.fc_layers:
                    expected_input_size = model.fc_layers[0].in_features

                if expected_input_size:
                    print(f"Model expects input_size={expected_input_size}")
                    required_seq_length = expected_input_size // bits
                    print(f"For bits={bits}, required sequence length={required_seq_length}")
                    target_length = required_seq_length
                else:
                    print("Warning: Couldn't determine model's expected input size")

        print(f"Final target_length: {target_length}")

        # Encode sequence based on encoding type
        encoded_sequence = encode_sequence(
            sequence, encoding_type=encoding_type, target_length=target_length, k=config.get("k", 4)
        )
        encoded_sequence = encoded_sequence.to(device)
        
        print(f"Final encoded tensor shape: {encoded_sequence.shape}")
        
        # Prepare input based on model type
        if encoding_type == "4row" or encoding_type.startswith("4rowmatrix"):
            if encoded_sequence.dim() == 2:
                # Ensure format is [batch, channels, height, width] for CNN models
                batch_input = encoded_sequence.unsqueeze(0).unsqueeze(0)
            else:
                batch_input = encoded_sequence.unsqueeze(0)
        else:
            # For vector-based encodings (k-mer, bits)
            batch_input = encoded_sequence.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)

            # Get prediction
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            # Get top 3 predictions
            if class_names:
                top_k = min(3, len(class_names))
            else:
                top_k = 3
            top_probs, top_indices = torch.topk(probabilities[0], k=top_k)

            # Calculate expected input size based on encoding type
            if encoding_type == "4row":
                expected_input_size = target_length * 4  # 4 bits per position
            elif encoding_type.startswith("bits"):
                bits = 4
                if "_" in encoding_type:
                    try:
                        bits = int(encoding_type.split("_")[1])
                    except:
                        pass
                expected_input_size = target_length * bits
            else:
                expected_input_size = target_length

            results = {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "encoded_length": target_length,
                "expected_input_size": expected_input_size,
                "actual_encoded_size": encoded_sequence.numel(),
                "model_name": model_display_name,
                "target_level": rank,
                "predicted_class": class_names[predicted_class]
                if class_names and predicted_class < len(class_names)
                else f"Class_{predicted_class}",
                "confidence": confidence,
                "top_predictions": [
                    {
                        "class": class_names[idx.item()]
                        if class_names and idx.item() < len(class_names)
                        else f"Class_{idx.item()}",
                        "probability": prob.item(),
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ],
                "hyperparameters": hyperparameters,
            }

            return results
    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return {"error": str(e), "traceback": traceback_str}


@app.route("/", methods=["GET", "POST"])
def index():
    models = list_models()
    encodings = get_encodings()
    ranks = get_ranks()
    prediction = None
    error = None

    if request.method == "POST":
        sequence = request.form.get("sequence", "").strip()
        model_path = request.form.get("model")

        if not sequence or not model_path:
            error = "Please provide both a DNA sequence and select a model."
        else:
            # Validate sequence
            sequence = sequence.upper()
            valid_bases = set("ACGTN")
            if not all(base in valid_bases for base in sequence):
                error = "Invalid DNA sequence. Only A, C, G, T, N are allowed."
            else:
                prediction = predict_sequence(sequence, model_path)
                if "error" in prediction:
                    error = prediction["error"]
                    prediction = None

    return render_template(
        "index.html",
        models=sorted(models, key=lambda m: m["display_name"].lower()),
        encodings=encodings,
        ranks=ranks,
        prediction=prediction,
        error=error,
    )


@app.route("/api/models")
def get_models_api():
    """API endpoint to get available models."""
    models = list_models()
    return jsonify(models)


@app.route("/api/model_hparams/<model_name>")
def get_model_hparams_api(model_name):
    """API endpoint to get model hyperparameters."""
    models = list_models()
    for model in models:
        if model["name"] == model_name:
            model_dir = Path(model["path"]).parent
            hparams = get_model_hyperparameters(model_dir)
            return jsonify(hparams)
    return jsonify({"error": "Model not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
