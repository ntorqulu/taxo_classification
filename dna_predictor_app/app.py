from flask import Flask, render_template, request, jsonify
from model_utils import list_models, get_encodings, get_ranks, get_model_by_name
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path - try multiple possible locations
possible_src_paths = [
    Path(__file__).parent.parent / "src",  # If running from dna_predictor_app
    Path(__file__).parent / "src",         # If src is copied to app directory
    Path("src"),                           # If running from parent directory
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
    from models.architectures.nanni2024 import nanni_cnn1, nanni_cnn2, nanni_att, nanni_att_kmer
    from models.architectures.enhanced_mlp import EnhancedMLP
    from models.architectures.cnn_model import CNNModel
    from models.architectures.bert_model import BERTTaxoModel
    from models.architectures.basic_model import BasicTaxoModel
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
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', '')
    
    # If model_type not in config, try to infer from the checkpoint path
    if not model_type:
        path = Path(checkpoint_path)
        for part in path.parts:
            if any(model_name in part.lower() for model_name in 
                  ['cnn', 'mlp', 'bert', 'nanni', 'enhanced']):
                model_type = part.lower()
                break
    
    # Initialize the appropriate model based on type
    if 'nanni_cnn2' in model_type:
        model = nanni_cnn2(
            sequence_length=model_config.get('sequence_length', 313),
            hidden_size=model_config.get('hidden_size', 1024),
            output_size=model_config.get('output_size', 16)
        )
    elif 'nanni_cnn1' in model_type:
        model = nanni_cnn1(
            sequence_length=model_config.get('sequence_length', 313),
            hidden_size=model_config.get('hidden_size', 8),
            output_size=model_config.get('output_size', 16)
        )
    elif 'nanni_att' in model_type and 'kmer' in model_type:
        model = nanni_att_kmer(
            input_size=model_config.get('input_size', 1024),
            output_size=model_config.get('output_size', 16),
            num_heads=model_config.get('num_heads', 8),
            embed_dim=model_config.get('embed_dim', 64),
            hidden_size=model_config.get('hidden_size', 100)
        )
    elif 'nanni_att' in model_type:
        model = nanni_att(
            sequence_length=model_config.get('sequence_length', 313),
            output_size=model_config.get('output_size', 16),
            num_heads=model_config.get('num_heads', 8),
            embed_dim=model_config.get('embed_dim', 64),
            hidden_size=model_config.get('hidden_size', 100)
        )
    elif 'enhanced_mlp' in model_type or 'enhancedmlp' in model_type:
        model = EnhancedMLP(
            input_size=model_config.get('input_size', 1024),
            hidden_sizes=model_config.get('hidden_sizes', [256, 128]),
            output_size=model_config.get('output_size', 16),
            dropout=model_config.get('dropout', 0.2),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
    elif 'cnn' in model_type:
        model = CNNModel(
            input_size=model_config.get('input_size', 1024),
            output_size=model_config.get('output_size', 16),
            kernel_sizes=model_config.get('kernel_sizes', [3, 5, 7]),
            num_filters=model_config.get('num_filters', [64, 128, 256]),
            fc_sizes=model_config.get('fc_sizes', [512, 256]),
            dropout=model_config.get('dropout', 0.3)
        )
    elif 'bert' in model_type:
        model = BERTTaxoModel(
            vocab_size=model_config.get('vocab_size', 4),
            max_length=model_config.get('max_length', 512),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 3),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.2),
            output_size=model_config.get('output_size', 16)
        )
    elif 'basic' in model_type:
        model = BasicTaxoModel(
            input_size=model_config.get('input_size', 1024),
            hidden_size=model_config.get('hidden_size', 256),
            output_size=model_config.get('output_size', 16)
        )
    else:
        # Default to using the model.load class method if available
        try:
            # Look for load method in the checkpoint
            model_classname = model_config.get('name', '').split('.')[-1]
            model_class = globals().get(model_classname)
            if model_class and hasattr(model_class, 'load'):
                model = model_class.load(checkpoint_path, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Return class_names from top-level checkpoint if present
    class_names = checkpoint.get('class_names', None)
    return model, device, model_config, class_names

def encode_sequence(sequence: str, encoding_type: str = '4row', target_length: int = 313, k: int = 4):
    """Encode a DNA sequence based on the specified encoding type."""
    sequence_coder = SequenceCoder()
    
    # Pad or truncate sequence to target length
    if len(sequence) < target_length:
        sequence = sequence + 'N' * (target_length - len(sequence))
    elif len(sequence) > target_length:
        sequence = sequence[:target_length]
    
    print(f"Sequence length after padding/truncation: {len(sequence)}")
    
    # Encode based on encoding type
    if encoding_type == '4row' or encoding_type.startswith('4rowmatrix'):
        encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
    elif encoding_type.startswith('kmer'):
        # Extract k from encoding_type if specified (e.g., 'kmer_3')
        if '_' in encoding_type:
            try:
                k = int(encoding_type.split('_')[1])
            except:
                pass
        encoded = sequence_coder.coding_kmer_optimized([sequence], k=k)
        encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
    elif encoding_type.startswith('bits'):
        # Extract bits from encoding_type if specified (e.g., 'bits_2')
        bits = 4
        if '_' in encoding_type:
            try:
                bits = int(encoding_type.split('_')[1])
            except:
                pass
        
        # Use coding_one_hot_bit_optimized with correct parameters
        print(f"Using bits={bits} for encoding, with sequence length {len(sequence)}")
        
        # The max_seq_length parameter is what determines the output size
        encoded = sequence_coder.coding_one_hot_bit_optimized(
            [sequence], 
            bits=bits,
            max_seq_length=target_length  # This ensures the correct output size
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
        model_metadata = get_model_by_name(Path(model_path).parent.name)
        if model_metadata:
            encoding_type = model_metadata.get('encoding', '4row')
            rank = model_metadata.get('rank', 'order')
            model_display_name = model_metadata.get('display_name', 'Unknown Model')
        else:
            encoding_type = '4row'  # Default
            rank = config.get('label_column_name', 'order')
            model_display_name = config.get('name', 'Unknown Model')
        
        # --- Use class_names from checkpoint as primary source ---
        class_names = checkpoint_class_names
        if not class_names:
            if 'class_names' in config:
                class_names = config['class_names']
            elif hasattr(model, 'class_names'):
                class_names = getattr(model, 'class_names', None)
            elif model_metadata and 'class_names' in model_metadata:
                class_names = model_metadata['class_names']
        
        target_length = config.get('sequence_length', 313)  # Default value
        
        # In predict_sequence function, after loading the model but before encoding
        if encoding_type.startswith('bits'):
            # Extract bits value
            bits = 4
            if '_' in encoding_type:
                try:
                    bits = int(encoding_type.split('_')[1])
                except:
                    pass
                    
            # For models that have input_size attribute
            if hasattr(model, 'input_size'):
                expected_input_size = model.input_size
            elif 'input_size' in config:
                expected_input_size = config['input_size']
            elif isinstance(model, CNNModel) and hasattr(model, 'fc_layers') and model.fc_layers:
                # For CNNModel, try to get from first layer
                expected_input_size = model.fc_layers[0].in_features
            
            if expected_input_size:
                print(f"Model expects input_size={expected_input_size}")
                
                # Calculate required sequence length correctly
                required_seq_length = expected_input_size // bits
                print(f"For bits={bits}, required sequence length={required_seq_length}")
                
                # Override target_length with the required length
                target_length = required_seq_length
            else:
                print("Warning: Couldn't determine model's expected input size")

        # Encode sequence based on encoding type
        encoded_sequence = encode_sequence(
            sequence, 
            encoding_type=encoding_type,
            target_length=target_length,
            k=config.get('k', 4)
        )
        encoded_sequence = encoded_sequence.to(device)     
        # Prepare input based on model type
        if encoding_type == '4row' or encoding_type.startswith('4rowmatrix'):
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
            
            results = {
                'sequence': sequence,
                'sequence_length': len(sequence),
                'model_name': model_display_name,
                'target_level': rank,
                'predicted_class': class_names[predicted_class] if class_names and predicted_class < len(class_names) else f"Class_{predicted_class}",
                'confidence': confidence,
                'top_predictions': [
                    {
                        'class': class_names[idx.item()] if class_names and idx.item() < len(class_names) else f"Class_{idx.item()}",
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            }
            
            return results
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return {'error': str(e), 'traceback': traceback_str}
        

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
            valid_bases = set('ACGTN')
            if not all(base in valid_bases for base in sequence):
                error = "Invalid DNA sequence. Only A, C, G, T, N are allowed."
            else:
                prediction = predict_sequence(sequence, model_path)
                if 'error' in prediction:
                    error = prediction['error']
                    prediction = None

    return render_template(
        "index.html",
        models=models,
        encodings=encodings,
        ranks=ranks,
        prediction=prediction,
        error=error
    )

@app.route("/api/models")
def get_models_api():
    """API endpoint to get available models."""
    models = list_models()
    return jsonify(models)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
