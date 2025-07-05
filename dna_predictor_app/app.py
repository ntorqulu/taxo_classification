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
    from models.architectures.nanni2024 import nanni_cnn2
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
    """Load the trained model from checkpoint."""
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})
    sequence_length = model_config.get('sequence_length', 313)
    output_size = model_config.get('output_size', 16)
    hidden_size = model_config.get('hidden_size', 8)
    
    # Initialize model (using nanni_cnn2 for your case)
    model = nanni_cnn2(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device, model_config

def encode_sequence_4row(sequence: str, target_length: int = 313):
    """Encode a single sequence using 4-row matrix encoding."""
    sequence_coder = SequenceCoder()
    
    # Pad or truncate sequence to target length
    if len(sequence) < target_length:
        sequence = sequence + 'N' * (target_length - len(sequence))
    elif len(sequence) > target_length:
        sequence = sequence[:target_length]
    
    # Encode using 4-row matrix
    encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
    encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)
    
    return encoded_tensor

def predict_sequence(sequence: str, model_path: str):
    """Predict the taxonomic classification of a single sequence."""
    
    # Default class names for 16 orders (update with your actual order names)
    class_names = [
        'Blattodea', 'Coleoptera', 'Diptera', 'Ephemeroptera', 'Hemiptera', 
        'Hymenoptera', 'Lepidoptera', 'Neuroptera', 'No_insecta', 'Odonata', 
        'Orthoptera', 'Other_insecta', 'Plecoptera', 'Psocoptera',
        'Thysanoptera', 'Trichoptera'
    ]
    
    try:
        # Load model
        model, device, config = load_model(model_path)
        
        # Encode sequence
        encoded_sequence = encode_sequence_4row(sequence, config.get('sequence_length', 313))
        encoded_sequence = encoded_sequence.to(device)
        
        # Add batch dimension and channel dimension: [1, 1, 4, sequence_length]
        batch_input = encoded_sequence.unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(class_names)))
            
            results = {
                'sequence': sequence,
                'sequence_length': len(sequence),
                'model_name': 'Nanni CNN2 - Order Classification',
                'target_level': 'order_name',
                'predicted_class': class_names[predicted_class] if predicted_class < len(class_names) else f"Class_{predicted_class}",
                'confidence': confidence,
                'top_predictions': [
                    {
                        'class': class_names[idx.item()] if idx.item() < len(class_names) else f"Class_{idx.item()}",
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            }
            
            return results
            
    except Exception as e:
        return {'error': str(e)}

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
