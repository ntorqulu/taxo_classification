import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from models.architectures.nanni2024 import nanni_cnn1
from feature_extraction.main import SequenceCoder
from dataset.utils import info

def load_trained_model(checkpoint_path: str, device: str = 'auto'):
    """Load the trained model from checkpoint."""
    
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint.get('model_config', {})
    sequence_length = model_config.get('sequence_length', 313)
    output_size = model_config.get('output_size', 16)
    hidden_size = model_config.get('hidden_size', 8)
    
    # Initialize model
    model = nanni_cnn1(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model expects sequence length: {sequence_length}")
    print(f"Number of classes: {output_size}")
    
    return model, device, sequence_length

def encode_sequence_4row(sequence: str, target_length: int = 313):
    """Encode a single sequence using 4-row matrix encoding."""
    
    # Initialize sequence coder
    sequence_coder = SequenceCoder()
    
    # Pad or truncate sequence to target length
    if len(sequence) < target_length:
        # Pad with 'N' if too short
        sequence = sequence + 'N' * (target_length - len(sequence))
    elif len(sequence) > target_length:
        # Truncate if too long
        sequence = sequence[:target_length]
    
    print(f"Sequence length after padding/truncating: {len(sequence)}")
    
    # Encode using 4-row matrix
    encoded = sequence_coder.coding_one_hot_4rowMatrix_optimized([sequence], return_tensor=False)
    
    # Convert to tensor [1, 4, sequence_length]
    encoded_tensor = torch.tensor(encoded[0], dtype=torch.float32)  # encoded is a list, take first element
    
    print(f"Encoded tensor shape: {encoded_tensor.shape}")
    
    return encoded_tensor

def predict_sequence(sequence: str, checkpoint_path: str, class_names: list = None):
    """
    Predict the taxonomic classification of a single sequence.
    
    Args:
        sequence: DNA sequence string
        checkpoint_path: Path to trained model checkpoint
        class_names: List of class names for the 16 orders
    
    Returns:
        Dictionary with prediction results
    """
    
    # Default class names for 16 orders (update with your actual order names)
    if class_names is None:
        class_names = [
            'Blattodea', 'Coleoptera', 'Diptera', 'Ephemeroptera', 'Hemiptera', 
           'Hymenoptera', 'Lepidoptera', 'Neuroptera', 'No_insecta', 'Odonata', 
           'Orthoptera', 'Other_insecta', 'Plecoptera', 'Psocoptera',
           'Thysanoptera','Trichoptera'
        ]
        """
        {'Blattodea': 0, 'Coleoptera': 1, 'Diptera': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Hymenoptera': 5, 'Lepidoptera': 6, 'Neuroptera': 7, 'No_insecta': 8, 'Odonata': 9, 'Orthoptera': 10, 'Other_insecta': 11, 'Plecoptera': 12, 'Psocoptera': 13, 'Thysanoptera': 14, 'Trichoptera': 15}
        """
    
    try:
        # Load model
        model, device, expected_length = load_trained_model(checkpoint_path)
        
        # Encode sequence
        encoded_sequence = encode_sequence_4row(sequence, expected_length)
        encoded_sequence = encoded_sequence.to(device)
        
        # Make prediction
        with torch.no_grad():
            # Add batch dimension and channel dimension: [1, 1, 4, sequence_length]
            batch_input = encoded_sequence.unsqueeze(0).unsqueeze(0)
            
            print(f"Model input shape: {batch_input.shape}")
            
            # Forward pass
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
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def predict_from_file(file_path: str, checkpoint_path: str):
    """Predict sequences from a FASTA or text file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Simple parsing - assume it's either pure sequence or FASTA
        if content.startswith('>'):
            # FASTA format
            lines = content.split('\n')
            sequences = []
            current_seq = ""
            for line in lines:
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                sequences.append(current_seq)
        else:
            # Assume plain sequence
            sequences = [content.replace('\n', '').replace(' ', '')]
        
        results = []
        for i, seq in enumerate(sequences):
            print(f"\nPredicting sequence {i+1}/{len(sequences)}...")
            result = predict_sequence(seq, checkpoint_path)
            results.append(result)
        
        return results
        
    except Exception as e:
        return [{'error': f"Error reading file: {str(e)}"}]

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict taxonomic classification of DNA sequences')
    parser.add_argument('--sequence', type=str, help='DNA sequence to classify')
    parser.add_argument('--file', type=str, help='Path to file containing sequence(s)')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    if not args.sequence and not args.file:
        # Use example sequence if none provided
        args.sequence = "ACTATCAACCAACATTTCACATGCAGGTGCATCTGTAGATATAGCTATCTTTTCGTTACACCTAGCAGGTGTAAGATCCATCCTAGGATCAGTAAACTTTATCTCCACAATTATTAATATACGACCGGCCGGAATAAACGCCGAAAGAATCCCCCTATTTGTATGATCTGTAAGAATTACAGCACTATTACTCTTACTCTCATTACCAGTATTAGCCGGTGCTATCACTATACTCTTAACAGATCGTAACTTAAATACATCATTCTTTGACCCAGCTGGGGGAGGGGATCCGATCTTATACCAACATTTATTT"
        print("Using example sequence...")
    
    if not args.model:
        args.model = "checkpoints/nanni_cnn1_20250618-233940_order_name_bits0/nanni_cnn1_20250618-233940_epoch10.pt"  # Default model path, change as needed
    
    # Path to your trained model
    checkpoint_path = args.model
    
    if args.file:
        # Predict from file
        results = predict_from_file(args.file, checkpoint_path)
        
        for i, result in enumerate(results):
            print(f"\n{'='*50}")
            print(f"SEQUENCE {i+1} PREDICTION RESULTS")
            print(f"{'='*50}")
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                if 'traceback' in result:
                    print(f"Traceback:\n{result['traceback']}")
            else:
                print(f"Sequence length: {result['sequence_length']}")
                print(f"Predicted taxonomic order: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
                
                print(f"\n📊 Top 3 Predictions:")
                for j, pred in enumerate(result['top_predictions'], 1):
                    print(f"{j}. {pred['class']}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
    
    elif args.sequence:
        # Predict single sequence
        result = predict_sequence(args.sequence, checkpoint_path)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            if 'traceback' in result:
                print(f"Traceback:\n{result['traceback']}")
        else:
            print(f"\n🧬 Sequence Prediction Results:")
            print(f"Sequence length: {result['sequence_length']}")
            print(f"Predicted taxonomic order: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            
            print(f"\n📊 Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")