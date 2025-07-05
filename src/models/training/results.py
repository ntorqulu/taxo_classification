import numpy as np
from typing import List, Dict, Union, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path


class TrainingResults:
    """Class to track and store training results."""
    
    def __init__(self):
        self.epochs = []
    
    def add_epoch(self, epoch_data: Dict[str, Any]):
        """Add epoch results."""
        self.epochs.append(epoch_data)
    
    def get_best_epoch(self, metric: str = 'val_accuracy') -> Dict[str, Any]:
        """Get the best epoch based on a metric."""
        if not self.epochs:
            return {}
        
        best_epoch = max(self.epochs, key=lambda x: x.get(metric, 0))
        return best_epoch
    
    def save(self, path: Union[str, Path]):
        """Save results to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump({
                'epochs': self.epochs,
                'best_epoch': self.get_best_epoch()
            }, f, indent=2)
    
    def load(self, path: Union[str, Path]):
        """Load results from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
            self.epochs = data.get('epochs', [])


def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy from true and predicted labels.
    
    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels or logits
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    # If y_pred contains logits (has argmax method)
    if hasattr(y_pred, 'argmax'):
        preds = y_pred.argmax(-1)
    else:
        # y_pred already contains class predictions
        preds = y_pred
    
    # Count correct predictions
    correct = sum(1 for t, p in zip(y_true, preds) if t == p)
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0

def compute_precision_recall_f1(y_true, y_pred, average='macro'):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    return precision, recall, f1

def compute_metrics(y_true, y_pred):
    """
    Compute all metrics: accuracy, precision, recall, and F1.
    
    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels
        
    Returns:
        Dictionary with metrics
    """
    accuracy = compute_accuracy(y_true, y_pred)
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
