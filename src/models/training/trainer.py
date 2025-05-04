import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from tqdm import tqdm
import os
import time
from ..architectures.base_model import BaseModel

class Trainer:
    """Generic trainer for taxonomy classification models."""
    
    def __init__(self,
                 model: BaseModel,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 metric_functions: Optional[Dict[str, Callable]] = None):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model : BaseModel
            Model to train
        criterion : nn.Module
            Loss function
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. If None, uses Adam with lr=0.001
        lr_scheduler : torch.optim.lr_scheduler, optional
            Learning rate scheduler
        device : str
            Device to use for training ('cuda' or 'cpu')
        metric_functions : Dict[str, Callable], optional
            Dictionary of metric functions to evaluate during training
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model.to(device)
        
        # Default metrics
        self.metric_functions = metric_functions or {
            'accuracy': lambda y_true, y_pred: (y_pred == y_true).float().mean().item()
        }
        
        self.logger = logging.getLogger(__name__)
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Add metrics to history
        for metric_name in self.metric_functions.keys():
            self.history[f'train_{metric_name}'] = []
            self.history[f'val_{metric_name}'] = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader for training data
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of metrics
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for inputs, targets in tqdm(train_loader, desc="Training", unit="batch"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs['logits'], targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs['logits'], dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = {'loss': epoch_loss}
        
        # Calculate additional metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        for name, metric_fn in self.metric_functions.items():
            metrics[name] = metric_fn(y_true, y_pred)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Parameters:
        -----------
        val_loader : DataLoader
            DataLoader for validation data
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of metrics
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", unit="batch"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs['logits'], targets)
                
                # Track metrics
                running_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs['logits'], dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader.dataset)
        metrics = {'loss': epoch_loss}
        
        # Calculate additional metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        for name, metric_fn in self.metric_functions.items():
            metrics[name] = metric_fn(y_true, y_pred)
        
        return metrics
    
    def fit(self, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            epochs: int = 10,
            patience: Optional[int] = None,
            model_dir: Optional[str] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader for training data
        val_loader : DataLoader, optional
            DataLoader for validation data
        epochs : int
            Number of epochs to train
        patience : int, optional
            Early stopping patience. If None, no early stopping.
        model_dir : str, optional
            Directory to save model checkpoints. If None, no checkpoints are saved.
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create model directory if specified
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            for name in self.metric_functions.keys():
                self.history[f'train_{name}'].append(train_metrics[name])
            
            # Validate if validation data is provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                for name in self.metric_functions.keys():
                    self.history[f'val_{name}'].append(val_metrics[name])
                
                # Check for improvement
                if val_metrics['loss'] < best_val_loss:
                    improvement = best_val_loss - val_metrics['loss']
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save best model if directory is specified
                    if model_dir:
                        self.model.save(os.path.join(model_dir, 'best_model.pt'))
                        
                    if verbose:
                        self.logger.info(f"Validation loss improved by {improvement:.6f}")
                else:
                    patience_counter += 1
            
            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            if verbose:
                summary = f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                summary += f"loss: {train_metrics['loss']:.4f}"
                for name, value in train_metrics.items():
                    if name != 'loss':
                        summary += f" - {name}: {value:.4f}"
                        
                if val_loader is not None:
                    summary += f" - val_loss: {val_metrics['loss']:.4f}"
                    for name, value in val_metrics.items():
                        if name != 'loss':
                            summary += f" - val_{name}: {value:.4f}"
                
                self.logger.info(summary)
            
            # Early stopping
            if patience is not None and patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model if directory is specified
        if model_dir:
            self.model.save(os.path.join(model_dir, 'final_model.pt'))
        
        return self.history