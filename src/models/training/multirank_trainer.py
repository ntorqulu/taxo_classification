import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import time
from torch.utils.tensorboard import SummaryWriter
from src.dataset.utils import info

from src.models.architectures.hierarchical_model import HierarchicalLoss, HierarchicalAccuracy
from src.constants.taxonomy_labels import TAXONOMY_LEVELS


class HierarchicalTrainer:
    """
    Trainer for hierarchical taxonomy classification models.
    
    This trainer handles models that predict multiple taxonomic levels simultaneously.
    """
    
    def __init__(self, 
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                log_dir: Optional[str] = None,
                checkpoint_dir: Optional[str] = None,
                class_names_per_level: Optional[Dict[str, List[str]]] = None,
                from_checkpoint: bool = False,
                from_checkpoint_path: Optional[str] = None):
        """
        Initialize hierarchical trainer.
        
        Args:
            model: Hierarchical model to train
            criterion: Hierarchical loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            class_names_per_level: Dictionary mapping taxonomic levels to their class names
            from_checkpoint: Whether to load from checkpoint
            from_checkpoint_path: Path to checkpoint file
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.class_names_per_level = class_names_per_level or {}
        self.from_checkpoint = from_checkpoint
        self.from_checkpoint_path = from_checkpoint_path
        
        # Create mapping from index to class name for each level
        self.class_idx_to_name_per_level = {}
        for level, class_names in self.class_names_per_level.items():
            self.class_idx_to_name_per_level[level] = {
                idx: name for idx, name in enumerate(class_names)
            }
        
        # Set up logging
        self.log_dir = log_dir or os.path.join('runs', f"{model.name}_{time.strftime('%Y%m%d-%H%M%S')}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir or 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, dictionary of accuracies per level)
        """
        self.model.train()
        total_loss = 0.0
        level_correct = {level: 0 for level in TAXONOMY_LEVELS}
        level_total = {level: 0 for level in TAXONOMY_LEVELS}
        
        # Get the total dataset size
        dataset_size = len(train_loader.dataset)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            features = batch['features'].to(self.device)
            targets = {level: target.to(self.device) for level, target in batch['targets'].items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features)
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute accuracy for each level
            for level in TAXONOMY_LEVELS:
                if level in predictions and level in targets:
                    pred = predictions[level].argmax(dim=1)
                    correct = pred.eq(targets[level]).sum().item()
                    level_correct[level] += correct
                    level_total[level] += targets[level].size(0)
            
            # Log batch results
            if batch_idx % 10 == 0:
                batch_accuracies = {}
                for level in TAXONOMY_LEVELS:
                    if level_total[level] > 0:
                        batch_accuracies[level] = 100. * level_correct[level] / level_total[level]
                
                acc_str = ", ".join([f"{level}: {acc:.2f}%" for level, acc in batch_accuracies.items()])
                info(f'Train Epoch: {epoch} [{batch_idx * len(features)}/{dataset_size}] '
                     f'Loss: {loss.item():.6f}, Acc: {acc_str}')
                
        # Compute averages
        avg_loss = total_loss / len(train_loader)
        avg_accuracies = {}
        for level in TAXONOMY_LEVELS:
            if level_total[level] > 0:
                avg_accuracies[level] = level_correct[level] / level_total[level]
            else:
                avg_accuracies[level] = 0.0
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        for level, acc in avg_accuracies.items():
            self.writer.add_scalar(f'Accuracy/{level}/train', acc, epoch)
        
        return avg_loss, avg_accuracies

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int, prefix: str = 'val') -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]], float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            val_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
            
        Returns:
            Tuple of (average loss, accuracies per level, detailed metrics per level, hierarchical accuracy)
        """
        from models.training.results import compute_metrics
        
        self.model.eval()
        total_loss = 0.0
        level_predictions = {level: [] for level in TAXONOMY_LEVELS}
        level_targets = {level: [] for level in TAXONOMY_LEVELS}
        level_logits = {level: [] for level in TAXONOMY_LEVELS}  # Store logits for hierarchical accuracy
        
        for batch in val_loader:
            # Move data to device
            features = batch['features'].to(self.device)
            targets = {level: target.to(self.device) for level, target in batch['targets'].items()}
            
            # Forward pass
            predictions = self.model(features)
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Track metrics
            total_loss += loss.item()
            
            # Collect predictions and targets for each level
            for level in TAXONOMY_LEVELS:
                if level in predictions and level in targets:
                    pred = predictions[level].argmax(dim=1)
                    level_predictions[level].extend(pred.cpu().numpy())
                    level_targets[level].extend(targets[level].cpu().numpy())
                    # Store logits for hierarchical accuracy computation
                    level_logits[level].append(predictions[level].cpu())
        
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics for each level
        level_accuracies = {}
        level_metrics = {}
        
        for level in TAXONOMY_LEVELS:
            if level_predictions[level] and level_targets[level]:
                metrics = compute_metrics(level_targets[level], level_predictions[level])
                level_accuracies[level] = metrics['accuracy']
                level_metrics[level] = metrics
            else:
                level_accuracies[level] = 0.0
                level_metrics[level] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Compute hierarchical accuracy (all levels correct)
        # Concatenate logits from all batches
        hierarchical_logits = {}
        hierarchical_targets = {}
        for level in TAXONOMY_LEVELS:
            if level_logits[level]:
                hierarchical_logits[level] = torch.cat(level_logits[level], dim=0)
                hierarchical_targets[level] = torch.tensor(level_targets[level])
        
        hierarchical_acc = HierarchicalAccuracy.compute_hierarchical_accuracy(
            hierarchical_logits, hierarchical_targets
        )
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'Hierarchical_Accuracy/{prefix}', hierarchical_acc, epoch)
        
        for level in TAXONOMY_LEVELS:
            if level in level_accuracies:
                self.writer.add_scalar(f'Accuracy/{level}/{prefix}', level_accuracies[level], epoch)
                if level in level_metrics:
                    metrics = level_metrics[level]
                    self.writer.add_scalar(f'Precision/{level}/{prefix}', metrics['precision'], epoch)
                    self.writer.add_scalar(f'Recall/{level}/{prefix}', metrics['recall'], epoch)
                    self.writer.add_scalar(f'F1/{level}/{prefix}', metrics['f1'], epoch)
        
        # Log summary
        acc_str = ", ".join([f"{level}: {acc:.4f}" for level, acc in level_accuracies.items()])
        info(f'{prefix.capitalize()} metrics - Loss: {avg_loss:.4f}, Hierarchical Acc: {hierarchical_acc:.4f}, '
             f'Level Accuracies: {acc_str}')
        
        return avg_loss, level_accuracies, level_metrics, hierarchical_acc

    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader] = None,
             epochs: int = 10,
             patience: int = 5,
             save_best: bool = True,
             fast_mode: bool = False,
             eval_frequency: int = 1) -> Dict[str, List]:
        """
        Train the hierarchical model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            epochs: Number of training epochs
            patience: Early stopping patience
            save_best: Whether to save the best model
            fast_mode: Whether to use fast evaluation mode
            eval_frequency: How often to evaluate (every N epochs)
            
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        best_hierarchical_acc = 0.0
        patience_counter = 0
        best_epoch = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'hierarchical_acc': []
        }
        
        # Start training timer
        t0 = time.time()
        info(f"Starting training for {epochs} epochs in {'fast' if fast_mode else 'standard'} mode")
        
        # Handle checkpoint loading
        if self.from_checkpoint:
            if self.from_checkpoint_path is None:
                raise ValueError("from_checkpoint_path must be provided when from_checkpoint is True")
            if not os.path.exists(self.from_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file '{self.from_checkpoint_path}' does not exist")
            start_epoch = self.load_checkpoint(self.from_checkpoint_path) + 1
            info(f"Loaded model from checkpoint: {self.from_checkpoint_path}")
        else:
            start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            # Training
            train_loss, train_accuracies = self.train_epoch(train_loader, epoch)
            
            # Validation
            if epoch % eval_frequency == 0:
                val_loss, val_accuracies, val_metrics, hierarchical_acc = self.evaluate(val_loader, epoch, 'val')
                
                # Update history
                history['train_losses'].append(train_loss)
                history['val_losses'].append(val_loss)
                history['train_accuracies'].append(train_accuracies)
                history['val_accuracies'].append(val_accuracies)
                history['hierarchical_acc'].append(hierarchical_acc)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Track best model
                improved = False
                if hierarchical_acc > best_hierarchical_acc:
                    best_hierarchical_acc = hierarchical_acc
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    improved = True
                elif hierarchical_acc == best_hierarchical_acc and val_loss < best_val_loss:
                    # If hierarchical accuracy is the same, use loss as tie-breaker
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    improved = True
                else:
                    patience_counter += 1
                
                # Save checkpoint if improved
                if improved and save_best:
                    self._save_checkpoint(epoch, is_best=True)
                
                # Save regular checkpoint every 5 epochs (similar to single-rank)
                if epoch % 5 == 0 and save_best:
                    self._save_checkpoint(epoch)
                
                # Early stopping
                if patience_counter >= patience:
                    info(f'Early stopping triggered after {epoch} epochs. Best epoch was {best_epoch} with hierarchical accuracy {best_hierarchical_acc:.4f}.')
                    break
                
                # Test evaluation (if provided)
                if test_loader is not None and epoch == epochs:
                    test_loss, test_accuracies, test_metrics, test_hierarchical_acc = self.evaluate(test_loader, epoch, 'test')
                    info(f'Final test results - Loss: {test_loss:.4f}, Hierarchical Acc: {test_hierarchical_acc:.4f}')
        
        # Training completed
        total_time = time.time() - t0
        mins, secs = divmod(total_time, 60)
        hours, mins = divmod(mins, 60)
        
        info(f"Training completed in {int(hours)}h {int(mins)}m {int(secs)}s")
        info(f"Best epoch: {best_epoch} with hierarchical accuracy: {best_hierarchical_acc:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            info(f"Restored best model from epoch {best_epoch}")
        
        return history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'model_name': self.model.name,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()
            
        # Save regular checkpoint (similar to single-rank)
        filename = f"{self.model.name}_epoch{epoch}.pt"
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        
        # Save best model checkpoint (similar to single-rank)
        if is_best:
            best_filename = f"{self.model.name}_best.pt"
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, best_filename))
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        info(f'Loaded checkpoint from epoch {epoch}')
        
        return epoch
    
    def predict(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader for prediction data
            
        Returns:
            Dictionary mapping taxonomic levels to their predictions
        """
        self.model.eval()
        predictions = {level: [] for level in TAXONOMY_LEVELS}
        probabilities = {level: [] for level in TAXONOMY_LEVELS}
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                batch_predictions = self.model(features)
                
                for level in TAXONOMY_LEVELS:
                    if level in batch_predictions:
                        # Get class predictions
                        pred = batch_predictions[level].argmax(dim=1)
                        predictions[level].extend(pred.cpu().numpy())
                        
                        # Get probabilities
                        probs = torch.softmax(batch_predictions[level], dim=1)
                        probabilities[level].extend(probs.cpu().numpy())
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        } 

    def get_label_stats(self) -> dict[str, dict[str, Tuple[int, float]]]:
        # Implementation of get_label_stats method
        pass 