import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import time
from torch.utils.tensorboard import SummaryWriter
from models.architectures.base_model import BaseModel
from dataset.utils import info
from models.results import Results, compute_accuracy

class Trainer:
    """Class for training and evaluating models."""
    
    def __init__(self, 
                model: nn.Module,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                log_dir: Optional[str] = None,
                checkpoint_dir: Optional[str] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            class_names: List of class names (e.g., phylum names)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.class_names = class_names if class_names is not None else []
        
        # Create mapping from index to class name for easier lookup
        self.class_idx_to_name = {idx: name for idx, name in enumerate(self.class_names)} if self.class_names else {}
        
        # Set up logging
        self.log_dir = log_dir or os.path.join('runs', f"{model.name}_{time.strftime('%Y%m%d-%H%M%S')}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir or 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, average accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
                
        # Compute averages
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct / total
        
        # After each epoch, log the loss and accuracy
        info(f'Train Epoch: {epoch} Loss: {avg_loss:.6f}, Acc: {100. * avg_accuracy:.2f}%')
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int, prefix: str = 'val') -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            val_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
            
        Returns:
            Tuple of (average loss, average accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target.view(-1))
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        # Compute averages
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'Accuracy/{prefix}', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    @torch.no_grad()
    def quick_evaluate(self, data_loader: DataLoader, epoch: int, prefix: str = 'val') -> Tuple[float, float]:
        """
        Quick evaluation without per-class metrics or visualizations.
        
        Args:
            data_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs
            
        Returns:
            Tuple of (average loss, average accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            if data.dim() == 3:
                data = data.unsqueeze(1)
                
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target.view(-1))
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        # Compute averages
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'Accuracy/{prefix}', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    @torch.no_grad()
    def compute_per_class_metrics(self, data_loader: DataLoader, epoch: int, prefix: str = 'val') -> Dict[str, float]:
        """
        Compute and log metrics for each class individually.
        
        Args:
            data_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
            
        Returns:
            Dictionary with per-class metrics
        """
        self.model.eval()
        
        # Get number of classes from first batch
        batch = next(iter(data_loader))
        output = self.model(batch[0][:1].to(self.device))
        num_classes = output.size(1)
        
        if epoch == 1 and prefix == 'val':
            info(f"Class name mappings for {num_classes} classes:")
            for class_idx in range(num_classes):
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                info(f"  Class index {class_idx} -> '{class_name}'")
        
        # Initialize counters for each class
        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)
        loss_per_class = torch.zeros(num_classes)
        
        # Process all batches
        for data, target in data_loader:
            data, target = data.to(self.device), target.view(-1).to(self.device)
            
            # Forward pass
            output = self.model(data)
            predictions = output.argmax(dim=1)
            
            # Calculate metrics for each class
            for class_idx in range(num_classes):
                # Find samples with this class as ground truth
                class_mask = (target == class_idx)
                class_count = class_mask.sum().item()
                
                if class_count > 0:
                    # Accuracy: correctly predicted / total for this class
                    # Count correct predictions for this class
                    correct = (predictions[class_mask] == class_idx).sum().item()
                    correct_per_class[class_idx] += correct
                    total_per_class[class_idx] += class_count
                    
                    # Loss: calculate separately using one-hot targets
                    # Select only the relevant outputs
                    relevant_outputs = output[class_mask]
                    
                    # Create targets for this class (all same class)
                    class_targets = torch.full((class_count,), class_idx, 
                                            device=self.device, dtype=torch.long)
                    
                    # Compute loss
                    class_loss = self.criterion(relevant_outputs, class_targets)
                    loss_per_class[class_idx] += class_loss.item() * class_count
        
        # Calculate final metrics and log to TensorBoard
        metrics = {}
        
        for class_idx in range(num_classes):
            if total_per_class[class_idx] > 0:
                # Calculate metrics
                accuracy = correct_per_class[class_idx] / total_per_class[class_idx]
                avg_loss = loss_per_class[class_idx] / total_per_class[class_idx]
                
                # Store in return dict
                metrics[f'class_{class_idx}_acc'] = accuracy.item()
                metrics[f'class_{class_idx}_loss'] = avg_loss.item()
                
                # Get class name if available
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
                
                # Log to TensorBoard
                self.writer.add_scalar(f'Class_{class_name}/Accuracy_{prefix}', accuracy, epoch)
                self.writer.add_scalar(f'Class_{class_name}/Loss_{prefix}', avg_loss, epoch)
                
                # Also log class sample count to track class imbalance
                self.writer.add_scalar(f'Class_{class_name}/Samples_{prefix}', total_per_class[class_idx], epoch)
                
        return metrics

    def log_confusion_matrix(self, data_loader: DataLoader, epoch: int, prefix: str = 'val'):
        """
        Generate and log confusion matrix to TensorBoard.
        
        Args:
            data_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
        except ImportError:
            info("Skipping confusion matrix visualization: required libraries not available")
            return
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # Collect predictions and targets
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                preds = output.argmax(dim=1)
                
                # Ensure both are flattened to 1D arrays
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Ensure they're both 1D
        if all_targets.ndim > 1:
            all_targets = all_targets.flatten()
        
        if all_preds.ndim > 1:
            all_preds = all_preds.flatten()
        
        # Get unique classes from both arrays
        unique_classes = np.unique(np.concatenate([all_targets, all_preds]))
        num_classes = len(unique_classes)
        
        # Use class names if available
        if self.class_names and len(self.class_names) >= num_classes:
            labels = self.class_names[:num_classes]
        else:
            labels = [f"Class {i}" for i in range(num_classes)]
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
        
        # Create two versions: raw counts and normalized
        
        # 1. Raw counts
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
        plt.title(f'{prefix} Confusion Matrix (Raw Counts) - Epoch {epoch}')
        plt.ylabel('True Phylum')
        plt.xlabel('Predicted Phylum')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # Add to TensorBoard (convert to CxHxW format)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        self.writer.add_image(f'Confusion_Matrix_Raw/{prefix}', image_tensor, epoch)
        plt.close()
        
        # 2. Normalize by row (true labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
        plt.title(f'{prefix} Confusion Matrix (Normalized) - Epoch {epoch}')
        plt.ylabel('True Phylum')
        plt.xlabel('Predicted Phylum')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # Add to TensorBoard (convert to CxHxW format)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        self.writer.add_image(f'Confusion_Matrix_Norm/{prefix}', image_tensor, epoch)
        plt.close()
    
    def log_class_performance_chart(self, data_loader: DataLoader, epoch: int, prefix: str = 'val'):
        """
        Generate and log per-class performance chart to TensorBoard.
        
        Args:
            data_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image
        except ImportError:
            info("Skipping class performance chart: required libraries not available")
            return
            
        # Get per-class metrics
        metrics = self.compute_per_class_metrics(data_loader, epoch, prefix)
        
        # Extract class indices and accuracies
        class_indices = sorted([int(k.split('_')[1]) for k in metrics.keys() if k.endswith('_acc')])
        accuracies = [metrics[f'class_{idx}_acc'] for idx in class_indices]
        
        # Use class names if available
        if self.class_names and len(self.class_names) >= len(class_indices):
            labels = [self.class_names[idx] for idx in class_indices]
        else:
            labels = [f"Class {idx}" for idx in class_indices]
        
        # Create figure - horizontal bar chart sorted by accuracy
        plt.figure(figsize=(10, max(6, len(class_indices) * 0.4)))
        
        # Sort by accuracy
        sorted_data = sorted(zip(labels, accuracies), key=lambda x: x[1])
        sorted_labels, sorted_accuracies = zip(*sorted_data)
        
        y_pos = np.arange(len(sorted_labels))
        bars = plt.barh(y_pos, sorted_accuracies, color='skyblue')
        
        # Add accuracy values on bars
        for i, acc in enumerate(sorted_accuracies):
            plt.text(acc + 0.01, i, f'{acc:.2f}', va='center')
        
        plt.yticks(y_pos, sorted_labels)
        plt.title(f'Phylum Classification Accuracy ({prefix}) - Epoch {epoch}')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # Add to TensorBoard (convert to CxHxW format)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        self.writer.add_image(f'Class_Performance/{prefix}', image_tensor, epoch)
        plt.close()
        
        # Also create a radar chart for multidimensional view
        if len(class_indices) > 2:  # Only create radar chart if we have at least 3 classes
            plt.figure(figsize=(8, 8))
            
            # Prepare data for radar chart
            angles = np.linspace(0, 2*np.pi, len(class_indices), endpoint=False).tolist()
            
            # Close the polygon by appending the first point
            accuracies_radar = accuracies.copy()
            accuracies_radar.append(accuracies[0])
            angles.append(angles[0])
            
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, accuracies_radar, 'o-', linewidth=2, color='skyblue')
            ax.fill(angles, accuracies_radar, alpha=0.25, color='skyblue')
            
            # Set category labels
            plt.xticks(angles[:-1], labels, size=10)
            
            # Set y ticks
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray')
            plt.ylim(0, 1)
            
            plt.title(f'Phylum Classification Accuracy - Radar View - Epoch {epoch}')
            
            # Convert to image and log to TensorBoard
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
            self.writer.add_image(f'Phylum_Radar/{prefix}', image_tensor, epoch)
            plt.close()
    
    def log_phylum_evolution_chart(self, history: Dict[str, List], epoch: int):
        """
        Generate a chart showing the evolution of classification performance for each phylum.
        
        Args:
            history: Training history dictionary
            epoch: Current epoch 
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image
        except ImportError:
            info("Skipping phylum evolution chart: required libraries not available")
            return
        
        # Extract per-class metrics history
        per_class_history = history.get('per_class_metrics', [])
        if not per_class_history or epoch < 2:
            return  # Not enough data points yet
            
        # Find all unique class indices across epochs
        class_indices = set()
        for metrics in per_class_history:
            for key in metrics.keys():
                if key.endswith('_acc'):
                    class_idx = int(key.split('_')[1])
                    class_indices.add(class_idx)
                    
        class_indices = sorted(list(class_indices))
        
        # Create accuracy evolution chart
        plt.figure(figsize=(12, 8))
        
        epochs_range = list(range(1, len(per_class_history)+1))
        
        # Plot each class evolution
        for idx in class_indices:
            # Extract accuracy values for this class across epochs
            values = []
            for metrics in per_class_history:
                key = f'class_{idx}_acc'
                if key in metrics:
                    values.append(metrics[key])
                else:
                    values.append(None)  # Missing data point
            
            # Get class name if available
            if idx < len(self.class_names):
                class_name = self.class_names[idx]
            else:
                class_name = f"Class {idx}"
                
            # Plot line
            plt.plot(epochs_range, values, 'o-', label=class_name)
            
        plt.title(f'Phylum Classification Accuracy Evolution - Epoch {epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Convert to image and log to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        self.writer.add_image(f'Phylum_Evolution', image_tensor, epoch)
        plt.close()
    
    def log_learning_progress_table(self, metrics: Dict[str, float], epoch: int, prefix: str = 'val'):
        """
        Log a table with learning progress for each phylum.
        
        Args:
            metrics: Dictionary with per-class metrics
            epoch: Current epoch number  
            prefix: Prefix for TensorBoard logs
        """
        # Extract class metrics
        accuracies = []
        losses = []
        class_indices = []
        
        for key, value in metrics.items():
            if key.endswith('_acc'):
                class_idx = int(key.split('_')[1])
                class_indices.append(class_idx)
                accuracies.append(value)
                
                # Get corresponding loss if available
                loss_key = f"class_{class_idx}_loss"
                loss = metrics.get(loss_key, float('nan'))
                losses.append(loss)
                
        # Get class names
        class_names = []
        for idx in class_indices:
            if idx < len(self.class_names):
                class_names.append(self.class_names[idx])
            else:
                class_names.append(f"Class {idx}")
        
        # Create markdown table
        table = "| Phylum | Accuracy | Loss |\n"
        table += "| --- | --- | --- |\n"
        
        # Sort by accuracy (descending)
        sorted_data = sorted(zip(class_names, accuracies, losses), key=lambda x: x[1], reverse=True)
        
        for name, acc, loss in sorted_data:
            table += f"| {name} | {acc:.4f} | {loss:.4f} |\n"
        
        # Log to TensorBoard
        self.writer.add_text(f"{prefix}_phylum_metrics", table, epoch)
        
    def train(self, 
         train_loader: DataLoader,
         val_loader: DataLoader,
         test_loader: Optional[DataLoader] = None,
         epochs: int = 10,
         patience: int = 5,
         save_best: bool = True,
         fast_mode: bool = False,
         eval_frequency: int = 1) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            epochs: Number of epochs to train
            patience: Early stopping patience
            save_best: Whether to save best model
            fast_mode: If True, use faster evaluation with minimal metrics
            eval_frequency: How often to run detailed evaluation (epochs)
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'per_class_metrics': []  # Add this to store per-class metrics
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        t0 = time.time()
        info(f"Starting training for {epochs} epochs in {'fast' if fast_mode else 'detailed'} mode")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Decide if we should run detailed evaluation in this epoch
            run_detailed = (epoch % eval_frequency == 0 or epoch == 1 or epoch == epochs)
            
            # Compute train metrics (more detailed if not in fast mode)
            if not fast_mode and run_detailed:
                train_class_metrics = self.compute_per_class_metrics(train_loader, epoch, prefix='train')
                history.setdefault('train_per_class_metrics', []).append(train_class_metrics)
                self.log_learning_progress_table(train_class_metrics, epoch, prefix='train')
                
                # Create visualizations less frequently
                if epoch % (eval_frequency * 2) == 0 or epoch == 1 or epoch == epochs:
                    self.log_class_performance_chart(train_loader, epoch, prefix='train')
                    self.log_confusion_matrix(train_loader, epoch, prefix='train')
            
            # Validation - always needed for early stopping
            if fast_mode and not run_detailed:
                # Fast evaluation
                val_loss, val_acc = self.quick_evaluate(val_loader, epoch, prefix='val')
            else:
                # Standard evaluation
                val_loss, val_acc = self.evaluate(val_loader, epoch, prefix='val')
                
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Detailed validation metrics only if not in fast mode and on scheduled epochs
            if not fast_mode and run_detailed:
                # Calculate per-class metrics
                val_class_metrics = self.compute_per_class_metrics(val_loader, epoch, prefix='val')
                history['per_class_metrics'].append(val_class_metrics)
                
                # Log learning progress table
                self.log_learning_progress_table(val_class_metrics, epoch, prefix='val')
                
                # Create visualizations
                self.log_class_performance_chart(val_loader, epoch, prefix='val')
                self.log_confusion_matrix(val_loader, epoch, prefix='val')
                    
                # Plot evolution of phylum performance
                if epoch >= eval_frequency*2:
                    self.log_phylum_evolution_chart(history, epoch)
            
            # Test evaluation
            if test_loader is not None:
                if fast_mode:
                    # In fast mode, evaluate test set only on final epoch
                    if epoch == epochs or (patience_counter >= patience):
                        test_loss, test_acc = self.quick_evaluate(test_loader, epoch, prefix='test')
                        history['test_loss'].append(test_loss)
                        history['test_acc'].append(test_acc)
                elif run_detailed:
                    # In detailed mode, evaluate test set on scheduled epochs
                    test_loss, test_acc = self.evaluate(test_loader, epoch, prefix='test')
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)
                    
                    # Calculate per-class metrics on test set
                    test_class_metrics = self.compute_per_class_metrics(test_loader, epoch, prefix='test')
                    
                    # Log test metrics table
                    self.log_learning_progress_table(test_class_metrics, epoch, prefix='test')
                    
                    # Create test visualizations
                    if epoch % (eval_frequency * 2) == 0 or epoch == epochs:
                        self.log_class_performance_chart(test_loader, epoch, prefix='test')
                        self.log_confusion_matrix(test_loader, epoch, prefix='test')
            
            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                
            # Save regular checkpoint (less frequently in fast mode)
            checkpoint_freq = 10 if fast_mode else 5
            if epoch % checkpoint_freq == 0:
                self._save_checkpoint(epoch)
                
            # Early stopping
            if patience_counter >= patience:
                info(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break
                    
        # Training completed
        seconds = time.time() - t0
        minutes = int(seconds / 60)
        seconds = int(seconds - minutes * 60)
        info(f"Training completed in {minutes}m {seconds}s")
        
        # Final evaluation on test set if provided
        if test_loader is not None:
            info("Evaluating best model on test set...")
            best_model_path = os.path.join(self.checkpoint_dir, f"{self.model.name}_best.pt")
            if os.path.exists(best_model_path):
                # Load best model
                checkpoint = torch.load(best_model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            test_loss, test_acc = self.evaluate(test_loader, epochs, prefix='test')
            
            # Always do detailed metrics for final evaluation unless in fast mode
            if not fast_mode:
                test_class_metrics = self.compute_per_class_metrics(test_loader, epochs, prefix='final_test')
                self.log_class_performance_chart(test_loader, epochs, prefix='final_test')
                self.log_confusion_matrix(test_loader, epochs, prefix='final_test')
                self.log_learning_progress_table(test_class_metrics, epochs, prefix='final_test')
                
            info(f"Final test accuracy: {100. * test_acc:.2f}%")
                
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
            'model_name': self.model.name,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()
            
        # Save regular checkpoint
        filename = f"{self.model.name}_epoch{epoch}.pt"
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        
        # Save best model checkpoint
        if is_best:
            best_filename = f"{self.model.name}_best.pt"
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, best_filename))