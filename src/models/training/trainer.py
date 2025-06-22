import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import time
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import info

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
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int, prefix: str = 'val') -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate model on validation/test data.
        
        Args:
            val_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs ('val' or 'test')
            
        Returns:
            Tuple of (average loss, average accuracy, metrics dict)
        """
        from models.training.results import compute_metrics
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for data, target in val_loader:
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
            
            # Collect predictions and targets for overall metrics
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())
        
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        # Compute all metrics
        metrics = compute_metrics(all_targets, all_predictions)
        avg_accuracy = metrics['accuracy']
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'Accuracy/{prefix}', avg_accuracy, epoch)
        self.writer.add_scalar(f'Precision/{prefix}', metrics['precision'], epoch)
        self.writer.add_scalar(f'Recall/{prefix}', metrics['recall'], epoch)
        self.writer.add_scalar(f'F1/{prefix}', metrics['f1'], epoch)
        
        info(f'{prefix.capitalize()} metrics - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f}, ' +
            f'F1: {metrics["f1"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}')
        
        return avg_loss, avg_accuracy, metrics
    
    @torch.no_grad()
    def quick_evaluate(self, data_loader: DataLoader, epoch: int, prefix: str = 'val') -> Tuple[float, float, Dict[str, float]]:
        """
        Quick evaluation without per-class metrics or visualizations.
        
        Args:
            data_loader: DataLoader for validation/test data
            epoch: Current epoch number
            prefix: Prefix for TensorBoard logs
            
        Returns:
            Tuple of (average loss, average accuracy, metrics dict)
        """
        from models.training.results import compute_metrics
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
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
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())
        
        # Compute average loss
        avg_loss = total_loss / len(data_loader)
        
        # Compute all metrics
        metrics = compute_metrics(all_targets, all_predictions)
        avg_accuracy = metrics['accuracy']
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Loss/{prefix}', avg_loss, epoch)
        self.writer.add_scalar(f'Accuracy/{prefix}', avg_accuracy, epoch)
        self.writer.add_scalar(f'Precision/{prefix}', metrics['precision'], epoch)
        self.writer.add_scalar(f'Recall/{prefix}', metrics['recall'], epoch)
        self.writer.add_scalar(f'F1/{prefix}', metrics['f1'], epoch)
        
        return avg_loss, avg_accuracy, metrics
    
    @torch.no_grad()
    def compute_per_class_metrics(self, data_loader: DataLoader, epoch: int, prefix: str = 'val') -> Dict[str, float]:
        """Compute metrics for each class."""
        from sklearn.metrics import precision_recall_fscore_support
        
        self.model.eval()
        
        # Get all predictions and targets
        all_predictions = []
        all_targets = []
        
        # Process all batches
        for data, target in data_loader:
            data, target = data.to(self.device), target.view(-1).to(self.device)
            if data.dim() == 3:
                data = data.unsqueeze(1)
            
            # Forward pass
            output = self.model(data)
            predictions = output.argmax(dim=1)
            
            # Store for overall metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        # Get number of classes
        batch = next(iter(data_loader))
        output = self.model(batch[0][:1].to(self.device))
        num_classes = output.size(1)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate per-class precision, recall, and F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average=None, labels=range(num_classes), zero_division=0
        )
        
        # Calculate overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='macro', zero_division=0
        )
        
        # Store overall metrics
        metrics['overall_precision'] = overall_precision
        metrics['overall_recall'] = overall_recall
        metrics['overall_f1'] = overall_f1
        
        # Log to TensorBoard
        self.writer.add_scalar(f'Overall/Precision_{prefix}', overall_precision, epoch)
        self.writer.add_scalar(f'Overall/Recall_{prefix}', overall_recall, epoch)
        self.writer.add_scalar(f'Overall/F1_{prefix}', overall_f1, epoch)
        
        # Store and log per-class metrics
        for class_idx in range(num_classes):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
            
            # Store per-class metrics
            metrics[f'class_{class_idx}_precision'] = precision[class_idx]
            metrics[f'class_{class_idx}_recall'] = recall[class_idx]
            metrics[f'class_{class_idx}_f1'] = f1[class_idx]
            
            # Log to TensorBoard
            self.writer.add_scalar(f'Class_{class_name}/Precision_{prefix}', precision[class_idx], epoch)
            self.writer.add_scalar(f'Class_{class_name}/Recall_{prefix}', recall[class_idx], epoch)
            self.writer.add_scalar(f'Class_{class_name}/F1_{prefix}', f1[class_idx], epoch)
            self.writer.add_scalar(f'Class_{class_name}/Support_{prefix}', support[class_idx], epoch)
        
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
            test_loader: DataLoader for test data (optional)
            epochs: Maximum number of epochs to train
            patience: Early stopping patience (epochs with no improvement)
            save_best: Whether to save the best model checkpoint
            fast_mode: Use faster evaluation with minimal metrics
            eval_frequency: How often to run full evaluation (epochs)
            
        Returns:
            Dictionary with training history
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'train_precision': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
        }
        
        # Add test metrics if test_loader is provided
        if test_loader is not None:
            history.update({
                'test_loss': [],
                'test_acc': [],
                'test_f1': [],
                'test_precision': [],
                'test_recall': [],
            })
        
        # Initialize tracking variables
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_model_state = None
        
        # Start training timer
        t0 = time.time()
        info(f"Starting training for {epochs} epochs in {'fast' if fast_mode else 'standard'} mode")
        
        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Calculate additional training metrics (F1, precision, recall)
            self.model.eval()
            with torch.no_grad():
                all_preds = []
                all_targets = []
                
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    if data.dim() == 3:
                        data = data.unsqueeze(1)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.view(-1).cpu().numpy())
                
                # Calculate training metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_preds, average='macro', zero_division=0
                )
            
            # Store and log training metrics
            history['train_f1'].append(f1)
            history['train_precision'].append(precision)
            history['train_recall'].append(recall)
            
            # Log to TensorBoard
            self.writer.add_scalar('F1/train', f1, epoch)
            self.writer.add_scalar('Precision/train', precision, epoch)
            self.writer.add_scalar('Recall/train', recall, epoch)
            
            # Log to console
            info(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ' +
                f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
            
            # Decide if we should run detailed evaluation in this epoch
            run_detailed = (epoch % eval_frequency == 0 or epoch == 1 or epoch == epochs)
            
            # Validation phase
            if fast_mode and not run_detailed:
                # Quick evaluation
                val_loss, val_acc, val_metrics = self.quick_evaluate(val_loader, epoch, prefix='val')
            else:
                # Standard evaluation
                val_loss, val_acc, val_metrics = self.evaluate(val_loader, epoch, prefix='val')
            
            # Store validation metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            # Detailed evaluation
            if not fast_mode and run_detailed:
                # Calculate per-class metrics
                val_class_metrics = self.compute_per_class_metrics(val_loader, epoch, prefix='val')
                history.setdefault('per_class_metrics', []).append(val_class_metrics)
            
            # Test evaluation if test_loader is provided
            if test_loader is not None and (run_detailed or epoch == epochs):
                test_loss, test_acc, test_metrics = self.evaluate(test_loader, epoch, prefix='test')
                
                # Store test metrics
                if epoch % eval_frequency == 0 or epoch == epochs:
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)
                    history['test_f1'].append(test_metrics['f1'])
                    history['test_precision'].append(test_metrics['precision'])
                    history['test_recall'].append(test_metrics['recall'])
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track best model
            improved = False
            
            # Check if current model is better than best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                improved = True
            elif val_acc == best_val_acc and val_loss < best_val_loss:
                # If accuracy is the same, use loss as tie-breaker
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
            
            # Save regular checkpoint every 5 epochs
            if epoch % 5 == 0 and save_best:
                self._save_checkpoint(epoch)
            
            # Early stopping check
            if patience_counter >= patience:
                info(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with validation accuracy {best_val_acc:.4f}.")
                break
        
        # Training completed
        total_time = time.time() - t0
        mins, secs = divmod(total_time, 60)
        hours, mins = divmod(mins, 60)
        
        info(f"Training completed in {int(hours)}h {int(mins)}m {int(secs)}s")
        info(f"Best epoch: {best_epoch} with validation accuracy: {best_val_acc:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            info(f"Restored best model from epoch {best_epoch}")
        
        # Final evaluation on test set
        if test_loader is not None:
            info("Evaluating final model on test set...")
            test_loss, test_acc, test_metrics = self.evaluate(test_loader, epochs, prefix='final')
            
            # Log final test results
            info(f"Final test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            info(f"Final test metrics - F1: {test_metrics['f1']:.4f}, " + 
                f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
            
            # Compute detailed per-class metrics for final model
            if not fast_mode:
                final_class_metrics = self.compute_per_class_metrics(test_loader, epochs, prefix='final')
                
                # Log confusion matrix for final model
                self.log_confusion_matrix(test_loader, epochs, prefix='final')
        
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