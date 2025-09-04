"""
Training pipeline for handwriting recognition models.
Includes training, validation, and model saving functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import os
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """Main training class for handwriting recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        save_dir: str = "saved_models"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to use for training (auto-detect if None)
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = None,
        save_best: bool = True,
        save_frequency: int = 5,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            learning_rate: Learning rate
            optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
            scheduler_type: Type of scheduler ("step", "cosine", "plateau")
            save_best: Whether to save the best model
            save_frequency: Frequency to save checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup scheduler
        scheduler = None
        if scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Optimizer: {optimizer_type}, Learning rate: {learning_rate}")
        if scheduler:
            print(f"Scheduler: {scheduler_type}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            if scheduler:
                if scheduler_type == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epochs"].append(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = self.save_dir / f"best_model_epoch_{epoch+1}.pt"
                self.save_model(self.best_model_path, epoch, val_acc)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_model(checkpoint_path, epoch, val_acc)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, path: Path, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['epochs'], self.history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.history['epochs'], self.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['epochs'], self.history['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(self.history['epochs'], self.history['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_model(
        self, 
        test_loader: DataLoader, 
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation results dictionary
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if return_predictions:
                    all_predictions.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy())
        
        results = {
            'test_loss': total_loss / len(test_loader),
            'test_accuracy': 100. * correct / total,
            'correct_predictions': correct,
            'total_samples': total
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['targets'] = all_targets
        
        return results


class MultiModelTrainer:
    """Train multiple models and compare their performance."""
    
    def __init__(self, save_dir: str = "saved_models"):
        """
        Initialize the multi-model trainer.
        
        Args:
            save_dir: Directory to save all models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.training_histories = {}
    
    def train_multiple_models(
        self,
        models_config: List[Dict[str, Any]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and compare their performance.
        
        Args:
            models_config: List of model configurations
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary with results for each model
        """
        from .models import create_model
        
        for config in models_config:
            model_name = config['name']
            model_type = config['type']
            num_classes = config['num_classes']
            
            print(f"\n{'='*50}")
            print(f"Training {model_name} ({model_type})")
            print(f"{'='*50}")
            
            # Create model
            model = create_model(model_type, num_classes, **config.get('kwargs', {}))
            
            # Create trainer
            trainer = ModelTrainer(
                model, 
                save_dir=self.save_dir / model_name
            )
            
            # Get training kwargs
            training_kwargs = config.get('training_kwargs', {})
            
            # Train model
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                **training_kwargs
            )
            
            # Evaluate model
            test_results = trainer.evaluate_model(test_loader)
            
            # Store results
            self.results[model_name] = {
                'model_info': model.get_model_info(),
                'test_results': test_results,
                'best_val_acc': trainer.best_val_acc,
                'training_time': sum(history.get('epochs', [0]))
            }
            
            self.training_histories[model_name] = history
            
            # Save training plot
            plot_path = self.save_dir / f"{model_name}_training_history.png"
            trainer.plot_training_history(plot_path)
        
        return self.results
    
    def compare_models(self) -> None:
        """Print a comparison of all trained models."""
        if not self.results:
            print("No models have been trained yet!")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            model_info = results['model_info']
            test_results = results['test_results']
            
            comparison_data.append({
                'Model': model_name,
                'Architecture': model_info.get('architecture', 'Unknown'),
                'Parameters': f"{model_info.get('total_parameters', 0):,}",
                'Test Accuracy': f"{test_results['test_accuracy']:.2f}%",
                'Best Val Accuracy': f"{results['best_val_acc']:.2f}%",
                'Training Time': f"{results['training_time']:.1f}s"
            })
        
        # Print table
        for data in comparison_data:
            print(f"{data['Model']:20} | {data['Architecture']:30} | "
                  f"{data['Parameters']:>10} | {data['Test Accuracy']:>12} | "
                  f"{data['Best Val Accuracy']:>15} | {data['Training Time']:>12}")
        
        print("="*80)
    
    def plot_comparison(self, save_path: Optional[Path] = None):
        """Plot comparison of all models."""
        if not self.results:
            print("No models have been trained yet!")
            return
        
        # Prepare data
        model_names = list(self.results.keys())
        test_accuracies = [self.results[name]['test_results']['test_accuracy'] for name in model_names]
        val_accuracies = [self.results[name]['best_val_acc'] for name in model_names]
        param_counts = [self.results[name]['model_info'].get('total_parameters', 0) for name in model_names]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test accuracy comparison
        ax1.bar(model_names, test_accuracies, color='skyblue')
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Validation accuracy comparison
        ax2.bar(model_names, val_accuracies, color='lightcoral')
        ax2.set_title('Best Validation Accuracy Comparison')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        ax3.bar(model_names, param_counts, color='lightgreen')
        ax3.set_title('Model Size Comparison')
        ax3.set_ylabel('Number of Parameters')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        # Accuracy vs Parameters scatter plot
        ax4.scatter(param_counts, test_accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax4.annotate(name, (param_counts[i], test_accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Number of Parameters')
        ax4.set_ylabel('Test Accuracy (%)')
        ax4.set_title('Accuracy vs Model Size')
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
